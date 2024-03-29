from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
import threading
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import platformdirs

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

husk_cache_dir = Path(platformdirs.user_cache_dir("husk", ensure_exists=True))

logger = logging.getLogger("husk")


GZIP_MAGIC = b"\x1f\x8b"


def get_husk_db() -> sqlite3.Connection:
    husk_db = husk_cache_dir / "db.db3"

    db = sqlite3.connect(husk_db)
    db.executescript(
        """
    CREATE TABLE IF NOT EXISTS blobs (
        program TEXT NOT NULL DEFAULT 'docker',
        digest TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        size INT NOT NULL,
        is_gzipped INT NOT NULL,
        gzipped_digest TEXT NULL
    );
    """
    )
    db.commit()

    return db


db = get_husk_db()


class Sha256:
    def __init__(self, sha: str) -> None:
        self.sha = sha.removeprefix("sha256:")

    def without_prefix(self) -> str:
        return self.sha

    def with_prefix(self) -> str:
        return f"sha256:{self.sha}"

    def short(self) -> str:
        return self.sha[:8]

    def __hash__(self) -> int:
        return hash(self.sha)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Sha256) and o.sha == self.sha

    __str__ = with_prefix
    __repr__ = with_prefix


@dataclass(kw_only=True)
class BlobInfo:
    digest: Sha256
    source: str
    size: int
    is_gzipped: bool
    gzipped_digest: Sha256 | None = None


def save_blob_info(blob: BlobInfo) -> None:
    db.execute(
        """
        INSERT INTO blobs (source, digest, size, is_gzipped, gzipped_digest)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING;
        """,
        [
            blob.source,
            blob.digest.with_prefix(),
            blob.size,
            int(blob.is_gzipped),
            blob.gzipped_digest.with_prefix() if blob.gzipped_digest else None,
        ],
    )
    db.commit()


def get_saved_blob_info(digest: Sha256) -> BlobInfo | None:
    blob = db.execute(
        """
        SELECT digest, source, size, is_gzipped, gzipped_digest
        FROM blobs
        WHERE digest=?;
        """,
        [digest.with_prefix()],
    ).fetchone()

    if not blob:
        return None

    blob = BlobInfo(
        digest=Sha256(blob[0]),
        source=blob[1],
        size=blob[2],
        is_gzipped=bool(blob[3]),
        gzipped_digest=Sha256(blob[4]) if blob[4] else None,
    )

    source = blob.source

    if source[0] == "/" and not Path(source).exists():
        db.execute(
            """
            DELETE FROM blobs WHERE digest=?;
            """,
            [digest.with_prefix()],
        )
        db.commit()

        return None

    if blob.gzipped_digest:
        return get_saved_blob_info(blob.gzipped_digest)

    return blob


def done_extracting_message() -> None:
    print(
        """\
Done! Run the following to spin up the container registry:

cd dist && docker compose up\
"""
    )


@dataclass(kw_only=True)
class Settings:
    refs: list[str] = field(default_factory=list)
    dist: Path | None = None


class ExtractContext:
    tmp_dir: Path
    dist: Path
    blobs_dir: Path
    manifests_dir: Path
    manifest_tags: list[str]
    remote_blob_sources: dict[Sha256, str]
    all_blobs: dict[Sha256, BlobInfo]
    max_image_name = 0

    def __init__(self) -> None:
        self.manifest_tags = []
        self.remote_blob_sources = {}
        self.all_blobs = {}

    async def setup(self, settings: Settings) -> None:
        self.tmp_dir = Path("/tmp/husk")  # noqa: S108
        self.tmp_dir.mkdir(exist_ok=True)

        self.dist = settings.dist or Path("dist")

        with suppress(FileNotFoundError):
            shutil.rmtree(self.dist)

        self.dist.mkdir()

        self.blobs_dir = self.dist / "app/blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        self.manifests_dir = self.dist / "app/manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)


DOCKER_REGISTRY_TOKENS: dict[str, str] = {}


async def get_remote_blob(qualified_image_name: str, blob_sha: Sha256) -> BlobInfo:
    assert qualified_image_name.startswith("docker.io/")

    async with httpx.AsyncClient() as client:
        image = qualified_image_name.removeprefix("docker.io/")

        access_token = DOCKER_REGISTRY_TOKENS.get(image)

        if not access_token:
            url = f"https://auth.docker.io/token?scope=repository:{image}:pull&service=registry.docker.io"

            resp = await client.get(url)
            assert resp.status_code == 200

            data = resp.json()
            assert isinstance(data, dict)

            access_token = data["token"]
            assert isinstance(access_token, str)

            DOCKER_REGISTRY_TOKENS[image] = access_token

        resp = await client.get(
            f"https://registry-1.docker.io/v2/{image}/blobs/{blob_sha}",
            follow_redirects=True,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Docker-Distribution-Api-Version": "registry/2.0",
                "Range": "bytes=0-1",
            },
        )
        assert resp.status_code == 206

        size = int(resp.headers["Content-Range"].split("/")[-1])

        is_gzipped = resp.content == GZIP_MAGIC

        blob = BlobInfo(
            digest=blob_sha,
            source=qualified_image_name,
            size=size,
            is_gzipped=is_gzipped,
        )

        save_blob_info(blob)

        return blob


recent_lines: dict[str, str] = {}


def cleanup() -> None:
    for _ in range(len(recent_lines)):
        print("\x1b[1A\x1b[2K", end="")


@contextmanager
def extractor_log_wrapper() -> Iterator[None]:
    yield

    cleanup()


image_name_ctx: ContextVar[str | None] = ContextVar("image_name", default=None)
writing = threading.Lock()


class PrettyPrintHandler(logging.Handler):
    max_len: int
    preserve: bool

    def __init__(
        self,
        level: logging._Level = logging.NOTSET,
        *,
        max_len: int,
        preserve: bool = False,
    ) -> None:
        super().__init__(level)

        self.max_len = max_len
        self.preserve = preserve

    def emit(self, record: logging.LogRecord) -> None:
        if image_name := image_name_ctx.get():
            msg = self.format(record)

            if self.preserve:
                print(f"{image_name:{self.max_len}} | {msg}")
                return

            with writing:
                cleanup()

                recent_lines[image_name] = msg

                for image, line in recent_lines.items():
                    print(f"{image:{self.max_len}} | {line}")


def setup_logging(ctx: ExtractContext) -> None:
    handler = PrettyPrintHandler(max_len=ctx.max_image_name)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.propagate = False


class BlobCache:
    blobs: dict[Sha256, ComputedBlob]

    def __init__(self) -> None:
        self.blobs = {}

    async def get(self, blob_sha: Sha256) -> tuple[bool, ComputedBlob]:
        """
        Return a blob from the cache. If it is already being computed, return True as the first
        argument, otherwise return False.
        """

        blob = self.blobs.get(blob_sha)

        if blob:
            return True, blob

        computed_blob = ComputedBlob()
        self.blobs[blob_sha] = computed_blob
        return False, computed_blob


class ComputedBlob:
    blob: BlobInfo | None = None

    blob_ready: asyncio.Event

    def __init__(self) -> None:
        self.blob_ready = asyncio.Event()

    def __await__(self) -> Generator[None, None, BlobInfo]:
        if self.blob is not None:
            return self.blob

        yield from self.blob_ready.wait().__await__()

        assert self.blob
        return self.blob

    @property
    def finished(self) -> bool:
        return self.blob_ready.is_set()

    def finish(self, blob: BlobInfo) -> None:
        self.blob = blob
        self.blob_ready.set()


# TODO: run all IO concurrently
def build_dist_folder(ctx: ExtractContext) -> None:
    with (ctx.blobs_dir / "remote_blobs.map").open("w+") as f:
        for digest, source in ctx.remote_blob_sources.items():
            f.write(
                f"~{digest.without_prefix()}$ {source.removeprefix('docker.io/library/')};\n"
            )

    with (ctx.manifests_dir / "tags.map").open("w+") as f:
        f.writelines(set(ctx.manifest_tags))

    for blob in ctx.all_blobs.values():
        blob_source = Path(blob.source)

        if blob_source.is_absolute():
            assert blob_source.exists()

            shutil.copyfile(blob_source, ctx.blobs_dir / blob.digest.without_prefix())

    data_dir = Path(__file__).parent.parent / "data"

    shutil.copyfile(data_dir / "nginx.conf", ctx.dist / "nginx.conf")
    shutil.copyfile(data_dir / "docker-compose.yml", ctx.dist / "docker-compose.yml")


def remove_image_domain_name(image_name: str) -> str:
    image_name_parts = image_name.split("/")

    if "." in image_name_parts[0]:
        return "/".join(image_name_parts[1:])

    return image_name

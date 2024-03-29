from __future__ import annotations

import asyncio
import fnmatch
import gzip
import json
import logging
import os
import shutil
import sys
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from husk.extract.common import (
    BlobCache,
    BlobInfo,
    ExtractContext,
    Settings,
    Sha256,
    build_dist_folder,
    done_extracting_message,
    extractor_log_wrapper,
    get_remote_blob,
    get_saved_blob_info,
    image_name_ctx,
    remove_image_domain_name,
    save_blob_info,
    setup_logging,
)

logger = logging.getLogger("husk")


async def list_images_for_reference(ref: str) -> list[dict[str, Any]]:
    process = await asyncio.subprocess.create_subprocess_exec(
        "docker",
        "image",
        "ls",
        "--no-trunc",
        "--format=json",
        "--digests",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await process.wait()

    assert process.returncode == 0
    assert process.stdout

    lines = await process.stdout.read()

    def known_name_matches_ref(image: dict[str, Any], ref: str) -> bool:
        return fnmatch.fnmatch(image["Repository"], ref)

    images = [json.loads(line) for line in lines.splitlines()]

    return [image for image in images if known_name_matches_ref(image, ref)]


async def inspect_image(ref: str) -> list[dict[str, Any]]:
    process = await asyncio.subprocess.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        ref,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await process.wait()

    assert process.returncode == 0
    assert process.stdout

    return json.loads(await process.stdout.read())


def get_blob_source_from_layer(
    ctx: DockerContext, sha: Sha256
) -> tuple[str, Sha256] | None:
    # TODO: figure out if there is an API to pull this info
    diff_dir = (
        ctx.docker_root_dir / "image/overlay2/distribution/v2metadata-by-diffid/sha256/"
    )

    file = diff_dir / sha.without_prefix()

    if not file.exists():
        return None

    blobs = json.loads(file.read_text())

    for blob in blobs:
        source = blob["SourceRepository"]
        digest = Sha256(blob["Digest"])

        if not blob["HMAC"] and source.startswith("docker.io/"):
            ctx.remote_blob_sources[digest] = source

            return source, digest

    return None


def get_container_config_blob(ctx: DockerContext, sha: Sha256) -> BlobInfo:
    if blob := get_saved_blob_info(sha):
        return blob

    config_dir = ctx.docker_root_dir / "image/overlay2/imagedb/content/sha256/"

    config_file = config_dir / sha.without_prefix()
    copied_blob = ctx.dist / "app/blobs" / sha.without_prefix()

    shutil.copyfile(config_file, copied_blob)

    blob = BlobInfo(
        digest=sha,
        source=str(config_file),
        size=copied_blob.stat().st_size,
        is_gzipped=False,
    )

    save_blob_info(blob)

    return blob


blob_cache = BlobCache()


async def extract_blob_from_local_image(
    ctx: DockerContext,
    image: str,
    blob_sha: Sha256,
    cache_key: str,
) -> BlobInfo:
    being_computed, computed_blob = await blob_cache.get(blob_sha)

    if being_computed:
        return await computed_blob

    image_tar = ctx.tmp_dir / "images" / image / f"{cache_key}.tar"

    (ctx.tmp_dir / "blobs").mkdir(exist_ok=True, parents=True)
    image_tar.parent.mkdir(exist_ok=True, parents=True)

    if not image_tar.exists():
        logger.info("Exporting to tar file")

        process = await asyncio.subprocess.create_subprocess_exec(
            "docker",
            "save",
            image,
            "-o",
            str(image_tar),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        assert process.returncode == 0

    with tarfile.open(image_tar, "r") as tar:
        member = tar.getmember(f"blobs/sha256/{blob_sha.without_prefix()}")

        blob_file = tar.extractfile(member)
        assert blob_file

        # TODO: what if blob is already gzipped?
        logger.info("Extracting and compressing blob %s", blob_sha.short())

        gzip_file = ctx.tmp_dir / f"blobs/{blob_sha.without_prefix()}.tar.gz"

        def zip_blob() -> None:
            with gzip.GzipFile(gzip_file, "wb", compresslevel=9, mtime=0) as tar_gz:
                while chunk := blob_file.read(1024 * 1024):
                    tar_gz.write(chunk)

        await asyncio.get_running_loop().run_in_executor(None, zip_blob)

        with Path(gzip_file).open("rb") as tar_gz:  # noqa: ASYNC101
            # TODO: turn this into a classmethod
            gzipped_digest = Sha256(sha256(tar_gz.read()).hexdigest())

        logger.debug(
            "Blob %s is now %s after compressing",
            blob_sha.short(),
            gzipped_digest.short(),
        )

        gzip_file = gzip_file.rename(
            gzip_file.with_name(gzipped_digest.without_prefix())
        )

        uncompressed = member.size
        compressed = gzip_file.stat().st_size

        blob = BlobInfo(
            source=str(gzip_file),
            digest=blob_sha,
            size=uncompressed,
            is_gzipped=False,
            gzipped_digest=gzipped_digest,
        )
        save_blob_info(blob)

        gzipped_blob = BlobInfo(
            source=str(gzip_file),
            digest=gzipped_digest,
            size=compressed,
            is_gzipped=True,
        )
        save_blob_info(gzipped_blob)

        computed_blob.finish(gzipped_blob)

        return gzipped_blob


async def extract_references(
    ctx: DockerContext, ref: str
) -> list[tuple[str, asyncio.Task[None]]]:
    images = await list_images_for_reference(ref)

    if not images:
        print(f"Warning, no images match ref `{ref}`")
        return []

    return [
        (get_full_image_name(image), asyncio.create_task(extract_image(ctx, image)))
        for image in images
    ]


def get_full_image_name(image: dict[str, Any]) -> str:
    image_name = image["Repository"]
    tag = None if image["Tag"] == "<none>" else image["Tag"]

    if tag:
        return f"{image_name}:{tag}"

    return f"{image_name}@{image['Digest']}"


async def extract_image(ctx: DockerContext, image: dict[str, Any]) -> None:
    # TODO: turn image into dataclass
    image_name = image["Repository"]
    tag = image["Digest"] if image["Tag"] == "<none>" else image["Tag"]

    full_image_name = get_full_image_name(image)

    image_name_ctx.set(full_image_name)

    logger.info("Extracting")

    data = await inspect_image(full_image_name)
    rootfs_layers: list[Sha256] = [
        Sha256(layer) for layer in data[0]["RootFS"]["Layers"]
    ]

    assert rootfs_layers

    config_blob = get_container_config_blob(ctx, Sha256(image["ID"]))

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": Sha256(image["ID"]).with_prefix(),
            "size": config_blob.size,
        },
    }

    layers = []

    for layer in rootfs_layers:
        blob = get_saved_blob_info(layer)

        if not blob:
            if tmp := get_blob_source_from_layer(ctx, layer):
                source, digest = tmp

                blob = get_saved_blob_info(digest)

                if not blob:
                    assert source.startswith("docker.io/")

                    blob = await get_remote_blob(source, digest)

            if not blob:
                blob = await extract_blob_from_local_image(
                    ctx, image_name, layer, image["ID"]
                )

        ctx.all_blobs[blob.digest] = blob

        suffix = "+gzip" if blob.is_gzipped else ""
        media_type = f"application/vnd.oci.image.layer.v1.tar{suffix}"

        layers.append(
            {
                "mediaType": media_type,
                "digest": blob.digest.with_prefix(),
                "size": blob.size,
            }
        )

    manifest["layers"] = layers

    minified_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_digest = sha256(minified_manifest.encode()).hexdigest()

    logger.info("Done")

    image_name = remove_image_domain_name(image_name)

    ctx.manifest_tags.append(
        f"/v2/{image_name}/manifests/{tag} /v2/{image_name}/manifests/sha256:{manifest_digest};\n"
    )

    shutil.copyfile(
        config_blob.source, ctx.blobs_dir / config_blob.digest.without_prefix()
    )

    manifest_repo_dir = ctx.manifests_dir / image_name
    manifest_repo_dir.mkdir(parents=True, exist_ok=True)
    (manifest_repo_dir / f"sha256:{manifest_digest}").write_text(minified_manifest)


async def main(settings: Settings) -> None:
    ctx = DockerContext()
    await ctx.setup(settings)

    extractors: list[tuple[str, asyncio.Task[None]]] = []

    for ref in settings.refs:
        extractors.extend(await extract_references(ctx, ref))

    if not extractors:
        sys.exit(1)

    ctx.max_image_name = max(len(x[0]) for x in extractors)

    setup_logging(ctx)

    with extractor_log_wrapper():
        await asyncio.gather(*[x[1] for x in extractors])

    build_dist_folder(ctx)

    process = await asyncio.subprocess.create_subprocess_exec(
        "chown",
        "-R",
        f"{os.getenv('SUDO_UID', '1000')}:{os.getenv('SUDO_GID', '1000')}",
        str(ctx.dist),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()

    done_extracting_message()


class DockerContext(ExtractContext):
    docker_root_dir: Path

    async def setup(self, settings: Settings) -> None:
        await super().setup(settings)

        self.docker_root_dir = await self.get_root_dir()

    @staticmethod
    async def get_root_dir() -> Path:
        process = await asyncio.subprocess.create_subprocess_exec(
            "docker",
            "system",
            "info",
            "--format=json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await process.wait()

        assert process.returncode == 0
        assert process.stdout

        line = await process.stdout.read()
        data = json.loads(line)

        driver = data["Driver"]

        if driver != "overlay2":
            print(
                f"Driver `{driver}` is not supported. Only `overlay2` is supported at this time."
            )
            sys.exit(1)

        return Path(data["DockerRootDir"])

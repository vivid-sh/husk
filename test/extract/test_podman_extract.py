import json
import shutil
import subprocess
from pathlib import Path
from tempfile import mkdtemp

from husk import main


class TestPodmanExtract:
    test_image_name = "husk-testing-image-build"
    dist: Path

    @classmethod
    def setup_class(cls) -> None:
        cls.dist = Path(mkdtemp())

        cls.build_test_image()

        # TODO: allow for changing dist folder location
        cls.extract_test_image()

    @classmethod
    def teardown_class(cls) -> None:
        shutil.rmtree(cls.dist)

    def test_dist_folder_structure_is_correct(self) -> None:
        assert self.dist.exists()
        assert (self.dist / "docker-compose.yml").exists()
        assert (self.dist / "nginx.conf").exists()
        assert (self.dist / "app/blobs").exists()
        assert (self.dist / "app/manifests").exists()
        assert (self.dist / "app/manifests" / self.test_image_name).exists()

    def test_image_manifest_is_correct(self) -> None:
        manifest_dir = self.dist / "app/manifests" / self.test_image_name
        manifests = list(manifest_dir.iterdir())

        assert len(manifests) == 1

        manifest_file = manifests[0]

        tags = (self.dist / "app/manifests/tags.map").read_text().splitlines()

        assert len(tags) == 1
        assert "latest" in tags[0]
        assert manifest_file.name in tags[0]

        manifest = json.loads(manifest_file.read_text())

        config_blob = manifest["config"]["digest"]
        config_blob_size = manifest["config"]["size"]
        config_blob_file = self.dist / "app/blobs" / config_blob.removeprefix("sha256:")

        assert config_blob_file.exists()
        assert config_blob_file.stat().st_size == config_blob_size

        remote_blobs = (self.dist / "app/blobs/remote_blobs.map").read_text()

        for layer in manifest["layers"]:
            blob = layer["digest"].removeprefix("sha256:")

            blob_file = self.dist / "app/blobs" / blob

            if blob_file.exists():
                assert blob_file.stat().st_size

            else:
                assert blob in remote_blobs

    @classmethod
    def build_test_image(cls) -> None:
        image_build_dir = Path(__file__).parent / "data"

        process = subprocess.run(
            [
                "podman",
                "build",
                "-f",
                str(image_build_dir / "Dockerfile"),
                str(image_build_dir),
                "-t",
                cls.test_image_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(process.stdout)

        assert process.returncode == 0

    @classmethod
    def extract_test_image(cls) -> None:
        main(["extract", "--podman", cls.test_image_name, "-o", str(cls.dist)])

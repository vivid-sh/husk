import asyncio
import os
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from husk.extract.common import Settings
from husk.extract.docker import main as docker_extract
from husk.extract.podman import main as podman_extract


def usage() -> None:
    print(
        """\
usage: husk [-h | --help]
       husk extract [--docker] [--podman] ref [refs ...]

examples:
       # extract a single image
       husk extract my-example-image:v1

       # extract all tags for a given image
       husk extract 'my-example-image:*'

       # extract all images for a tagged domain
       husk extract 'example.com/*'

       # extract multiple image refs
       husk extract 'example.com/*' 'example.org/*'
"""
    )

    sys.exit(1)


def main(argv: Sequence[str]) -> None:
    parser = ArgumentParser()

    extract_sub_parser = parser.add_subparsers().add_parser("extract")

    extract_sub_parser.add_argument("--docker", action="store_true")
    extract_sub_parser.add_argument("--podman", action="store_true")
    extract_sub_parser.add_argument("--output-dir", "-o")
    extract_sub_parser.add_argument("refs", nargs="*")

    args = parser.parse_args(argv)

    if not hasattr(args, "refs"):
        usage()

    if not args.refs:
        print("Error, expected one or more image refs\n")
        usage()

    use_docker = not args.podman

    settings = Settings(refs=args.refs)

    if args.output_dir:
        settings.dist = Path(args.output_dir)

    if use_docker:
        if os.geteuid() != 0:
            print("Husk must be ran as root when using the docker extractor!")
            sys.exit(1)

        asyncio.run(docker_extract(settings))

    else:
        asyncio.run(podman_extract(settings))


def cli_main() -> None:
    main(sys.argv[1:])

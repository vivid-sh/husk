.PHONY: typos mypy ruff isort black test

all: typos mypy ruff isort black test

typos:
	typos --format brief

ruff:
	ruff check husk test --output-format=concise

mypy:
	mypy -p husk
	mypy -p test

isort:
	isort . --diff --check

black:
	black husk test --check --diff

test:
	sudo pytest

fmt:
	ruff check husk test --fix
	isort .
	black husk test

.PHONY: check lint fmt type test

check: lint type test

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type:
	uv run mypy aura tests

test:
	uv run pytest -v

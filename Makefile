.PHONY: check lint fmt type test e2e desktop desktop-build desktop-yes

check: lint type test

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type:
	uv run mypy aura tests

test:
	uv run pytest -v

e2e:
	uv run pytest -m e2e -v

# One-shot desktop launcher (interactive — prompts before installing deps)
desktop:
	@bash desktop/run.sh

# Same as above but auto-installs missing deps without asking
desktop-yes:
	@bash desktop/run.sh --yes

# Production build (.app / .dmg / .exe)
desktop-build:
	@bash desktop/run.sh --build

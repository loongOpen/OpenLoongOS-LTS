.PHONY: sync
sync:
	uv sync --extra cu128 --group dev

.PHONY: sync-cpu
sync-cpu:
	uv sync --extra cpu --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: test
test:
	uv run pytest

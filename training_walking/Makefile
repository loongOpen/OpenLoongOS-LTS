UV_EXTRA ?= cu128

.PHONY: sync
sync:
	uv sync --locked --extra $(UV_EXTRA) --group dev

.PHONY: sync-cpu
sync-cpu:
	uv sync --locked --extra cpu --group dev

.PHONY: format
format:
	uv run --locked --extra $(UV_EXTRA) --group dev ruff format
	uv run --locked --extra $(UV_EXTRA) --group dev ruff check --fix

.PHONY: test
test:
	uv run --locked --extra $(UV_EXTRA) --group dev pytest

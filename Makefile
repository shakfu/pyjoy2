.PHONY: all sync repl build test format lint typecheck clean


all: sync


sync:
	@uv sync


repl: sync
	@uv run python -m pyjoy2


build: sync
	@uv build


test: sync
	@uv run pytest


format:
	@uv run ruff format .


lint:
	@uv run ruff check --fix src/


typecheck:
	@uv run ty check src/


clean:
	@rm -rf dist


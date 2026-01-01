.PHONY: all sync repl build test clean


all: sync


sync:
	@uv sync


repl: sync
	@uv run python -m pyjoy2


build: sync
	@uv build


test: sync
	@uv run pytest


clean:
	@rm -rf dist


.PHONY: all sync repl build test coverage cov-html format lint typecheck profile profile-stats clean


all: sync


sync:
	@uv sync


repl: sync
	@uv run python -m pyjoy2


build: sync
	@uv build


test: sync
	@uv run pytest


coverage: sync
	@uv run pytest --cov=src/pyjoy2 --cov-report=term-missing


cov-html: sync
	@uv run pytest --cov=src/pyjoy2 --cov-report html:cov_html


format:
	@uv run ruff format .


lint:
	@uv run ruff check --fix src/


typecheck:
	@uv run ty check src/


# Profiling targets
# Usage: make profile EXPR="[1 2 3 4 5] [dup *] map"
#        make profile FILE=examples/benchmark.joy
EXPR ?= 1000 [1 -] [dup 0 >] while pop
FILE ?=

profile: sync
	@if [ -n "$(FILE)" ]; then \
		uv run python -m cProfile -s cumtime -m pyjoy2 "$(FILE)" 2>&1 | head -50; \
	else \
		uv run python -m cProfile -s cumtime -m pyjoy2 -e '$(EXPR)' 2>&1 | head -50; \
	fi

profile-stats: sync
	@if [ -n "$(FILE)" ]; then \
		uv run python -m cProfile -o profile.stats -m pyjoy2 "$(FILE)"; \
	else \
		uv run python -m cProfile -o profile.stats -m pyjoy2 -e '$(EXPR)'; \
	fi
	@echo "Profile saved to profile.stats"
	@echo "View with: uv run python -c \"import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(30)\""


clean:
	@rm -rf dist profile.stats


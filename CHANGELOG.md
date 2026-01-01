# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New list operations for quicksort support:
  - `small` - test if list has 0 or 1 elements
  - `partition` - split list by predicate: `[...] [P] -> [yes] [no]`
  - `enconcat` - concatenate with element in middle: `X [A] [B] -> [A X B...]`
- Recursive word definitions now work correctly with `.def`
- CLI script entry point: `uv run pyjoy2`

### Fixed

- `.def` now supports recursive definitions by using forward references

## [0.1.0] - 2026-01-01

### Added

- Initial release of pyjoy2
- Core stack implementation (`Stack` class) supporting any Python object
- Word registry system with `WORDS` dictionary
- Two decorators for defining words:
  - `@word` - auto-pops arguments, pushes return value
  - `@define(name)` - direct stack manipulation
- Comprehensive set of ~140 builtin words:
  - Stack operations: `dup`, `pop`, `swap`, `over`, `rot`, `nip`, `tuck`, `dupd`, `swapd`, `rollup`, `rolldown`, `clear`, `stack`, `unstack`, `depth`
  - Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `neg`, `abs`, `sign`, `min`, `max`, `succ`, `pred`, `pow`
  - Comparison: `<`, `<=`, `>`, `>=`, `=`, `!=`, `cmp`
  - Logic: `not`, `and`, `or`, `xor`
  - List operations: `first`, `rest`, `cons`, `uncons`, `concat`, `size`, `null`, `reverse`, `at`, `take`, `drop`, `range`, `list`, `split`, `join`, `sort`, `sum`, `prod`
  - Combinators: `i`, `x`, `dip`, `dipd`, `keep`, `bi`, `tri`, `ifte`, `branch`, `when`, `unless`, `loop`, `while`, `times`, `each`, `map`, `filter`, `fold`, `any`, `all`, `zip`, `enumerate`
  - Recursion: `linrec`, `binrec`, `primrec`
  - I/O: `print`, `.`, `puts`, `show`, `input`, `prompt`
  - Type operations: `type`, `int`, `float`, `str`, `bool`, `repr`
  - Misc: `id`, `apply`, `compose`, `curry`, `dup2`, `pop2`
- Tokenizer and parser supporting:
  - Numbers (integers, floats, negative, scientific notation)
  - Strings with escape sequences
  - Booleans (`true`, `false`)
  - Nil/null values
  - Quotations (nested lists)
  - Set literals
  - Comments (`(* ... *)` and `# ...`)
- Hybrid REPL with Python integration:
  - Backtick syntax for Python expressions: `` `expr` ``
  - Dollar syntax alternative: `$(expr)`
  - Bang syntax for Python statements: `!stmt`
  - Multi-line Python blocks (def, class, etc.)
  - REPL commands: `.s`, `.c`, `.w`, `.def`, `.import`, `.load`, `.help`
  - Pre-imported modules: `math`, `json`, `os`, `re`
- Command-line interface:
  - Interactive REPL mode
  - Expression evaluation (`-e` / `--eval`)
  - File execution
  - Debug mode (`--debug`)
- `run()` function for programmatic execution
- Full test suite with 265 tests
- Type annotations throughout

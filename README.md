# pyjoy2

A Pythonic re-imagining of the the Joy language

Package Structure (pyjoy2/):

- `__init__.py `- Package exports and version
- `__main__.py` - CLI entry point
- `core.py` - Stack class, word registry, decorators
- `parser.py` - Tokenizer and parser
- `builtins.py` - 142 builtin words
- `repl.py` - Hybrid REPL with Python integration

## Features:

- Stack operations: `dup`, `swap`, `rot`, `over`, `nip`, `tuck`, etc.
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `neg`, `abs`, `min`, `max`
- Comparison: `<`, `<=,` `>`, `>=`, `=`, `!=`
- List operations: `first`, `rest`, `cons`, `concat`, `map`, `filter`, `fold`
- Combinators: `i`, `dip`, `ifte`, `linrec`, `binrec`, `times`, `each`
- Python integration: `expr`, `$(expr)`, `!stmt` syntax
- Custom word definition: `@word`, `@define`, `.def name [...]`

## Usage:

From the commandline

```sh
python -m pyjoy2              # Start REPL
python -m pyjoy2 -e "3 4 +"   # Evaluate expression
python -m pyjoy2 file.joy     # Run file
```

As a python3 module

```python
from pyjoy2 import run, word

result = run("3 4 + dup *")   # Stack([49])

@word
def square(x):
  return x * x

run("5 square")               # Stack([25])
```


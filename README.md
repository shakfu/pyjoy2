# pyjoy2

A Pythonic reimagining of the Joy programming language.

PyJoy2 is a concatenative, stack-based functional language that seamlessly integrates with Python. Any Python object can live on the stack, and you can freely mix Joy-style functional programming with Python expressions.

## Installation

```sh
# Clone and install
git clone <repository-url>
cd pyjoy2
uv sync
```

## Quick Start

### Command Line

```sh
# Start the interactive REPL
python -m pyjoy2

# Evaluate an expression
python -m pyjoy2 -e "3 4 + dup *"

# Run a file
python -m pyjoy2 program.joy
```

### As a Python Module

```python
from pyjoy2 import run, word, define, Stack

# Run a program
result = run("3 4 + dup *")
print(result)  # Stack([49])

# Define custom words with the @word decorator
@word
def square(x):
    return x * x

run("5 square")  # Stack([25])

# Define words that manipulate the stack directly
@define("double")
def _double(stack):
    x = stack.pop()
    stack.push(x * 2)

run("7 double")  # Stack([14])
```

## REPL Usage

The REPL supports both Joy syntax and Python integration:

```
> 3 4 +
7
> dup *
49
> [1 2 3] [dup *] map
[1, 4, 9]
```

### Python Integration

```
> `math.sqrt(16)`           # Backtick: evaluate Python expression
4.0
> $(2 ** 10)                # Dollar syntax: alternative for expressions
1024
> !import sys               # Bang: execute Python statement
> !x = 42                   # Define Python variables
> `x * 2`
84
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `.s` | Show the stack with types |
| `.c` | Clear the stack |
| `.w` | List all available words |
| `.def name [body]` | Define a new word |
| `.import module` | Import a Python module |
| `.load file` | Load and execute a Joy file |
| `.help` | Show help |
| `quit` | Exit the REPL |

## Language Features

### Stack Operations

```
dup     # X -> X X         Duplicate top
pop     # X ->             Remove top
swap    # X Y -> Y X       Exchange top two
over    # X Y -> X Y X     Copy second to top
rot     # X Y Z -> Y Z X   Rotate top three
nip     # X Y -> Y         Remove second
tuck    # X Y -> Y X Y     Copy top under second
```

### Arithmetic

```
+  -  *  /  //  %          # Basic operations
neg  abs  sign             # Unary operations
min  max  pow              # Binary operations
succ  pred                 # Increment/decrement
```

### Comparison and Logic

```
<  <=  >  >=  =  !=        # Comparison (return boolean)
not  and  or  xor          # Logical operations
```

### List Operations

```
first    # [X ...] -> X           First element
rest     # [X ...] -> [...]       All but first
cons     # X [...] -> [X ...]     Prepend
concat   # [...] [...] -> [...]   Concatenate
size     # [...] -> N             Length
reverse  # [...] -> [...]         Reverse
```

### Combinators

```
i        # [P] -> ...             Execute quotation
dip      # X [P] -> ... X         Execute under top
keep     # X [P] -> ... X         Execute and restore
ifte     # [B] [T] [F] -> ...     If-then-else
times    # N [P] -> ...           Execute N times
map      # [...] [P] -> [...]     Transform each element
filter   # [...] [P] -> [...]     Keep matching elements
fold     # [...] V [P] -> V       Reduce with initial value
```

### Recursion Combinators

```
linrec   # [P] [T] [R1] [R2] -> ...   Linear recursion
binrec   # [P] [T] [R1] [R2] -> ...   Binary recursion
primrec  # N [I] [C] -> ...           Primitive recursion
```

## Examples

### Factorial

```
# Using linrec
5 [0 =] [pop 1] [dup 1 -] [*] linrec
# Result: 120
```

### Quicksort-style operations

```
[3 1 4 1 5 9 2 6] sort
# Result: [1, 1, 2, 3, 4, 5, 6, 9]
```

### Functional pipelines

```
10 range [2 %] filter [dup *] map 0 [+] fold
# Generate 0-9, keep odd, square each, sum
# Result: 165
```

### Mixing Python and Joy

```
`[x**2 for x in range(10)]` [25 <] filter
# Result: [0, 1, 4, 9, 16]
```

## Project Structure

```
pyjoy2/
  __init__.py    # Package exports
  __main__.py    # CLI entry point
  core.py        # Stack, WORDS registry, decorators
  parser.py      # Tokenizer and parser
  builtins.py    # ~140 builtin words
  repl.py        # Hybrid REPL
```

## Development

```sh
make sync       # Install dependencies
make test       # Run tests
make lint       # Run linter
make typecheck  # Run type checker
make repl       # Start REPL
```

## License

See [LICENSE](LICENSE) for details.

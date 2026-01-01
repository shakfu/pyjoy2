# Comparison: pyjoy2 vs Joy

This document compares pyjoy2 with the original Joy programming language, highlighting similarities, differences, and extensions.

## Overview

| Aspect | Joy | pyjoy2 |
|--------|-----|--------|
| Creator | Manfred von Thun | - |
| Implementation | C | Python |
| Type System | Dynamic, Joy-specific types | Dynamic, any Python object |
| Interop | Limited | Full Python integration |
| REPL | Basic | Hybrid Joy/Python |

## Syntax Comparison

### Identical Syntax

Most core Joy syntax is preserved in pyjoy2:

```
(* Comments *)        # Both support Joy-style comments
# Line comments       # pyjoy2 also supports Python-style

5 dup *               # Stack operations
[1 2 3] first         # List operations
[dup *] map           # Combinators
```

### Literals

| Type | Joy | pyjoy2 |
|------|-----|--------|
| Integers | `42`, `-7` | `42`, `-7` |
| Floats | `3.14` | `3.14`, `1e-5` |
| Strings | `"hello"` | `"hello"` (with escapes) |
| Booleans | `true`, `false` | `true`, `false` |
| Null | - | `nil`, `null` |
| Lists | `[1 2 3]` | `[1 2 3]` |
| Sets | `{1 2 3}` | `{1 2 3}` |
| Characters | `'a` | Not supported |

### Quotations

Both use square brackets for quotations (unevaluated programs):

```
[dup *]         # A quotation
i               # Execute quotation
```

## Semantic Differences

### Stack as Python List

In pyjoy2, the stack is a Python list subclass. This means:

```
# Any Python object can be on the stack
`datetime.now()`    # Push datetime object
`{"key": "value"}`  # Push dict
`lambda x: x*2`     # Push function
```

### List Operations

Joy uses `first`/`rest` terminology; pyjoy2 supports both Joy and Python conventions:

| Operation | Joy | pyjoy2 |
|-----------|-----|--------|
| First element | `first` | `first` |
| Remaining | `rest` | `rest` |
| Prepend | `cons` | `cons` |
| Append | - | via Python |
| Length | `size` | `size`, `null` |

### Type Coercion

pyjoy2 is more permissive due to Python's duck typing:

```
"3" "4" +           # Joy: type error, pyjoy2: "34"
[1 2] [3 4] +       # Joy: undefined, pyjoy2: [1, 2, 3, 4]
```

## Python Integration (pyjoy2 only)

### Expressions

```
`math.sqrt(16)`     # Backtick: evaluate Python, push result
$(2 ** 10)          # Dollar syntax: alternative
```

### Statements

```
!import pandas as pd
!x = 42
!def helper(n): return n * 2
```

### Mixed Programs

```
`range(10)` list [dup *] map     # Generate with Python, process with Joy
[1 2 3] `sum(S.pop())`           # Process with Joy, consume with Python
```

## Word Comparison

### Stack Operations

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `dup` | X -> X X | Same | |
| `pop` | X -> | Same | |
| `swap` | X Y -> Y X | Same | |
| `over` | X Y -> X Y X | Same | |
| `rot` | X Y Z -> Y Z X | Same | |
| `rollup` | X Y Z -> Z X Y | Same | |
| `rolldown` | X Y Z -> Y Z X | Same as `rot` |
| `dup2` | X Y -> X Y X Y | Same | |
| `pop2` | X Y -> | Same | |
| `dupd` | X Y -> X X Y | Same | |
| `swapd` | X Y Z -> Y X Z | Same | |
| `nip` | X Y -> Y | Same | |
| `tuck` | X Y -> Y X Y | Same | |
| `depth` | -> N | Same | Stack depth |
| `clear` | ... -> | Same | |
| `stack` | ... -> ... [...] | Same | Copy stack to list |
| `unstack` | [...] -> ... | Same | Replace stack |

### Arithmetic

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `+`, `-`, `*` | Same | Same | |
| `/` | Integer division | Float division | Python 3 semantics |
| `div` | Integer division | Not implemented | Use `//` |
| `//` | Not in Joy | Floor division | Python operator |
| `%`, `mod` | Modulo | `%` | |
| `neg`, `abs` | Same | Same | |
| `succ`, `pred` | Same | Same | +1, -1 |
| `max`, `min` | Same | Same | |
| `pow` | - | X Y -> X^Y | Python's `**` |
| `sign` | Same | Same | -1, 0, or 1 |

### Comparison

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `<`, `<=`, `>`, `>=` | Same | Same | |
| `=` | Equality | Same | Not assignment |
| `!=` | Inequality | Same | |
| `cmp` | - | X Y -> -1/0/1 | Three-way compare |

### Logic

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `true`, `false` | Same | Same | Literals |
| `not` | Same | Same | |
| `and`, `or` | Same | Same | |
| `xor` | - | Same | |

### List Operations

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `first` | Same | Same | |
| `rest` | Same | Same | |
| `cons` | Same | Same | |
| `uncons` | Same | Same | Pushes two values |
| `concat` | Same | Same | |
| `size` | Same | Same | |
| `null` | Same | Same | Is empty? |
| `reverse` | Same | Same | |
| `at` | Same | Same | Index access |
| `take`, `drop` | Same | Same | |
| `split` | Same | Same | |
| `join` | - | [...] S -> S | String join |
| `sort` | - | Same | |
| `small` | Same | Same | 0 or 1 elements |
| `sum`, `prod` | - | Same | |
| `enconcat` | Same | Same | X [A] [B] -> [A X B] |

### Combinators

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `i` | Same | Same | Execute quotation |
| `x` | Same | Same | `dup i` |
| `dip` | Same | Same | Execute under top |
| `dipd` | Same | Same | Execute under two |
| `keep` | Same | Same | Execute and restore |
| `bi` | Same | Same | Apply two quotations |
| `tri` | Same | Same | Apply three quotations |
| `ifte` | Same | Same | If-then-else |
| `branch` | Same | Same | Boolean dispatch |
| `when` | Same | Same | Conditional execute |
| `unless` | Same | Same | Negated when |
| `loop` | Same | Same | While top is true |
| `while` | Same | Same | Condition + body |
| `times` | Same | Same | N iterations |
| `map` | Same | Same | Transform list |
| `filter` | Same | Same | Select from list |
| `fold` | Same | Same | Reduce with initial |
| `each` | Same | Same | Execute for each |
| `any`, `all` | - | Same | Short-circuit |
| `zip` | - | Same | Pair lists |
| `enumerate` | - | Same | With indices |
| `partition` | `split` | Same | Split by predicate |

### Recursion Combinators

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `linrec` | Same | Same | Linear recursion |
| `binrec` | Same | Same | Binary recursion |
| `primrec` | Same | Same | Primitive recursion |
| `genrec` | In Joy | Not yet | General recursion |
| `tailrec` | In Joy | Not yet | Tail recursion |

### I/O

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `print` | Print + newline | Same | |
| `.` | Print top | Same | Dot command |
| `puts` | Print string | Same | No newline |
| `show` | - | Format value | |
| `input` | Read line | Same | |
| `prompt` | - | Read with prompt | |
| File I/O | Various | Via Python | Use backticks |

### Type Operations

| Operation | Joy | pyjoy2 | Notes |
|-----------|-----|--------|-------|
| `type` | Type name | Same | Returns string |
| `int`, `float`, `str` | Conversion | Same | |
| `bool` | - | Same | To boolean |
| `repr` | - | Same | Python repr |

## Missing Joy Features

The following Joy features are not yet implemented in pyjoy2:

1. **Additional recursion combinators**: `genrec`, `tailrec`, `condlinrec`
2. **Tree operations**: `treemap`, `treegenrec`
3. **Set operations**: Only basic set literals, no set-specific words
4. **File I/O words**: Use Python integration instead
5. **Character type**: No single-character type (use strings)
6. **Module system**: Joy's `DEFINE`, `HIDE`, `IN`, etc.

## Additional pyjoy2 Features

### Python Integration

```
# Direct Python expressions
`[x**2 for x in range(10)]`

# Import and use any library
!import numpy as np
`np.array([1,2,3])`

# Define words in Python
@word
def quadratic(a, b, c, x):
    return a*x**2 + b*x + c
```

### REPL Commands

```
.s          # Show stack with types
.c          # Clear stack
.w          # List all words
.def name [body]  # Define word
.import mod       # Import module
.load file        # Load Joy file
.help             # Show help
```

### Extended Literals

```
1e-5        # Scientific notation
nil         # Null value
{1 2 3}     # Set literal
```

### String Escape Sequences

```
"line1\nline2"    # Newline
"tab\there"       # Tab
"quote\"here"     # Escaped quote
```

## Example: Factorial Comparison

### Joy

```
DEFINE factorial == [0 =] [pop 1] [dup 1 - factorial *] ifte.
5 factorial.
```

### pyjoy2

```
# Using linrec
5 [0 =] [pop 1] [dup 1 -] [*] linrec

# Or define a word
.def factorial [[0 =] [pop 1] [dup 1 -] [*] linrec]
5 factorial

# Or use Python
`math.factorial(5)`
```

## Example: Quicksort Comparison

### Joy

```
DEFINE qsort ==
  [small] []
  [uncons [>] split]
  [enconcat] binrec.
```

### pyjoy2

```
# Method 1: Using built-in sort (calls Python's sorted)
[3 1 4 1 5] sort
# => [1 1 3 4 5]

# Method 2: Using Python's sorted directly
[3 1 4 1 5] `sorted(S.pop())`
# => [1 1 3 4 5]

# Method 3: Pure Joy-style quicksort using ifte
.def qsort [
  [small] []
  [uncons [over <] partition qsort swap qsort swap enconcat]
  ifte
]

[3 1 4 1 5] qsort
# => [1 1 3 4 5]

# Method 4: With custom comparator via Python
[3 1 4 1 5] `sorted(S.pop(), reverse=True)`
# => [5 4 3 1 1]
```

**Notes on the pyjoy2 qsort:**
- `small` tests if a list has 0 or 1 elements (base case)
- `partition` splits a list by predicate: `[...] [P] -> [yes] [no]`
- `enconcat` combines: `X [A] [B] -> [A X B...]`
- The Joy version uses `binrec` with `split`; pyjoy2 uses `ifte` with `partition` for clarity

## Migration Tips

1. **Replace `div` with `//`** for integer division
2. **Use Python for file I/O** via backticks or bang syntax
3. **Any Joy list is a Python list** - you can use Python methods
4. **Define complex words in Python** using `@word` decorator
5. **Use `.s` liberally** to inspect stack during development

## Conclusion

pyjoy2 maintains Joy's concatenative, stack-based semantics while adding Python's rich ecosystem. The core Joy words are preserved, making it easy to port Joy programs. The Python integration provides an escape hatch for operations that would be complex in pure Joy.

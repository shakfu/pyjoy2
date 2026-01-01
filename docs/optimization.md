# PyJoy2 Optimization Analysis

This document captures profiling results and potential optimization opportunities for PyJoy2.

## Profiling Setup

Run profiling with:
```bash
make profile FILE=examples/benchmark.joy
```

The benchmark exercises: fibonacci (binrec), quicksort (binrec), map/filter/fold, factorial (linrec), primrec, while loops, and list operations.

## Profile Results

From a typical benchmark run (~224k function calls, 52ms total):

| Function | Calls | Cumulative Time | Description |
|----------|-------|-----------------|-------------|
| `core.py:wrapper` (@define) | 3,426 | 42ms | Decorator wrapper overhead |
| `core.py:execute` | 7,894 | 35ms | Main execution loop |
| `builtins.py:recurse` (binrec) | 2,022 | 29ms | Binary recursion |
| `core.py:wrapper` (@word) | 7,970 | 21ms | Auto-pop decorator wrapper |
| `core.py:pop` | 13,858 | 12ms | Stack pop operations |
| `builtins.py:_while` | 1 | 10ms | While loop combinator |

## Identified Hotspots

### 1. `@define` Wrapper Overhead

The `@define` decorator creates a wrapper that just forwards to the original function:

```python
# core.py:172-174
def wrapper(stack: Stack) -> None:
    return f(stack)  # Pure indirection, no added value at runtime
```

**Opportunity:** Register `f` directly without wrapping.

### 2. `@word` Wrapper Complexity

The `@word` decorator does significant work per call:
- Checks if `n_params > 0`
- Validates stack has enough elements
- Pops args via generator expression creating a tuple
- Reverses args for correct order
- Calls the underlying function
- Checks if result is not None
- Checks if result is tuple for multi-push

For simple builtins like `+`, `-`, `*`, `/`, this overhead is significant relative to the actual operation.

### 3. Stack Copying in Combinators

Both `while` and `binrec` copy the entire stack every iteration:

```python
# builtins.py - _while
saved = list(s)      # O(n) copy
execute(s, cond)
result = s.pop()
s.clear()
s.extend(saved)      # O(n) restore
```

For deep stacks or many iterations, this becomes expensive.

### 4. `Stack.pop(n)` Creates Intermediate Tuple

```python
# core.py:45
result = tuple(list.pop(self) for _ in range(n))
```

Creates a generator and tuple for every multi-pop.

### 5. `execute()` Type Checks

Every item in a quotation triggers:
```python
if callable(program):
    ...
elif isinstance(program, list):
    ...
```

## Optimization Opportunities

| Optimization | Effort | Impact | Risk |
|--------------|--------|--------|------|
| Remove `@define` wrapper indirection | Low | Medium | Low |
| Specialize `pop(2)`, `pop(3)` | Low | Medium | Low |
| Avoid stack copy in `while`/`binrec` | Medium | High for loops | Medium |
| Pre-resolve words in quotations | Medium | Medium | Low |
| Inline simple words (`dup`, `swap`) | High | Medium | High |
| Compile quotations to bytecode | High | High | High |

### Low-Hanging Fruit

1. **Direct registration for `@define`**: Store `f` in WORDS directly, set attributes on `f` itself
2. **Specialized pop**: Add `pop2()`, `pop3()` methods that avoid tuple creation
3. **Cache callable checks**: Mark items in quotations during parse

### Medium Effort

1. **Stack delta tracking**: Instead of copying entire stack, track only what the condition/body should see
2. **Quotation compilation**: Convert quotations to a more efficient representation at parse time

### High Effort (Diminishing Returns)

1. **JIT-style optimization**: Inline frequently-called word sequences
2. **Bytecode compilation**: Compile Joy to Python bytecode

## Design Trade-offs

The current implementation prioritizes:
- **Readability**: Code is clear and Pythonic
- **Debuggability**: Stack traces are meaningful
- **Extensibility**: Easy to add new words
- **Correctness**: Behavior matches Joy semantics

Over:
- **Raw performance**: Acceptable for a teaching/hobby language

For most use cases, the current performance is adequate. Optimization should only be pursued if profiling shows it's needed for real workloads.

## Reproducing This Analysis

```bash
# Run benchmark
make profile FILE=examples/benchmark.joy

# Save detailed stats for analysis
make profile-stats FILE=examples/benchmark.joy

# View with custom sorting
uv run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumtime').print_stats(30)
"
```

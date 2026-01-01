# PyJoy2 Optimization Analysis

This document captures profiling results and optimization work for PyJoy2.

## Profiling Setup

Run profiling with:
```bash
make profile FILE=examples/benchmark.joy
```

The benchmark exercises: fibonacci (binrec), quicksort (binrec), map/filter/fold, factorial (linrec), primrec, while loops, and list operations.

## Implemented Optimizations

### 1. `@define` Direct Registration

**Before:** The decorator created a wrapper that just forwarded calls:
```python
def wrapper(stack: Stack) -> None:
    return f(stack)  # Pure indirection
```

**After:** Register `f` directly, set attributes on `f` itself:
```python
f._is_word = True
f.joy_word = word_name
WORDS[word_name] = f
return f
```

**Impact:** Eliminated 3,426 wrapper calls per benchmark run.

### 2. `@word` Specialized Wrappers

**Before:** One generic wrapper handling all cases:
- Checked `n_params > 0`
- Used `stack.pop(n)` creating tuples
- Reversed args with `args[::-1]`
- Unpacked with `f(*args)`

**After:** Generate specialized wrappers for 0, 1, 2, 3 params:
```python
# Example for 2 params
def wrapper(stack: Stack) -> None:
    _check(stack)
    b = list.pop(stack)
    a = list.pop(stack)
    result = f(a, b)
    if result is not None:
        stack.push(result)
```

**Impact:**
- No tuple creation or reversal
- Direct `list.pop()` bypasses `Stack.pop()` method overhead
- 48% reduction in wrapper time

## Performance Results

### Before Optimization
```
~224k function calls, 52ms total

| Function                    | Calls  | Time |
|-----------------------------|--------|------|
| core.py:wrapper (@define)   | 3,426  | 42ms |
| core.py:execute             | 7,894  | 35ms |
| builtins.py:recurse         | 2,022  | 29ms |
| core.py:wrapper (@word)     | 7,970  | 21ms |
| core.py:pop                 | 13,858 | 12ms |
```

### After Optimization
```
~179k function calls, 40ms total

| Function                    | Calls  | Time |
|-----------------------------|--------|------|
| core.py:execute             | 7,894  | 24ms |
| builtins.py:recurse         | 2,022  | 21ms |
| core.py:wrapper (@word)     | 7,869  | 11ms |
| builtins.py:_while          | 1      | 8ms  |
| core.py:_check              | 7,946  | 2ms  |
| core.py:pop                 | 5,912  | 2ms  |
```

### Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Function calls | 223,801 | 178,894 | **-20%** |
| Total time | 52ms | 40ms | **-23%** |
| `@word` wrapper | 21ms | 11ms | **-48%** |
| `Stack.pop` calls | 13,858 | 5,912 | **-57%** |

## Remaining Optimization Opportunities

| Optimization | Effort | Impact | Risk |
|--------------|--------|--------|------|
| Avoid stack copy in `while`/`binrec` | Medium | High for loops | Medium |
| Pre-resolve words in quotations | Medium | Medium | Low |
| Inline simple words (`dup`, `swap`) | High | Medium | High |
| Compile quotations to bytecode | High | High | High |

### Stack Copying in Combinators

Both `while` and `binrec` copy the entire stack every iteration:

```python
# builtins.py - _while
saved = list(s)      # O(n) copy
execute(s, cond)
result = s.pop()
s.clear()
s.extend(saved)      # O(n) restore
```

For deep stacks or many iterations, this becomes expensive. Could track stack depth delta instead.

### `execute()` Type Checks

Every item in a quotation triggers:
```python
if callable(program):
    ...
elif isinstance(program, list):
    ...
```

Could pre-compile quotations to avoid runtime type checks.

## Design Trade-offs

The implementation prioritizes:
- **Readability**: Code is clear and Pythonic
- **Debuggability**: Stack traces are meaningful
- **Extensibility**: Easy to add new words
- **Correctness**: Behavior matches Joy semantics

Over:
- **Raw performance**: Acceptable for a teaching/hobby language

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

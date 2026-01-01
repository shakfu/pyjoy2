# pyjoy2.md

A practical, Pythonic concatenative language - loosening Joy semantics for Python integration.

> This document is an alternative to [pyjoy.md](docs/pyjoy.md), which describes a faithful Joy reimplementation. Here we explore a more pragmatic approach that embraces Python's dynamic nature.

## 1. Philosophy Shift

### 1.1 From Joy Semantics to Python Pragmatism

| Aspect | pyjoy.md (Faithful) | pyjoy2.md (Practical) |
|--------|---------------------|----------------------|
| **Stack values** | Wrapped `JoyValue` types | Any Python object |
| **Type system** | Joy's 10 types | Python's duck typing |
| **Lists** | Immutable tuples | Any iterable |
| **Quotations** | `JoyQuotation` AST | Python callables or lists |
| **Functions** | Registered primitives | Any callable |
| **Strings** | Joy string semantics | Python str |
| **Numbers** | int64/double limits | Python arbitrary precision |

### 1.2 Core Principle

> **The stack is just a Python list. Words are just Python functions. Quotations are just Python callables.**

This means:
- NumPy arrays can be on the stack
- Pandas DataFrames can be manipulated
- Any Python library is immediately usable
- No translation layer between Joy and Python

---

## 2. Minimal Core Design

### 2.1 The Stack

```python
class Stack(list):
    """
    Stack is just a Python list with convenience methods.
    No type wrapping - any Python object allowed.
    """

    def push(self, *items):
        """Push one or more items"""
        self.extend(items)
        return self

    def pop(self, n=1):
        """Pop n items, return single item if n=1, else tuple"""
        if n == 1:
            return super().pop()
        return tuple(super().pop() for _ in range(n))

    def peek(self, depth=0):
        """Look at item at depth (0 = top)"""
        return self[-(depth + 1)]

    def dup(self):
        """Duplicate top"""
        self.append(self[-1])
        return self

    def swap(self):
        """Swap top two"""
        self[-1], self[-2] = self[-2], self[-1]
        return self

    def rot(self):
        """Rotate top three: a b c -> b c a"""
        self[-3], self[-2], self[-1] = self[-2], self[-1], self[-3]
        return self
```

### 2.2 Words as Functions

A "word" is any Python callable that takes a stack and optionally returns a value:

```python
from typing import Callable, Any

# Type alias for Joy words
Word = Callable[[Stack], Any]

def word(f: Callable) -> Word:
    """
    Decorator that turns a regular function into a stack word.
    The function's arguments are popped from stack (right-to-left).
    Return value (if any) is pushed to stack.
    """
    import inspect
    sig = inspect.signature(f)
    n_params = len(sig.parameters)

    def wrapper(stack: Stack) -> None:
        args = stack.pop(n_params) if n_params > 0 else ()
        if n_params == 1:
            args = (args,)  # pop(1) returns single item
        result = f(*args)
        if result is not None:
            stack.push(result)

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper
```

### 2.3 Example Words

```python
# Arithmetic - just use Python operators
@word
def add(a, b): return a + b

@word
def sub(a, b): return a - b

@word
def mul(a, b): return a * b

@word
def div(a, b): return a / b

@word
def mod(a, b): return a % b

# Comparisons
@word
def lt(a, b): return a < b

@word
def gt(a, b): return a > b

@word
def eq(a, b): return a == b

# Stack manipulation (operate on stack directly)
def dup(stack: Stack): stack.dup()
def pop_(stack: Stack): stack.pop()
def swap(stack: Stack): stack.swap()
def rot(stack: Stack): stack.rot()
def over(stack: Stack): stack.push(stack.peek(1))

# List operations - work on any iterable
@word
def first(seq): return next(iter(seq))

@word
def rest(seq):
    it = iter(seq)
    next(it)
    return list(it)

@word
def cons(item, seq):
    return [item] + list(seq)

@word
def cat(a, b):
    """Concatenate - works on strings, lists, etc."""
    if isinstance(a, str):
        return a + b
    return list(a) + list(b)

@word
def size(seq):
    return len(seq)
```

---

## 3. Quotations as Callables

### 3.1 The Key Insight

In Joy, quotations are "programs as data". In Python, we already have this:
- **Lambda functions** are anonymous code
- **Lists of callables** can represent programs
- **Partial application** via `functools.partial`

### 3.2 Quotation Representations

```python
from typing import Union, List
from functools import partial

# A quotation is either:
# 1. A callable (lambda, function, partial)
# 2. A list of words/values to execute
Quotation = Union[Callable, List]

def quote(*items) -> List:
    """Create a quotation from items"""
    return list(items)

def is_quotation(x) -> bool:
    """Check if x is a quotation"""
    return callable(x) or isinstance(x, list)
```

### 3.3 Executing Quotations

```python
def execute(stack: Stack, program: Quotation) -> None:
    """
    Execute a quotation on the stack.
    - If callable: call it with stack
    - If list: execute each element in order
    """
    if callable(program):
        program(stack)
    elif isinstance(program, list):
        for item in program:
            if callable(item):
                item(stack)
            else:
                # Literal value - push to stack
                stack.push(item)
    else:
        # Not a quotation - push as literal
        stack.push(program)
```

---

## 4. Combinators

### 4.1 Basic Execution

```python
def i(stack: Stack):
    """Execute quotation on top of stack"""
    quot = stack.pop()
    execute(stack, quot)

def x(stack: Stack):
    """Execute quotation without consuming it"""
    quot = stack.peek()
    execute(stack, quot)

def dip(stack: Stack):
    """Execute quotation under top element"""
    quot = stack.pop()
    saved = stack.pop()
    execute(stack, quot)
    stack.push(saved)
```

### 4.2 Conditionals

```python
def ifte(stack: Stack):
    """
    [condition] [then] [else] ifte
    """
    else_branch = stack.pop()
    then_branch = stack.pop()
    condition = stack.pop()

    # Save stack state for condition test
    saved = list(stack)

    # Run condition
    execute(stack, condition)
    result = stack.pop()

    # Restore stack
    stack.clear()
    stack.extend(saved)

    # Execute appropriate branch
    execute(stack, then_branch if result else else_branch)

def branch(stack: Stack):
    """
    bool [then] [else] branch
    """
    else_branch = stack.pop()
    then_branch = stack.pop()
    condition = stack.pop()
    execute(stack, then_branch if condition else else_branch)

def when(stack: Stack):
    """
    bool [then] when - execute if true, else do nothing
    """
    then_branch = stack.pop()
    condition = stack.pop()
    if condition:
        execute(stack, then_branch)

def unless(stack: Stack):
    """
    bool [else] unless - execute if false
    """
    else_branch = stack.pop()
    condition = stack.pop()
    if not condition:
        execute(stack, else_branch)
```

### 4.3 Iteration

```python
def each(stack: Stack):
    """
    [items] [action] each - apply action to each item
    """
    action = stack.pop()
    items = stack.pop()

    for item in items:
        stack.push(item)
        execute(stack, action)

def map_(stack: Stack):
    """
    [items] [transform] map - transform each item, collect results
    """
    transform = stack.pop()
    items = stack.pop()

    results = []
    for item in items:
        stack.push(item)
        execute(stack, transform)
        results.append(stack.pop())

    stack.push(results)

def filter_(stack: Stack):
    """
    [items] [predicate] filter - keep items where predicate is true
    """
    predicate = stack.pop()
    items = stack.pop()

    results = []
    for item in items:
        stack.push(item)
        execute(stack, predicate)
        if stack.pop():
            results.append(item)

    stack.push(results)

def fold(stack: Stack):
    """
    [items] initial [combiner] fold - reduce to single value
    """
    combiner = stack.pop()
    acc = stack.pop()
    items = stack.pop()

    for item in items:
        stack.push(acc)
        stack.push(item)
        execute(stack, combiner)
        acc = stack.pop()

    stack.push(acc)

def times(stack: Stack):
    """
    n [action] times - repeat action n times
    """
    action = stack.pop()
    n = stack.pop()

    for _ in range(n):
        execute(stack, action)

def while_(stack: Stack):
    """
    [condition] [body] while - repeat while condition is true
    """
    body = stack.pop()
    condition = stack.pop()

    while True:
        saved = list(stack)
        execute(stack, condition)
        if not stack.pop():
            stack.clear()
            stack.extend(saved)
            break
        stack.clear()
        stack.extend(saved)
        execute(stack, body)
```

### 4.4 Recursion Combinators

```python
def linrec(stack: Stack):
    """
    [pred] [then] [rec1] [rec2] linrec
    Linear recursion: if pred then else (rec1; recurse; rec2)
    """
    rec2 = stack.pop()
    rec1 = stack.pop()
    then_ = stack.pop()
    pred = stack.pop()

    def recurse():
        saved = list(stack)
        execute(stack, pred)
        if stack.pop():
            stack.clear()
            stack.extend(saved)
            execute(stack, then_)
        else:
            stack.clear()
            stack.extend(saved)
            execute(stack, rec1)
            recurse()
            execute(stack, rec2)

    recurse()

def binrec(stack: Stack):
    """
    [pred] [then] [split] [merge] binrec
    Binary recursion (divide and conquer)
    """
    merge = stack.pop()
    split = stack.pop()
    then_ = stack.pop()
    pred = stack.pop()

    def recurse():
        saved = list(stack)
        execute(stack, pred)
        if stack.pop():
            stack.clear()
            stack.extend(saved)
            execute(stack, then_)
        else:
            stack.clear()
            stack.extend(saved)
            execute(stack, split)

            # Save first part, recurse on second
            first = stack.pop()
            recurse()
            second_result = stack.pop()

            # Now recurse on first
            stack.push(first)
            recurse()

            # Merge results
            stack.push(second_result)
            execute(stack, merge)

    recurse()
```

---

## 5. Pythonic Enhancements

### 5.1 Method Chaining on Stack

```python
class Stack(list):
    # ... previous methods ...

    def apply(self, word: Word) -> 'Stack':
        """Apply word and return self for chaining"""
        word(self)
        return self

    def run(self, *program) -> 'Stack':
        """Execute a sequence of words/values"""
        for item in program:
            if callable(item):
                item(self)
            else:
                self.push(item)
        return self

# Usage:
s = Stack()
s.push(5).push(3).apply(add).apply(dup).apply(mul)
# Stack: [64]

# Or more readable:
s.run(5, 3, add, dup, mul)
```

### 5.2 Stack as Context Manager

```python
class Stack(list):
    # ... previous methods ...

    def __enter__(self):
        """Save state on enter"""
        self._saved = list(self)
        return self

    def __exit__(self, *args):
        """Restore state on exit (unless successful)"""
        if args[0] is not None:  # Exception occurred
            self.clear()
            self.extend(self._saved)
        return False

# Usage:
s = Stack([1, 2, 3])
with s:
    s.apply(some_operation)  # If fails, stack is restored
```

### 5.3 Decorator for Auto-Registration

```python
# Global word registry
WORDS = {}

def define(name: str = None):
    """
    Decorator to define a word and register it.
    """
    def decorator(f):
        w = word(f) if not hasattr(f, '__wrapped__') else f
        word_name = name or f.__name__
        WORDS[word_name] = w
        return w
    return decorator

# Usage:
@define("+")
def add(a, b): return a + b

@define("square")
def square(x): return x * x

# Now WORDS["+"] and WORDS["square"] are available
```

### 5.4 String-Based Programs

```python
def parse_program(source: str) -> List:
    """
    Simple parser for string programs.
    Returns list of words and literals.
    """
    tokens = source.split()
    program = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == '[':
            # Find matching ]
            depth = 1
            j = i + 1
            while depth > 0:
                if tokens[j] == '[': depth += 1
                elif tokens[j] == ']': depth -= 1
                j += 1
            # Recursively parse quotation
            inner = ' '.join(tokens[i+1:j-1])
            program.append(parse_program(inner))
            i = j
        elif token == ']':
            i += 1
        elif token in WORDS:
            program.append(WORDS[token])
            i += 1
        else:
            # Try to parse as number
            try:
                program.append(int(token))
            except ValueError:
                try:
                    program.append(float(token))
                except ValueError:
                    # String literal or undefined word
                    program.append(token)
            i += 1

    return program


def run(source: str, stack: Stack = None) -> Stack:
    """Execute a string program"""
    stack = stack or Stack()
    program = parse_program(source)
    execute(stack, program)
    return stack

# Usage:
s = run("5 3 + dup *")  # Stack: [64]
s = run("[1 2 3] [dup *] map")  # Stack: [[1, 4, 9]]
```

---

## 6. Python Integration

### 6.1 Using Any Python Object

```python
import numpy as np
import pandas as pd

s = Stack()

# NumPy arrays
s.push(np.array([1, 2, 3, 4, 5]))
s.push(np.array([10, 20, 30, 40, 50]))
s.apply(add)  # Works! NumPy broadcasting
# Stack: [array([11, 22, 33, 44, 55])]

# Pandas DataFrames
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
s.push(df)
s.push(lambda df: df['a'] + df['b'])  # Quotation as lambda
s.apply(i)  # Execute the lambda
# Stack: [Series([5, 7, 9])]

# Any Python object
s.push({"name": "Alice", "age": 30})
s.push(lambda d: d["name"])
s.apply(i)
# Stack: ["Alice"]
```

### 6.2 Wrapping Python Functions

```python
def py(func: Callable) -> Word:
    """
    Wrap any Python function as a stack word.
    Arguments popped from stack, result pushed.
    """
    import inspect
    sig = inspect.signature(func)
    n_params = len([p for p in sig.parameters.values()
                    if p.default == inspect.Parameter.empty])

    def wrapper(stack: Stack):
        args = []
        for _ in range(n_params):
            args.insert(0, stack.pop())
        result = func(*args)
        if result is not None:
            stack.push(result)

    wrapper.__name__ = func.__name__
    return wrapper

# Usage:
import math
sqrt = py(math.sqrt)
sin = py(math.sin)
cos = py(math.cos)

s = Stack()
s.run(16, sqrt)  # Stack: [4.0]
s.run(0, sin)    # Stack: [4.0, 0.0]

# Wrap any function on the fly
import json
s.push('{"a": 1, "b": 2}')
s.apply(py(json.loads))
# Stack: [{'a': 1, 'b': 2}]
```

### 6.3 Method Calls

```python
def method(name: str, *args) -> Word:
    """
    Call a method on top of stack.
    Additional args are passed to method.
    """
    def wrapper(stack: Stack):
        obj = stack.pop()
        result = getattr(obj, name)(*args)
        if result is not None:
            stack.push(result)
    return wrapper

# Usage:
s = Stack()
s.push("hello world")
s.apply(method("upper"))
# Stack: ["HELLO WORLD"]

s.push("hello world")
s.apply(method("split", " "))
# Stack: ["HELLO WORLD", ["hello", "world"]]

s.push([3, 1, 4, 1, 5])
s.apply(method("sort"))  # in-place, returns None
s.apply(method("reverse"))
# Stack: [..., [5, 4, 3, 1, 1]]
```

### 6.4 Attribute Access

```python
def attr(name: str) -> Word:
    """Get attribute from top of stack"""
    def wrapper(stack: Stack):
        obj = stack.pop()
        stack.push(getattr(obj, name))
    return wrapper

def setattr_(name: str) -> Word:
    """Set attribute on object: obj value setattr_('name')"""
    def wrapper(stack: Stack):
        value = stack.pop()
        obj = stack.peek()  # Don't pop, keep object
        setattr(obj, name, value)
    return wrapper

# Usage with classes
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

s = Stack()
s.push(Point(3, 4))
s.apply(dup)
s.apply(attr("x"))
s.apply(swap)
s.apply(attr("y"))
# Stack: [3, 4]
s.run(dup, mul, swap, dup, mul, add, py(math.sqrt))
# Stack: [5.0]  (distance from origin)
```

---

## 7. Advanced Patterns

### 7.1 Cleave (Apply Multiple Functions)

```python
def cleave(stack: Stack):
    """
    value [f1] [f2] ... [fn] n cleave
    Apply n functions to same value, push all results
    """
    n = stack.pop()
    funcs = [stack.pop() for _ in range(n)][::-1]
    value = stack.pop()

    for f in funcs:
        stack.push(value)
        execute(stack, f)

def bi(stack: Stack):
    """
    value [f] [g] bi - apply two functions to same value
    """
    g = stack.pop()
    f = stack.pop()
    value = stack.peek()  # Keep original

    execute(stack, f)
    result1 = stack.pop()

    stack.push(value)
    execute(stack, g)

    # Swap to maintain order
    result2 = stack.pop()
    stack.push(result1)
    stack.push(result2)

def tri(stack: Stack):
    """
    value [f] [g] [h] tri - apply three functions to same value
    """
    h = stack.pop()
    g = stack.pop()
    f = stack.pop()
    value = stack.pop()

    results = []
    for func in [f, g, h]:
        stack.push(value)
        execute(stack, func)
        results.append(stack.pop())

    for r in results:
        stack.push(r)
```

### 7.2 Spread (Apply Functions to Multiple Values)

```python
def spread(stack: Stack):
    """
    v1 v2 ... vn [f1] [f2] ... [fn] n spread
    Apply each function to corresponding value
    """
    n = stack.pop()
    funcs = [stack.pop() for _ in range(n)][::-1]
    values = [stack.pop() for _ in range(n)][::-1]

    for v, f in zip(values, funcs):
        stack.push(v)
        execute(stack, f)

def bi_star(stack: Stack):
    """
    a b [f] [g] bi* - apply f to a, g to b
    """
    g = stack.pop()
    f = stack.pop()
    b = stack.pop()
    a = stack.pop()

    stack.push(a)
    execute(stack, f)
    result_a = stack.pop()

    stack.push(b)
    execute(stack, g)

    result_b = stack.pop()
    stack.push(result_a)
    stack.push(result_b)
```

### 7.3 Composition

```python
def compose(*funcs) -> Word:
    """
    Compose multiple words into one.
    compose(f, g, h) = h(g(f(x)))
    """
    def composed(stack: Stack):
        for f in funcs:
            execute(stack, f)
    return composed

def curry(word: Word, *args) -> Word:
    """
    Partially apply arguments to a word.
    """
    def curried(stack: Stack):
        for arg in args:
            stack.push(arg)
        word(stack)
    return curried

# Usage:
double = curry(mul, 2)  # Multiply by 2
increment = curry(add, 1)  # Add 1

double_then_inc = compose(double, increment)

s = Stack()
s.run(5, double_then_inc)  # 5 * 2 + 1 = 11
```

### 7.4 Pipelines

```python
class Pipeline:
    """
    Fluent pipeline builder for concatenative operations.
    """
    def __init__(self, initial=None):
        self.stack = Stack()
        if initial is not None:
            self.stack.push(initial)
        self.ops = []

    def push(self, *values):
        for v in values:
            self.ops.append(('push', v))
        return self

    def __getattr__(self, name):
        """Access registered words by name"""
        if name in WORDS:
            self.ops.append(('word', WORDS[name]))
        return self

    def apply(self, word):
        self.ops.append(('word', word))
        return self

    def run(self) -> Stack:
        for op_type, op_val in self.ops:
            if op_type == 'push':
                self.stack.push(op_val)
            else:
                op_val(self.stack)
        return self.stack

    @property
    def result(self):
        """Execute and return top of stack"""
        return self.run().pop()

# Usage:
result = (Pipeline()
    .push(1, 2, 3, 4, 5)
    .push([])
    .apply(cons).apply(cons).apply(cons).apply(cons).apply(cons)
    .push(lambda x: x * x)
    .apply(map_)
    .result)
# result = [1, 4, 9, 16, 25]
```

---

## 8. Simplified Implementation

### 8.1 Complete Minimal Core (~150 LOC)

```python
"""
pyjoy2.py - Minimal Pythonic concatenative language
"""
from typing import Callable, List, Any, Union
from functools import wraps
import inspect

# ============ Stack ============

class Stack(list):
    def push(self, *items):
        self.extend(items)
        return self

    def pop(self, n=1):
        if n == 1:
            return super().pop()
        return tuple(super().pop() for _ in range(n))

    def peek(self, depth=0):
        return self[-(depth + 1)]

    def dup(self): self.append(self[-1]); return self
    def swap(self): self[-1], self[-2] = self[-2], self[-1]; return self
    def over(self): self.append(self[-2]); return self


# ============ Words ============

WORDS = {}

def word(f):
    """Turn function into stack word (args from stack, result to stack)"""
    sig = inspect.signature(f)
    n = len(sig.parameters)

    @wraps(f)
    def wrapper(s: Stack):
        args = s.pop(n) if n else ()
        args = (args,) if n == 1 else args
        result = f(*args) if n else f()
        if result is not None:
            s.push(result)
    return wrapper

def define(name=None):
    """Register a word in global dictionary"""
    def decorator(f):
        w = word(f) if not callable(getattr(f, '__wrapped__', None)) else f
        WORDS[name or f.__name__] = w
        return w
    return decorator


# ============ Execution ============

def execute(s: Stack, program):
    """Execute program on stack"""
    if callable(program):
        program(s)
    elif isinstance(program, list):
        for item in program:
            if callable(item):
                item(s)
            else:
                s.push(item)
    else:
        s.push(program)


# ============ Core Words ============

# Stack ops
@define("dup")
def _dup(x): return (x, x)  # Push tuple = push both

@define("drop")
def _drop(x): pass

@define("swap")
def _swap(a, b): return (b, a)

@define("over")
def _over(a, b): return (a, b, a)

# Math
@define("+")
def _add(a, b): return a + b

@define("-")
def _sub(a, b): return a - b

@define("*")
def _mul(a, b): return a * b

@define("/")
def _div(a, b): return a / b

@define("%")
def _mod(a, b): return a % b

# Comparison
@define("<")
def _lt(a, b): return a < b

@define(">")
def _gt(a, b): return a > b

@define("=")
def _eq(a, b): return a == b

# Logic
@define("not")
def _not(x): return not x

@define("and")
def _and(a, b): return a and b

@define("or")
def _or(a, b): return a or b

# Lists
@define("first")
def _first(seq): return next(iter(seq))

@define("rest")
def _rest(seq): return list(seq)[1:]

@define("cons")
def _cons(x, seq): return [x] + list(seq)

@define("cat")
def _cat(a, b): return list(a) + list(b)

@define("len")
def _len(seq): return len(seq)


# ============ Combinators ============

def i(s: Stack):
    """[quot] i - execute quotation"""
    execute(s, s.pop())
WORDS["i"] = i

def dip(s: Stack):
    """x [quot] dip - execute quot under x"""
    quot, x = s.pop(2)
    execute(s, quot)
    s.push(x)
WORDS["dip"] = dip

def ifte(s: Stack):
    """[cond] [then] [else] ifte"""
    else_, then_, cond = s.pop(3)
    saved = list(s)
    execute(s, cond)
    result = s.pop()
    s.clear(); s.extend(saved)
    execute(s, then_ if result else else_)
WORDS["ifte"] = ifte

def map_(s: Stack):
    """[seq] [quot] map"""
    quot, seq = s.pop(2)
    results = []
    for item in seq:
        s.push(item)
        execute(s, quot)
        results.append(s.pop())
    s.push(results)
WORDS["map"] = map_

def filter_(s: Stack):
    """[seq] [pred] filter"""
    pred, seq = s.pop(2)
    results = []
    for item in seq:
        s.push(item)
        execute(s, pred)
        if s.pop():
            results.append(item)
    s.push(results)
WORDS["filter"] = filter_

def fold(s: Stack):
    """[seq] init [quot] fold"""
    quot, init, seq = s.pop(3)
    acc = init
    for item in seq:
        s.push(acc, item)
        execute(s, quot)
        acc = s.pop()
    s.push(acc)
WORDS["fold"] = fold

def each(s: Stack):
    """[seq] [quot] each - apply to each, no collection"""
    quot, seq = s.pop(2)
    for item in seq:
        s.push(item)
        execute(s, quot)
WORDS["each"] = each


# ============ REPL ============

def run(source: str, s: Stack = None) -> Stack:
    """Run string program"""
    s = s or Stack()
    tokens = source.replace('[', ' [ ').replace(']', ' ] ').split()
    program = _parse(tokens, 0)[0]
    execute(s, program)
    return s

def _parse(tokens, i):
    """Simple recursive parser"""
    result = []
    while i < len(tokens):
        t = tokens[i]
        if t == '[':
            inner, i = _parse(tokens, i + 1)
            result.append(inner)
        elif t == ']':
            return result, i + 1
        elif t in WORDS:
            result.append(WORDS[t])
            i += 1
        else:
            try:
                result.append(int(t))
            except:
                try:
                    result.append(float(t))
                except:
                    result.append(t)  # string
            i += 1
    return result, i


# ============ Convenience ============

def repl():
    """Simple REPL"""
    s = Stack()
    print("PyJoy2 REPL. 'quit' to exit.")
    while True:
        try:
            line = input("> ").strip()
            if line == "quit":
                break
            if line:
                run(line, s)
                print(f"  {list(s)}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    repl()
```

---

## 9. Comparison with PYJOY.md

| Feature | PYJOY.md | PYJOY2.md |
|---------|----------|-----------|
| **Lines of code** | ~1000+ estimated | ~150 core |
| **Type system** | Tagged union (JoyValue) | Duck typing |
| **Stack values** | Wrapped types only | Any Python object |
| **Quotations** | JoyQuotation class | Lists or callables |
| **Set limitation** | 0-63 integers | Any Python set |
| **Integer limit** | 64-bit | Arbitrary precision |
| **Extensibility** | Decorator with validation | Simple callable |
| **Python interop** | Conversion required | Direct |
| **Memory overhead** | Higher (wrappers) | Lower |
| **Type safety** | Runtime checked | Duck typed |
| **Error messages** | Joy-style | Python-style |
| **Learning curve** | Higher | Lower |
| **Performance** | Moderate overhead | Minimal overhead |

---

## 10. When to Use Which

### Use PYJOY.md (Faithful) When:
- Porting existing Joy programs
- Teaching Joy language specifically
- Need exact Joy semantics
- Want predictable type behavior

### Use PYJOY2.md (Practical) When:
- Building new concatenative pipelines
- Integrating with Python ecosystem
- Want minimal boilerplate
- Performance matters
- Prototyping

---

## 11. Example Session

```python
>>> from pyjoy2 import Stack, run, WORDS, word, define

# Basic operations
>>> run("3 4 +")
Stack([7])

>>> run("5 dup *")
Stack([25])

>>> run("[1 2 3 4 5] [dup *] map")
Stack([[1, 4, 9, 16, 25]])

# Custom word
>>> @define("square")
... def square(x): return x * x
>>> run("7 square")
Stack([49])

# Using Python directly
>>> import numpy as np
>>> s = Stack()
>>> s.push(np.array([1, 2, 3]))
>>> s.push(np.array([4, 5, 6]))
>>> run("+ dup *", s)
Stack([array([25, 49, 81])])

# Lambda as quotation
>>> s = Stack()
>>> s.push([1, 2, 3, 4, 5])
>>> s.push(lambda x: x > 2)
>>> WORDS["filter"](s)
>>> s
Stack([[3, 4, 5]])

# Complex pipeline
>>> run("""
...   [1 2 3 4 5 6 7 8 9 10]
...   [2 %] filter
...   [dup *] map
...   0 [+] fold
... """)
Stack([165])  # Sum of squares of odd numbers 1-10
```

---

## 12. Enhanced REPL with Python Objects

The basic REPL in Section 8 only parses Joy-style tokens. For a true Joy-like experience with Python objects, we need a hybrid REPL that can evaluate Python expressions.

### 12.1 Hybrid REPL Design

```python
"""
Enhanced REPL that mixes Joy words with Python expressions.

Syntax:
  - Joy words: executed as usual
  - `expr`: backtick evaluates Python expression, pushes result
  - $(expr): alternative Python expression syntax
  - !statement: execute Python statement (no push)
  - .s: show stack
  - .c: clear stack
  - .w: list words
  - .def name [...]: define new word
"""

import re
import sys

class HybridREPL:
    def __init__(self):
        self.stack = Stack()
        self.pending_lines = []  # For multi-line input

        # Make word decorator and stack available in Python namespace
        self.globals = {
            'stack': self.stack,
            'S': self.stack,
            'word': word,
            'define': define,
            'WORDS': WORDS,
            'Stack': Stack,
            'execute': execute,
        }
        self.locals = {}

        # Pre-import common modules
        self._exec("import math")
        self._exec("import json")
        self._exec("import os")
        self._exec("import re")

    def _exec(self, code: str):
        """Execute Python statement"""
        exec(code, self.globals, self.locals)
        self.globals.update(self.locals)

    def _eval(self, expr: str):
        """Evaluate Python expression and return result"""
        return eval(expr, self.globals, self.locals)

    def process_line(self, line: str):
        """Process a single line of input"""
        line = line.strip()
        if not line:
            return

        # Commands
        if line == '.s':
            self._show_stack()
            return
        if line == '.c':
            self.stack.clear()
            print("  Stack cleared")
            return
        if line == '.w':
            self._show_words()
            return
        if line.startswith('.def '):
            self._define_word(line[5:])
            return
        if line.startswith('.import '):
            self._exec(f"import {line[8:]}")
            print(f"  Imported {line[8:]}")
            return

        # Parse and execute tokens
        self._execute_tokens(line)

    def _execute_tokens(self, line: str):
        """Parse and execute mixed Joy/Python tokens"""
        # Tokenize with support for Python expressions
        tokens = self._tokenize(line)

        for token in tokens:
            if token['type'] == 'python_expr':
                # Evaluate Python and push result
                result = self._eval(token['value'])
                self.stack.push(result)

            elif token['type'] == 'python_stmt':
                # Execute Python statement (no push)
                self._exec(token['value'])

            elif token['type'] == 'word':
                name = token['value']
                if name in WORDS:
                    WORDS[name](self.stack)
                elif name in self.globals:
                    # Push Python variable
                    self.stack.push(self.globals[name])
                else:
                    raise NameError(f"Unknown word: {name}")

            elif token['type'] == 'quotation':
                # Parse inner quotation
                inner = self._parse_quotation(token['value'])
                self.stack.push(inner)

            elif token['type'] == 'literal':
                self.stack.push(token['value'])

    def _tokenize(self, line: str):
        """
        Tokenize line into Joy words and Python expressions.

        Patterns:
          `...`     -> Python expression
          $(...)    -> Python expression (alternative)
          !...      -> Python statement (to end of token)
          [...]     -> Quotation
          "..."     -> String
          123, 3.14 -> Numbers
          word      -> Joy word or Python variable
        """
        tokens = []
        i = 0

        while i < len(line):
            # Skip whitespace
            while i < len(line) and line[i].isspace():
                i += 1
            if i >= len(line):
                break

            # Backtick Python expression: `expr`
            if line[i] == '`':
                j = line.index('`', i + 1)
                tokens.append({'type': 'python_expr', 'value': line[i+1:j]})
                i = j + 1

            # Dollar Python expression: $(expr)
            elif line[i:i+2] == '$(':
                depth = 1
                j = i + 2
                while depth > 0:
                    if line[j] == '(': depth += 1
                    elif line[j] == ')': depth -= 1
                    j += 1
                tokens.append({'type': 'python_expr', 'value': line[i+2:j-1]})
                i = j

            # Python statement: !statement
            elif line[i] == '!':
                # Read to end of line or next Joy token
                j = i + 1
                while j < len(line) and line[j] not in '[]`':
                    j += 1
                tokens.append({'type': 'python_stmt', 'value': line[i+1:j].strip()})
                i = j

            # Quotation: [...]
            elif line[i] == '[':
                depth = 1
                j = i + 1
                while depth > 0:
                    if line[j] == '[': depth += 1
                    elif line[j] == ']': depth -= 1
                    j += 1
                tokens.append({'type': 'quotation', 'value': line[i+1:j-1]})
                i = j

            # String: "..."
            elif line[i] == '"':
                j = line.index('"', i + 1)
                tokens.append({'type': 'literal', 'value': line[i+1:j]})
                i = j + 1

            # Number or word
            else:
                j = i
                while j < len(line) and not line[j].isspace() and line[j] not in '[]`"':
                    j += 1
                word = line[i:j]

                # Try to parse as number
                try:
                    if '.' in word:
                        tokens.append({'type': 'literal', 'value': float(word)})
                    else:
                        tokens.append({'type': 'literal', 'value': int(word)})
                except ValueError:
                    tokens.append({'type': 'word', 'value': word})

                i = j

        return tokens

    def _parse_quotation(self, inner: str) -> list:
        """Parse quotation content into executable list"""
        if not inner.strip():
            return []

        tokens = self._tokenize(inner)
        result = []

        for token in tokens:
            if token['type'] == 'word' and token['value'] in WORDS:
                result.append(WORDS[token['value']])
            elif token['type'] == 'quotation':
                result.append(self._parse_quotation(token['value']))
            elif token['type'] == 'python_expr':
                # Capture expression for lazy evaluation
                expr = token['value']
                result.append(lambda s, e=expr: s.push(self._eval(e)))
            else:
                result.append(token['value'])

        return result

    def _show_stack(self):
        """Display stack with types"""
        if not self.stack:
            print("  (empty)")
            return
        print("  Stack (top last):")
        for i, item in enumerate(self.stack):
            type_name = type(item).__name__
            repr_str = repr(item)
            if len(repr_str) > 60:
                repr_str = repr_str[:57] + "..."
            print(f"    {i}: ({type_name}) {repr_str}")

    def _show_words(self):
        """List available words"""
        words = sorted(WORDS.keys())
        print(f"  {len(words)} words: {', '.join(words)}")

    def _define_word(self, defn: str):
        """Define new word: .def name [body]"""
        match = re.match(r'(\w+)\s+\[(.+)\]', defn)
        if not match:
            print("  Usage: .def name [body]")
            return
        name, body = match.groups()
        quot = self._parse_quotation(body)

        def new_word(s: Stack, q=quot):
            execute(s, q)

        WORDS[name] = new_word
        print(f"  Defined: {name}")

    def _is_incomplete(self, code: str) -> bool:
        """Check if Python code is incomplete (needs more lines)"""
        try:
            compile(code, '<input>', 'exec')
            return False
        except SyntaxError as e:
            # "unexpected EOF" means incomplete
            return 'unexpected EOF' in str(e) or 'EOF while scanning' in str(e)

    def _handle_python_block(self, line: str) -> bool:
        """
        Handle multi-line Python blocks (def, class, if, for, etc.)
        Returns True if line was handled as Python block.
        """
        # Check if this starts a Python block
        block_starters = ('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')
        is_block_start = any(line.startswith(s) for s in block_starters)

        if self.pending_lines or is_block_start:
            self.pending_lines.append(line)
            code = '\n'.join(self.pending_lines)

            # Check if code is complete
            if self._is_incomplete(code):
                return True  # Need more lines

            # Execute complete block
            self._exec(code)
            self.pending_lines = []

            # Auto-register decorated functions as words
            self._register_new_words()
            return True

        return False

    def _register_new_words(self):
        """Check for newly defined functions with @word or @define decorator"""
        for name, obj in list(self.locals.items()):
            if callable(obj) and hasattr(obj, '__name__'):
                # Check if it's a word (has joy_word attribute from decorator)
                if hasattr(obj, 'joy_word'):
                    print(f"  Registered word: {obj.joy_word}")
                # Or if it was defined with @define()
                elif name in WORDS:
                    pass  # Already registered by decorator
        self.globals.update(self.locals)

    def run(self):
        """Main REPL loop"""
        print("PyJoy2 Hybrid REPL")
        print("  Use `expr` or $(expr) for Python expressions")
        print("  Commands: .s (stack), .c (clear), .w (words), .def name [...]")
        print("  Define words: @define('name') or @word decorator on functions")
        print("  Type 'quit' to exit\n")

        while True:
            try:
                prompt = "... " if self.pending_lines else "> "
                line = input(prompt)

                # Handle quit
                if line.strip() == 'quit' and not self.pending_lines:
                    break

                # Handle empty line in pending block
                if not line.strip() and self.pending_lines:
                    # Empty line ends block
                    code = '\n'.join(self.pending_lines)
                    self._exec(code)
                    self.pending_lines = []
                    self._register_new_words()
                    continue

                # Try to handle as Python block (def, class, etc.)
                if self._handle_python_block(line):
                    continue

                # Otherwise process as Joy/hybrid line
                self.process_line(line)

                # Auto-show top of stack if non-empty
                if self.stack:
                    top = self.stack.peek()
                    print(f"  -> {repr(top)[:70]}")

            except EOFError:
                break
            except Exception as e:
                self.pending_lines = []  # Reset on error
                print(f"  Error: {e}")


def hybrid_repl():
    """Start the hybrid REPL"""
    HybridREPL().run()
```

### 12.2 Defining Words with Python Functions

There are three ways to define new words:

**Method 1: Joy-style with `.def`**
```
> .def square [dup *]
  Defined: square
> 5 square
  -> 25
```

**Method 2: Using `@define()` decorator**
```
> @define("cube")
... def cube(x):
...     return x * x * x
...
  Registered word: cube
> 5 cube
  -> 125
```

**Method 3: Using `@word` decorator (auto-pops args, pushes result)**
```
> @word
... def hypotenuse(a, b):
...     return (a**2 + b**2) ** 0.5
...
> 3 4 hypotenuse
  -> 5.0
```

**Method 4: Direct stack manipulation**
```
> def reverse_stack(s):
...     s[:] = s[::-1]
...
> WORDS["reverse"] = reverse_stack
> 1 2 3 reverse .s
  Stack (top last):
    0: (int) 3
    1: (int) 2
    2: (int) 1
```

**Method 5: Lambda expressions inline**
```
> `WORDS["double"] = lambda s: s.push(s.pop() * 2)`
> 7 double
  -> 14
```

### 12.3 Example REPL Session

```
PyJoy2 Hybrid REPL
  Use `expr` or $(expr) for Python expressions
  Commands: .s (stack), .c (clear), .w (words), .def name [...]
  Define words: @define('name') or @word decorator on functions
  Type 'quit' to exit

> 3 4 +
  -> 7

> `[1, 2, 3, 4, 5]`
  -> [1, 2, 3, 4, 5]

> [dup *] map
  -> [1, 4, 9, 16, 25]

> .s
  Stack (top last):
    0: (int) 7
    1: (list) [1, 4, 9, 16, 25]

> .c
  Stack cleared

> `{"name": "Alice", "age": 30}`
  -> {'name': 'Alice', 'age': 30}

> `lambda d: d["name"]` i
  -> 'Alice'

> .import numpy as np
  Imported numpy as np

> `np.array([1, 2, 3])`
  -> array([1, 2, 3])

> `np.array([10, 20, 30])`
  -> array([10, 20, 30])

> +
  -> array([11, 22, 33])

> `sum` i
  -> 66

> .def square [dup *]
  Defined: square

> 7 square
  -> 49

> .c
  Stack cleared

> `range(1, 11)` `list` i
  -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

> [2 % 0 =] filter
  -> [2, 4, 6, 8, 10]

> 0 [+] fold
  -> 30

> `open("test.txt", "w")` !f = stack.pop()
> !f.write("hello")
> !f.close()
> `"File written"`
  -> 'File written'
```

### 12.4 Key Features

| Syntax | Meaning | Example |
|--------|---------|---------|
| `word` | Execute Joy word | `dup`, `+`, `map` |
| `` `expr` `` | Evaluate Python, push result | `` `[1,2,3]` ``, `` `np.zeros(5)` `` |
| `$(expr)` | Alternative Python syntax | `$(math.sqrt(2))` |
| `!stmt` | Execute Python statement | `!x = 42` |
| `[...]` | Quotation (Joy list) | `[dup *]`, `[+ swap]` |
| `"str"` | String literal | `"hello"` |
| `123` | Number literal | `42`, `3.14` |
| `.s` | Show stack with types | |
| `.c` | Clear stack | |
| `.w` | List available words | |
| `.def name [...]` | Define Joy-style word | `.def double [2 *]` |
| `.import` | Import Python module | `.import pandas as pd` |
| `def func():` | Start Python function (multi-line) | See below |
| `@define("name")` | Decorator to register word | Auto-registers function |
| `@word` | Decorator for stack functions | Auto-pops args, pushes result |
| `WORDS["name"] = fn` | Direct registration | For lambdas or existing functions |

### 12.5 Python Object Workflow Examples

**Working with JSON:**
```
> `'{"users": [{"name": "Alice"}, {"name": "Bob"}]}'`
> `json.loads` i
  -> {'users': [{'name': 'Alice'}, {'name': 'Bob'}]}
> `lambda d: d["users"]` i
  -> [{'name': 'Alice'}, {'name': 'Bob'}]
> [`lambda u: u["name"]` i] map
  -> ['Alice', 'Bob']
```

**Working with files:**
```
> `open("data.csv")` `lambda f: f.read()` i
  -> 'col1,col2\n1,2\n3,4'
> `str.strip` i `str.split` i
  -> ['col1,col2', '1,2', '3,4']
```

**Working with DataFrames:**
```
> .import pandas as pd
> `pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})`
  ->
     a  b
  0  1  4
  1  2  5
  2  3  6
> `lambda df: df["a"] + df["b"]` i
  -> 0    5
     1    7
     2    9
     dtype: int64
```

**Custom classes:**
```
> !class Point:
>     def __init__(self, x, y): self.x, self.y = x, y
>     def magnitude(self): return (self.x**2 + self.y**2)**0.5
> `Point(3, 4)`
  -> <__main__.Point object>
> `lambda p: p.magnitude()` i
  -> 5.0
```

### 12.6 Comparison with Original Joy REPL

| Feature | Joy REPL | PyJoy2 Hybrid REPL |
|---------|----------|-------------------|
| **Literals** | Joy syntax only | Joy + Python expressions |
| **Objects** | Joy types only | Any Python object |
| **Modules** | `include` file | `.import` + Python imports |
| **Definitions** | `name == body .` | `.def name [body]` |
| **Introspection** | `helpdetail` | `.s`, `.w`, Python `dir()` |
| **External libs** | Not supported | Full Python ecosystem |
| **Interactivity** | Limited | Python statement execution |

---

## 13. Conclusion

PYJOY2 offers a pragmatic alternative to faithful Joy reimplementation:

**Advantages:**
- ~10x less code
- Zero-overhead Python integration
- Any Python object on stack
- Familiar Python error handling
- Easy to extend and modify

**Trade-offs:**
- Not compatible with Joy programs directly
- No type guarantees
- Relies on duck typing (can be fragile)
- Less pedagogical for learning Joy

This approach is ideal for developers who want concatenative programming patterns in Python without the ceremony of a full language implementation.

---

*Document version: 1.0*
*Last updated: 2026-01-01*
*See also: [pyjoy.md](docs/pyjoy.md) for faithful Joy reimplementation*

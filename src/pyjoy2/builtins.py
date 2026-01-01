"""
pyjoy2.builtins - Standard library of Joy words.

Organized by category:
- Stack operations
- Arithmetic
- Comparison
- Logic
- List operations
- Combinators
"""

from __future__ import annotations
from .core import Stack, WORDS, word, define, execute

__all__ = ["register_builtins"]


# ============================================================
# Stack Operations
# ============================================================


@define("dup")
def _dup(s: Stack):
    """X -> X X : Duplicate top of stack."""
    s.dup()


@define("pop")
def _pop(s: Stack):
    """X -> : Remove top of stack."""
    s.pop()


@define("id")
def _id(s: Stack):
    """-> : No operation (identity)."""
    pass


WORDS["nop"] = WORDS["id"]


@define("swap")
def _swap(s: Stack):
    """X Y -> Y X : Exchange top two items."""
    s.swap()


@define("over")
def _over(s: Stack):
    """X Y -> X Y X : Copy second item to top."""
    s.over()


@define("rot")
def _rot(s: Stack):
    """X Y Z -> Y Z X : Rotate top three items."""
    s.rot()


@define("drop")
def _drop(s: Stack):
    """X -> : Alias for pop."""
    s.pop()


@define("nip")
def _nip(s: Stack):
    """X Y -> Y : Remove second item."""
    y = s.pop()
    s.pop()
    s.push(y)


@define("tuck")
def _tuck(s: Stack):
    """X Y -> Y X Y : Copy top under second."""
    y = s.pop()
    x = s.pop()
    s.push(y)
    s.push(x)
    s.push(y)


@define("dupd")
def _dupd(s: Stack):
    """X Y -> X X Y : Duplicate second item."""
    y = s.pop()
    s.dup()
    s.push(y)


@define("swapd")
def _swapd(s: Stack):
    """X Y Z -> Y X Z : Swap under top."""
    z = s.pop()
    s.swap()
    s.push(z)


@define("rollup")
def _rollup(s: Stack):
    """X Y Z -> Z X Y : Roll up."""
    z = s.pop()
    y = s.pop()
    x = s.pop()
    s.push(z)
    s.push(x)
    s.push(y)


@define("rolldown")
def _rolldown(s: Stack):
    """X Y Z -> Y Z X : Roll down (same as rot)."""
    s.rot()


@define("clear")
def _clear(s: Stack):
    """... -> : Clear the stack."""
    s.clear()


WORDS["clr"] = WORDS["clear"]


@define("stack")
def _stack(s: Stack):
    """... -> ... [...] : Push copy of stack as list."""
    s.push(list(s))


@define("unstack")
def _unstack(s: Stack):
    """[...] -> ... : Replace stack with list contents."""
    lst = s.pop()
    s.clear()
    for item in lst:
        s.push(item)


@define("depth")
def _depth(s: Stack):
    """... -> ... N : Push stack depth."""
    s.push(len(s))


# ============================================================
# Arithmetic
# ============================================================


@word
def add(a, b):
    """N1 N2 -> N3 : Addition."""
    return a + b


WORDS["+"] = WORDS["add"]


@word
def sub(a, b):
    """N1 N2 -> N3 : Subtraction."""
    return a - b


WORDS["-"] = WORDS["sub"]


@word
def mul(a, b):
    """N1 N2 -> N3 : Multiplication."""
    return a * b


WORDS["*"] = WORDS["mul"]


@word
def div(a, b):
    """N1 N2 -> N3 : Division."""
    return a / b


WORDS["/"] = WORDS["div"]


@word
def floordiv(a, b):
    """N1 N2 -> N3 : Integer division."""
    return a // b


WORDS["//"] = WORDS["floordiv"]


@word
def mod(a, b):
    """N1 N2 -> N3 : Modulo."""
    return a % b


WORDS["%"] = WORDS["mod"]


@word
def neg(x):
    """N -> N : Negate."""
    return -x


@word
def abs_(x):
    """N -> N : Absolute value."""
    return abs(x)


WORDS["abs"] = WORDS["abs_"]


@word
def sign(x):
    """N -> N : Sign (-1, 0, or 1)."""
    return (x > 0) - (x < 0)


@word
def min_(a, b):
    """N1 N2 -> N : Minimum."""
    return min(a, b)


WORDS["min"] = WORDS["min_"]


@word
def max_(a, b):
    """N1 N2 -> N : Maximum."""
    return max(a, b)


WORDS["max"] = WORDS["max_"]


@word
def succ(x):
    """N -> N : Successor (x + 1)."""
    return x + 1


@word
def pred(x):
    """N -> N : Predecessor (x - 1)."""
    return x - 1


# Aliases
WORDS["inc"] = WORDS["succ"]
WORDS["dec"] = WORDS["pred"]


@word
def pow_(a, b):
    """N1 N2 -> N : Power."""
    return a**b


WORDS["pow"] = WORDS["pow_"]
WORDS["**"] = WORDS["pow_"]


# ============================================================
# Comparison
# ============================================================


@word
def lt(a, b):
    """X Y -> B : Less than."""
    return a < b


WORDS["<"] = WORDS["lt"]


@word
def le(a, b):
    """X Y -> B : Less than or equal."""
    return a <= b


WORDS["<="] = WORDS["le"]


@word
def gt(a, b):
    """X Y -> B : Greater than."""
    return a > b


WORDS[">"] = WORDS["gt"]


@word
def ge(a, b):
    """X Y -> B : Greater than or equal."""
    return a >= b


WORDS[">="] = WORDS["ge"]


@word
def eq(a, b):
    """X Y -> B : Equal."""
    return a == b


WORDS["="] = WORDS["eq"]
WORDS["=="] = WORDS["eq"]


@word
def ne(a, b):
    """X Y -> B : Not equal."""
    return a != b


WORDS["!="] = WORDS["ne"]
WORDS["<>"] = WORDS["ne"]


@word
def cmp(a, b):
    """X Y -> N : Compare (-1, 0, 1)."""
    return (a > b) - (a < b)


# Numeric predicates
@word
def zero_p(x):
    """N -> B : True if zero."""
    return x == 0


WORDS["zero?"] = WORDS["zero_p"]


@word
def pos_p(x):
    """N -> B : True if positive."""
    return x > 0


WORDS["pos?"] = WORDS["pos_p"]
WORDS["positive?"] = WORDS["pos_p"]


@word
def neg_p(x):
    """N -> B : True if negative."""
    return x < 0


WORDS["neg?"] = WORDS["neg_p"]
WORDS["negative?"] = WORDS["neg_p"]


# ============================================================
# Logic
# ============================================================


@word
def not_(x):
    """B -> B : Logical not."""
    return not x


WORDS["not"] = WORDS["not_"]


@word
def and_(a, b):
    """B1 B2 -> B : Logical and."""
    return a and b


WORDS["and"] = WORDS["and_"]


@word
def or_(a, b):
    """B1 B2 -> B : Logical or."""
    return a or b


WORDS["or"] = WORDS["or_"]


@word
def xor(a, b):
    """B1 B2 -> B : Logical xor."""
    return bool(a) != bool(b)


# ============================================================
# List/Sequence Operations
# ============================================================


@word
def first(seq):
    """[X ...] -> X : First element."""
    if hasattr(seq, "__iter__"):
        return next(iter(seq))
    raise TypeError(f"first: not a sequence: {type(seq)}")


@word
def rest(seq):
    """[X ...] -> [...] : All but first element."""
    if isinstance(seq, (list, tuple)):
        return list(seq[1:])
    elif isinstance(seq, str):
        return seq[1:]
    else:
        it = iter(seq)
        next(it)
        return list(it)


@word
def cons(x, seq):
    """X [...] -> [X ...] : Prepend element."""
    if isinstance(seq, str):
        return str(x) + seq
    return [x] + list(seq)


@define("uncons")
def _uncons(s: Stack):
    """[X ...] -> X [...] : Split first and rest."""
    seq = s.pop()
    if isinstance(seq, (list, tuple)):
        s.push(seq[0])
        s.push(list(seq[1:]))
    elif isinstance(seq, str):
        s.push(seq[0])
        s.push(seq[1:])
    else:
        it = iter(seq)
        first = next(it)
        s.push(first)
        s.push(list(it))


@word
def concat(a, b):
    """[...] [...] -> [...] : Concatenate."""
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    return list(a) + list(b)


WORDS["cat"] = WORDS["concat"]


@word
def enconcat(x, a, b):
    """X [A] [B] -> [...] : Concatenate as A ++ [X] ++ B."""
    return list(a) + [x] + list(b)


@word
def size(seq):
    """[...] -> N : Length of sequence."""
    return len(seq)


WORDS["len"] = WORDS["size"]
WORDS["length"] = WORDS["size"]


@word
def null(seq):
    """[...] -> B : True if empty."""
    return len(seq) == 0


WORDS["empty"] = WORDS["null"]
WORDS["empty?"] = WORDS["null"]


@word
def reverse(seq):
    """[...] -> [...] : Reverse sequence."""
    if isinstance(seq, str):
        return seq[::-1]
    return list(reversed(seq))


@word
def at(seq, i):
    """[...] N -> X : Element at index."""
    return seq[i]


@word
def take(n, seq):
    """N [...] -> [...] : Take first n elements."""
    if isinstance(seq, str):
        return seq[:n]
    return list(seq)[:n]


@word
def drop_(n, seq):
    """N [...] -> [...] : Drop first n elements."""
    if isinstance(seq, str):
        return seq[n:]
    return list(seq)[n:]


# Note: drop without underscore is stack drop


@word
def range_(stop):
    """N -> [...] : Range from 0 to N-1."""
    return list(range(stop))


WORDS["range"] = WORDS["range_"]


@word
def range2(start, stop):
    """N1 N2 -> [...] : Range from N1 to N2-1."""
    return list(range(start, stop))


@word
def list_(x):
    """X -> [X] : Wrap in list (or convert iterable)."""
    if isinstance(x, (list, tuple, str, range)):
        return list(x)
    return [x]


WORDS["list"] = WORDS["list_"]


@word
def split(seq, sep):
    """S1 S2 -> [...] : Split string by separator."""
    return seq.split(sep)


@word
def join(seq, sep):
    """[...] S -> S : Join with separator."""
    return sep.join(str(x) for x in seq)


@word
def sort(seq):
    """[...] -> [...] : Sort sequence."""
    return sorted(seq)


@word
def small(seq):
    """[...] -> B : True if list has 0 or 1 elements."""
    return len(seq) <= 1


@word
def sum_(seq):
    """[...] -> N : Sum of sequence."""
    return sum(seq)


WORDS["sum"] = WORDS["sum_"]


@word
def prod(seq):
    """[...] -> N : Product of sequence."""
    result = 1
    for x in seq:
        result *= x
    return result


# ============================================================
# String Operations
# ============================================================


@word
def chars(s):
    """S -> [C...] : String to list of characters."""
    return list(s)


@word
def unchars(lst):
    """[C...] -> S : List of characters to string."""
    return "".join(str(c) for c in lst)


@word
def upper(s):
    """S -> S : Convert to uppercase."""
    return s.upper()


@word
def lower(s):
    """S -> S : Convert to lowercase."""
    return s.lower()


@word
def trim(s):
    """S -> S : Remove leading/trailing whitespace."""
    return s.strip()


@word
def ltrim(s):
    """S -> S : Remove leading whitespace."""
    return s.lstrip()


@word
def rtrim(s):
    """S -> S : Remove trailing whitespace."""
    return s.rstrip()


@word
def starts_with(s, prefix):
    """S P -> B : True if S starts with prefix P."""
    return s.startswith(prefix)


WORDS["starts-with?"] = WORDS["starts_with"]


@word
def ends_with(s, suffix):
    """S X -> B : True if S ends with suffix X."""
    return s.endswith(suffix)


WORDS["ends-with?"] = WORDS["ends_with"]


@word
def replace(s, old, new):
    """S OLD NEW -> S : Replace all occurrences."""
    return s.replace(old, new)


@word
def words(s):
    """S -> [S...] : Split string on whitespace."""
    return s.split()


@word
def unwords(lst):
    """[S...] -> S : Join strings with spaces."""
    return " ".join(str(x) for x in lst)


# ============================================================
# Combinators
# ============================================================


@define("i")
def _i(s: Stack):
    """[P] -> ... : Execute quotation."""
    quot = s.pop()
    execute(s, quot)


WORDS["call"] = WORDS["i"]


@define("x")
def _x(s: Stack):
    """[P] -> ... [P] : Execute without consuming."""
    quot = s.peek()
    execute(s, quot)


@define("dip")
def _dip(s: Stack):
    """X [P] -> ... X : Execute under top."""
    quot = s.pop()
    x = s.pop()
    execute(s, quot)
    s.push(x)


@define("dipd")
def _dipd(s: Stack):
    """X Y [P] -> ... X Y : Execute under top two."""
    quot = s.pop()
    y = s.pop()
    x = s.pop()
    execute(s, quot)
    s.push(x)
    s.push(y)


@define("keep")
def _keep(s: Stack):
    """X [P] -> ... X : Execute and restore top."""
    quot = s.pop()
    x = s.peek()
    execute(s, quot)
    s.push(x)


@define("bi")
def _bi(s: Stack):
    """X [P] [Q] -> ... : Apply P and Q to X."""
    q = s.pop()
    p = s.pop()
    x = s.peek()
    execute(s, p)
    s.push(x)
    execute(s, q)


@define("tri")
def _tri(s: Stack):
    """X [P] [Q] [R] -> ... : Apply P, Q, R to X."""
    r = s.pop()
    q = s.pop()
    p = s.pop()
    x = s.peek()
    execute(s, p)
    s.push(x)
    execute(s, q)
    s.push(x)
    execute(s, r)


@define("ifte")
def _ifte(s: Stack):
    """[B] [T] [F] -> ... : If-then-else."""
    false_branch = s.pop()
    true_branch = s.pop()
    condition = s.pop()

    # Save stack
    saved = list(s)

    # Execute condition
    execute(s, condition)
    result = s.pop()

    # Restore stack
    s.clear()
    s.extend(saved)

    # Execute appropriate branch
    if result:
        execute(s, true_branch)
    else:
        execute(s, false_branch)


@define("branch")
def _branch(s: Stack):
    """B [T] [F] -> ... : Branch on boolean."""
    false_branch = s.pop()
    true_branch = s.pop()
    cond = s.pop()
    execute(s, true_branch if cond else false_branch)


@define("when")
def _when(s: Stack):
    """B [T] -> ... : Execute if true."""
    then_branch = s.pop()
    cond = s.pop()
    if cond:
        execute(s, then_branch)


@define("cond")
def _cond(s: Stack):
    """[[B1] [A1] [B2] [A2] ...] -> ... : Multi-way conditional.

    Alternating [condition] [action] pairs. Evaluates conditions in order
    until one is true, then executes its action.
    Use [true] as the final condition for a default case.

    Example: 5 [[0 <] ["negative"] [0 =] ["zero"] [true] ["positive"]] cond
    """
    clauses = s.pop()

    if len(clauses) % 2 != 0:
        raise ValueError("cond: need even number of elements (condition/action pairs)")

    # Save stack state
    saved = list(s)

    for i in range(0, len(clauses), 2):
        condition = clauses[i]
        action = clauses[i + 1]

        # Restore stack and test condition
        s.clear()
        s.extend(saved)
        execute(s, condition)
        result = s.pop()

        if result:
            # Restore stack and execute action
            s.clear()
            s.extend(saved)
            execute(s, action)
            return

    # No condition matched - restore stack
    s.clear()
    s.extend(saved)


@define("case")
def _case(s: Stack):
    """X [V1 [A1] V2 [A2] ...] -> ... : Pattern match on value.

    Alternating value [action] pairs. Compares X against each value V
    in order. When a match is found, pushes X back and executes the action.
    If no match, X remains on stack unchanged.

    Example: 2 [1 [pop "one"] 2 [pop "two"] 3 [pop "three"]] case
    """
    clauses = s.pop()
    value = s.pop()

    if len(clauses) % 2 != 0:
        raise ValueError("case: need even number of elements (value/action pairs)")

    for i in range(0, len(clauses), 2):
        match_val = clauses[i]
        action = clauses[i + 1]

        if value == match_val:
            s.push(value)
            execute(s, action)
            return

    # No match - push value back
    s.push(value)


@define("unless")
def _unless(s: Stack):
    """B [F] -> ... : Execute if false."""
    else_branch = s.pop()
    cond = s.pop()
    if not cond:
        execute(s, else_branch)


@define("loop")
def _loop(s: Stack):
    """[P] -> ... : Execute while top is true."""
    quot = s.pop()
    while True:
        execute(s, quot)
        if not s.pop():
            break


@define("while")
def _while(s: Stack):
    """[B] [P] -> ... : While B is true, execute P."""
    body = s.pop()
    cond = s.pop()

    while True:
        saved = list(s)
        execute(s, cond)
        result = s.pop()
        if not result:
            s.clear()
            s.extend(saved)
            break
        s.clear()
        s.extend(saved)
        execute(s, body)


@define("times")
def _times(s: Stack):
    """N [P] -> ... : Execute P n times."""
    quot = s.pop()
    n = s.pop()
    for _ in range(n):
        execute(s, quot)


@define("each")
def _each(s: Stack):
    """[...] [P] -> ... : Apply P to each element."""
    quot = s.pop()
    items = s.pop()
    for item in items:
        s.push(item)
        execute(s, quot)


WORDS["step"] = WORDS["each"]


@define("map")
def _map(s: Stack):
    """[...] [P] -> [...] : Transform each element."""
    quot = s.pop()
    items = s.pop()

    results = []
    for item in items:
        s.push(item)
        execute(s, quot)
        results.append(s.pop())

    s.push(results)


@define("filter")
def _filter(s: Stack):
    """[...] [P] -> [...] : Keep elements where P is true."""
    quot = s.pop()
    items = s.pop()

    results = []
    for item in items:
        s.push(item)
        execute(s, quot)
        if s.pop():
            results.append(item)

    s.push(results)


@define("fold")
def _fold(s: Stack):
    """[...] V [P] -> V : Reduce with initial value."""
    quot = s.pop()
    acc = s.pop()
    items = s.pop()

    for item in items:
        s.push(acc)
        s.push(item)
        execute(s, quot)
        acc = s.pop()

    s.push(acc)


WORDS["reduce"] = WORDS["fold"]


@define("any")
def _any(s: Stack):
    """[...] [P] -> B : True if P is true for any element."""
    quot = s.pop()
    items = s.pop()

    for item in items:
        s.push(item)
        execute(s, quot)
        if s.pop():
            s.push(True)
            return

    s.push(False)


@define("all")
def _all(s: Stack):
    """[...] [P] -> B : True if P is true for all elements."""
    quot = s.pop()
    items = s.pop()

    for item in items:
        s.push(item)
        execute(s, quot)
        if not s.pop():
            s.push(False)
            return

    s.push(True)


@define("zip")
def _zip(s: Stack):
    """[...] [...] -> [[...] ...] : Zip two lists."""
    b = s.pop()
    a = s.pop()
    s.push([list(pair) for pair in zip(a, b)])


@define("enumerate")
def _enumerate(s: Stack):
    """[...] -> [[N X] ...] : Enumerate with indices."""
    items = s.pop()
    s.push([[i, x] for i, x in enumerate(items)])


@define("partition")
def _partition(s: Stack):
    """[...] [P] -> [...] [...] : Split list by predicate.

    Elements satisfying P go to first list, others to second.
    """
    quot = s.pop()
    items = s.pop()

    yes = []
    no = []
    for item in items:
        s.push(item)
        execute(s, quot)
        if s.pop():
            yes.append(item)
        else:
            no.append(item)

    s.push(yes)
    s.push(no)


# ============================================================
# Recursion Combinators
# ============================================================


@define("linrec")
def _linrec(s: Stack):
    """[P] [T] [R1] [R2] -> ... : Linear recursion."""
    r2 = s.pop()
    r1 = s.pop()
    then = s.pop()
    pred = s.pop()

    def recurse():
        saved = list(s)
        execute(s, pred)
        result = s.pop()
        s.clear()
        s.extend(saved)

        if result:
            execute(s, then)
        else:
            execute(s, r1)
            recurse()
            execute(s, r2)

    recurse()


@define("binrec")
def _binrec(s: Stack):
    """[P] [T] [R1] [R2] -> ... : Binary recursion."""
    r2 = s.pop()
    r1 = s.pop()
    then = s.pop()
    pred = s.pop()

    def recurse():
        saved = list(s)
        execute(s, pred)
        result = s.pop()
        s.clear()
        s.extend(saved)

        if result:
            execute(s, then)
        else:
            execute(s, r1)
            first_arg = s.pop()

            recurse()
            second_result = s.pop()

            s.push(first_arg)
            recurse()

            s.push(second_result)
            execute(s, r2)

    recurse()


@define("primrec")
def _primrec(s: Stack):
    """N [I] [C] -> ... : Primitive recursion."""
    combine = s.pop()
    initial = s.pop()
    n = s.pop()

    execute(s, initial)
    for i in range(1, n + 1):
        s.push(i)
        execute(s, combine)


# ============================================================
# I/O
# ============================================================


@define("print")
def _print(s: Stack):
    """X -> : Print top of stack."""
    print(s.pop())


@define(".")
def _dot(s: Stack):
    """X -> : Print top of stack (alias)."""
    print(s.pop())


@define("puts")
def _puts(s: Stack):
    """S -> : Print string without newline."""
    print(s.pop(), end="")


@define("show")
def _show(s: Stack):
    """X -> X : Print and keep."""
    print(s.peek())


@define("input")
def _input(s: Stack):
    """-> S : Read line from stdin."""
    s.push(input())


@define("prompt")
def _prompt(s: Stack):
    """S -> S : Prompt and read."""
    prompt_str = s.pop()
    s.push(input(prompt_str))


# ============================================================
# Type Operations
# ============================================================


@word
def type_(x):
    """X -> S : Type name of X."""
    return type(x).__name__


WORDS["type"] = WORDS["type_"]
WORDS["typeof"] = WORDS["type_"]


@word
def int_(x):
    """X -> N : Convert to int."""
    return int(x)


WORDS["int"] = WORDS["int_"]


@word
def float_(x):
    """X -> F : Convert to float."""
    return float(x)


WORDS["float"] = WORDS["float_"]


@word
def str_(x):
    """X -> S : Convert to string."""
    return str(x)


WORDS["str"] = WORDS["str_"]


@word
def bool_(x):
    """X -> B : Convert to boolean."""
    return bool(x)


WORDS["bool"] = WORDS["bool_"]


@word
def repr_(x):
    """X -> S : String representation."""
    return repr(x)


WORDS["repr"] = WORDS["repr_"]


# ============================================================
# Misc
# ============================================================


@define("id")
def _id(s: Stack):
    """X -> X : Identity (do nothing)."""
    pass


@define("apply")
def _apply(s: Stack):
    """X [P] -> ... : Apply P to X (same as swap i)."""
    quot = s.pop()
    execute(s, quot)


@define("compose")
def _compose(s: Stack):
    """[P] [Q] -> [P Q] : Compose two quotations."""
    q = s.pop()
    p = s.pop()
    s.push(list(p) + list(q))


@define("curry")
def _curry(s: Stack):
    """X [P] -> [[X] P] : Curry value into quotation."""
    quot = s.pop()
    x = s.pop()
    s.push([[x]] + list(quot))


@define("dup2")
def _dup2(s: Stack):
    """X Y -> X Y X Y : Duplicate top two."""
    y = s.peek(0)
    x = s.peek(1)
    s.push(x)
    s.push(y)


@define("pop2")
def _pop2(s: Stack):
    """X Y -> : Remove top two."""
    s.pop()
    s.pop()


# ============================================================
# Assertions
# ============================================================


@define("assert")
def _assert(s: Stack):
    """B -> : Assert top is true, raise error if false."""
    value = s.pop()
    if not value:
        raise AssertionError("assertion failed")


@define("assert-eq")
def _assert_eq(s: Stack):
    """X Y -> : Assert X equals Y."""
    y = s.pop()
    x = s.pop()
    if x != y:
        raise AssertionError(f"assertion failed: {x!r} != {y!r}")


WORDS["assert="] = WORDS["assert-eq"]


# ============================================================
# Registration
# ============================================================


def register_builtins():
    """
    Ensure all builtins are registered.
    (They're registered on import, but this is explicit.)
    """
    pass  # Registration happens at module load time

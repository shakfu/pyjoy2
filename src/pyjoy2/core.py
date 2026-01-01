"""
pyjoy2.core - Core stack and word infrastructure.

A practical, Pythonic concatenative language.
"""

from __future__ import annotations
from typing import Callable, Any, cast
import inspect

__all__ = ["Stack", "WORDS", "word", "define", "execute", "Word"]

# Type alias for Joy words
Word = Callable[["Stack"], Any]

# Global word registry
WORDS: dict[str, Word] = {}


class Stack(list):
    """
    Joy evaluation stack.

    A Python list with convenience methods for stack operations.
    Any Python object can be pushed onto the stack.
    """

    def push(self, *items: Any) -> "Stack":
        """Push one or more items onto the stack."""
        self.extend(items)
        return self

    def pop(self, n: int = 1) -> Any:  # type: ignore[override]
        """
        Pop n items from the stack.
        Returns single item if n=1, else tuple of items (top first).
        """
        if n == 1:
            if not self:
                raise IndexError("Stack underflow: pop from empty stack")
            return list.pop(self)
        if len(self) < n:
            raise IndexError(f"Stack underflow: need {n}, have {len(self)}")
        result = tuple(list.pop(self) for _ in range(n))
        return result

    def peek(self, depth: int = 0) -> Any:
        """
        Peek at item at given depth without removing.
        depth=0 is top of stack.
        """
        if depth >= len(self):
            raise IndexError(f"Stack underflow: depth {depth}, size {len(self)}")
        return self[-(depth + 1)]

    def dup(self) -> "Stack":
        """Duplicate top of stack."""
        if not self:
            raise IndexError("Stack underflow: dup on empty stack")
        self.append(self[-1])
        return self

    def swap(self) -> "Stack":
        """Swap top two items."""
        if len(self) < 2:
            raise IndexError("Stack underflow: swap needs 2 items")
        self[-1], self[-2] = self[-2], self[-1]
        return self

    def over(self) -> "Stack":
        """Copy second item to top."""
        if len(self) < 2:
            raise IndexError("Stack underflow: over needs 2 items")
        self.append(self[-2])
        return self

    def rot(self) -> "Stack":
        """Rotate top three: a b c -> b c a"""
        if len(self) < 3:
            raise IndexError("Stack underflow: rot needs 3 items")
        self[-3], self[-2], self[-1] = self[-2], self[-1], self[-3]
        return self

    def drop(self, n: int = 1) -> "Stack":
        """Drop top n items."""
        for _ in range(n):
            self.pop()
        return self

    def depth(self) -> int:
        """Return current stack depth."""
        return len(self)

    def __repr__(self) -> str:
        return f"Stack({list(self)})"


def word(f: Callable[..., Any]) -> Word:
    """
    Decorator that turns a regular function into a stack word.

    Arguments are automatically popped from the stack (right-to-left).
    Return value (if not None) is pushed onto the stack.

    Example:
        @word
        def add(a, b):
            return a + b

        # Usage: 3 4 add -> 7
    """
    sig = inspect.signature(f)
    n_params = len(sig.parameters)
    func_name = cast(str, getattr(f, "__name__", "<unknown>"))

    # Check for stack underflow
    def _check(stack: Stack) -> None:
        if len(stack) < n_params:
            raise IndexError(
                f"{func_name}: needs {n_params} args, stack has {len(stack)}"
            )

    # Generate specialized wrapper based on param count
    if n_params == 0:

        def wrapper(stack: Stack) -> None:
            result = f()
            if result is not None:
                stack.push(result)

    elif n_params == 1:

        def wrapper(stack: Stack) -> None:
            _check(stack)
            a = list.pop(stack)
            result = f(a)
            if result is not None:
                stack.push(result)

    elif n_params == 2:

        def wrapper(stack: Stack) -> None:
            _check(stack)
            b = list.pop(stack)
            a = list.pop(stack)
            result = f(a, b)
            if result is not None:
                stack.push(result)

    elif n_params == 3:

        def wrapper(stack: Stack) -> None:
            _check(stack)
            c = list.pop(stack)
            b = list.pop(stack)
            a = list.pop(stack)
            result = f(a, b, c)
            if result is not None:
                stack.push(result)

    else:
        # Fallback for 4+ params (rare)
        def wrapper(stack: Stack) -> None:
            _check(stack)
            args = tuple(list.pop(stack) for _ in range(n_params))[::-1]
            result = f(*args)
            if result is not None:
                stack.push(result)

    wrapper.__name__ = func_name
    wrapper.__doc__ = f.__doc__
    wrapper._is_word = True  # type: ignore[attr-defined]
    wrapper._n_params = n_params  # type: ignore[attr-defined]

    # Auto-register with function name
    WORDS[func_name] = wrapper

    return wrapper


def define(name: str | None = None) -> Callable[[Callable[..., Any]], Word]:
    """
    Decorator to define and register a Joy word.

    The function receives the stack as its only argument and
    manipulates it directly.

    Example:
        @define("double")
        def _double(stack):
            x = stack.pop()
            stack.push(x * 2)

        # Usage: 5 double -> 10
    """

    def decorator(f: Callable[..., Any]) -> Word:
        word_name = name or cast(str, getattr(f, "__name__", "<unknown>"))

        # Set attributes directly on f, register without wrapper
        f._is_word = True  # type: ignore[attr-defined]
        f.joy_word = word_name  # type: ignore[attr-defined]

        WORDS[word_name] = f
        return f

    return decorator


def execute(stack: Stack, program: Any) -> None:
    """
    Execute a program on the stack.

    A program can be:
    - A callable (word): called with stack
    - A list: each element executed in order
    - Anything else: pushed as literal
    """
    if callable(program):
        program(stack)
    elif isinstance(program, list):
        for item in program:
            if callable(item):
                item(stack)
            elif isinstance(item, list):
                # Quotation - push as-is, don't execute
                stack.push(item)
            else:
                stack.push(item)
    else:
        stack.push(program)

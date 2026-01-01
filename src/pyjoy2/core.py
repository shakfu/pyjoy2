"""
pyjoy2.core - Core stack and word infrastructure.

A practical, Pythonic concatenative language.
"""
from __future__ import annotations
from typing import Callable, Any, Dict, List, Tuple, Union
from functools import wraps
import inspect

__all__ = ['Stack', 'WORDS', 'word', 'define', 'execute', 'Word']

# Type alias for Joy words
Word = Callable[['Stack'], Any]

# Global word registry
WORDS: Dict[str, Word] = {}


class Stack(list):
    """
    Joy evaluation stack.

    A Python list with convenience methods for stack operations.
    Any Python object can be pushed onto the stack.
    """

    def push(self, *items: Any) -> 'Stack':
        """Push one or more items onto the stack."""
        self.extend(items)
        return self

    def pop(self, n: int = 1) -> Any:
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

    def dup(self) -> 'Stack':
        """Duplicate top of stack."""
        if not self:
            raise IndexError("Stack underflow: dup on empty stack")
        self.append(self[-1])
        return self

    def swap(self) -> 'Stack':
        """Swap top two items."""
        if len(self) < 2:
            raise IndexError("Stack underflow: swap needs 2 items")
        self[-1], self[-2] = self[-2], self[-1]
        return self

    def over(self) -> 'Stack':
        """Copy second item to top."""
        if len(self) < 2:
            raise IndexError("Stack underflow: over needs 2 items")
        self.append(self[-2])
        return self

    def rot(self) -> 'Stack':
        """Rotate top three: a b c -> b c a"""
        if len(self) < 3:
            raise IndexError("Stack underflow: rot needs 3 items")
        self[-3], self[-2], self[-1] = self[-2], self[-1], self[-3]
        return self

    def drop(self, n: int = 1) -> 'Stack':
        """Drop top n items."""
        for _ in range(n):
            self.pop()
        return self

    def depth(self) -> int:
        """Return current stack depth."""
        return len(self)

    def __repr__(self) -> str:
        return f"Stack({list(self)})"


def word(f: Callable) -> Word:
    """
    Decorator that turns a regular function into a stack word.

    Arguments are automatically popped from the stack (right-to-left).
    Return value (if not None) is pushed onto the stack.
    If return is a tuple, all items are pushed.

    Example:
        @word
        def add(a, b):
            return a + b

        # Usage: 3 4 add -> 7
    """
    sig = inspect.signature(f)
    n_params = len(sig.parameters)

    @wraps(f)
    def wrapper(stack: Stack) -> None:
        if n_params > 0:
            if len(stack) < n_params:
                raise IndexError(
                    f"{f.__name__}: needs {n_params} args, stack has {len(stack)}"
                )
            # Pop args (they come off in reverse order)
            args = stack.pop(n_params)
            if n_params == 1:
                args = (args,)
            # Reverse to get correct order for function call
            args = args[::-1]
        else:
            args = ()

        result = f(*args)

        if result is not None:
            if isinstance(result, tuple) and hasattr(f, '_push_tuple'):
                # Push each element of tuple
                for item in result:
                    stack.push(item)
            else:
                stack.push(result)

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    wrapper._is_word = True
    wrapper._n_params = n_params

    # Auto-register with function name
    WORDS[f.__name__] = wrapper

    return wrapper


def define(name: str = None):
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
    def decorator(f: Callable) -> Word:
        word_name = name or f.__name__

        @wraps(f)
        def wrapper(stack: Stack) -> None:
            return f(stack)

        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        wrapper._is_word = True
        wrapper.joy_word = word_name

        WORDS[word_name] = wrapper
        return wrapper

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

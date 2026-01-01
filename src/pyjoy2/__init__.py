"""
pyjoy2 - A Pythonic Joy implementation.

A practical concatenative language with full Python interoperability.
Any Python object can live on the stack.

Usage:
    from pyjoy2 import Stack, run, repl, word, define

    # Run a program
    result = run("3 4 + dup *")
    print(result)  # Stack([49])

    # Define custom words
    @word
    def square(x):
        return x * x

    # Start REPL
    repl()
"""

from __future__ import annotations

from .core import Stack, WORDS, word, define, execute, Word
from .parser import parse, tokenize, Token, ParseError
from .repl import HybridREPL, repl, run

# Import builtins to register all words
from . import builtins as _builtins  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    # Core
    "Stack",
    "WORDS",
    "word",
    "define",
    "execute",
    "Word",
    # Parser
    "parse",
    "tokenize",
    "Token",
    "ParseError",
    # REPL
    "HybridREPL",
    "repl",
    "run",
]

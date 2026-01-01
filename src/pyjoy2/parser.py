"""
pyjoy2.parser - Tokenizer and parser for Joy programs.

Supports:
- Numbers: 42, 3.14, -17
- Strings: "hello world"
- Booleans: true, false
- Quotations: [dup *]
- Words: dup, swap, +, my-word
"""
from __future__ import annotations
from typing import List, Any, Iterator, Optional
from dataclasses import dataclass
import re

from .core import WORDS

__all__ = ['parse', 'tokenize', 'Token', 'ParseError']


class ParseError(Exception):
    """Parsing error with location info."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"{message} at line {line}, col {col}")


@dataclass
class Token:
    """A parsed token."""
    type: str
    value: Any
    line: int = 0
    col: int = 0


# Token patterns
PATTERNS = [
    ('COMMENT', r'\(\*.*?\*\)'),           # (* comment *)
    ('COMMENT2', r'#[^\n]*'),              # # comment
    ('FLOAT', r'-?\d+\.\d+(?:[eE][+-]?\d+)?'),
    ('INTEGER', r'-?\d+'),
    ('STRING', r'"(?:[^"\\]|\\.)*"'),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('SEMICOLON', r';'),
    ('PERIOD', r'\.(?!\d)'),               # Not followed by digit
    ('DEFINE', r'=='),
    ('WORD', r'[a-zA-Z_][a-zA-Z0-9_\-]*|[+\-*/<=>&|!?@#$%^~:]+'),
    ('WHITESPACE', r'\s+'),
]

TOKEN_RE = re.compile('|'.join(f'(?P<{n}>{p})' for n, p in PATTERNS))


def tokenize(source: str) -> Iterator[Token]:
    """
    Tokenize source code into tokens.

    Yields Token objects. Skips whitespace and comments.
    """
    line = 1
    line_start = 0

    for match in TOKEN_RE.finditer(source):
        kind = match.lastgroup
        value = match.group()
        col = match.start() - line_start

        # Track line numbers
        newlines = value.count('\n')
        if newlines:
            line += newlines
            line_start = match.end() - len(value.split('\n')[-1])

        # Skip whitespace and comments
        if kind in ('WHITESPACE', 'COMMENT', 'COMMENT2'):
            continue

        # Convert token value
        if kind == 'INTEGER':
            value = int(value)
        elif kind == 'FLOAT':
            value = float(value)
        elif kind == 'STRING':
            value = _unescape(value[1:-1])

        yield Token(kind, value, line, col)


def _unescape(s: str) -> str:
    """Process escape sequences in string."""
    return s.encode('utf-8').decode('unicode_escape')


def parse(source: str) -> List[Any]:
    """
    Parse source code into a program (list of terms).

    A program is a list where:
    - Literals are Python values
    - Quotations are nested lists
    - Words are looked up in WORDS or kept as strings
    """
    tokens = list(tokenize(source))
    return _parse_terms(tokens, 0, set())[0]


def _parse_terms(tokens: List[Token], pos: int,
                 terminators: set) -> tuple[List[Any], int]:
    """Parse sequence of terms until terminator."""
    result = []

    while pos < len(tokens):
        token = tokens[pos]

        if token.type in terminators:
            break

        term, pos = _parse_term(tokens, pos)
        if term is not None:
            result.append(term)

    return result, pos


def _parse_term(tokens: List[Token], pos: int) -> tuple[Any, int]:
    """Parse single term, return (value, new_pos)."""
    if pos >= len(tokens):
        return None, pos

    token = tokens[pos]

    if token.type == 'INTEGER':
        return token.value, pos + 1

    elif token.type == 'FLOAT':
        return token.value, pos + 1

    elif token.type == 'STRING':
        return token.value, pos + 1

    elif token.type == 'LBRACKET':
        # Parse quotation
        inner, new_pos = _parse_terms(tokens, pos + 1, {'RBRACKET'})
        if new_pos >= len(tokens) or tokens[new_pos].type != 'RBRACKET':
            raise ParseError("Expected ']'", token.line, token.col)
        return inner, new_pos + 1

    elif token.type == 'LBRACE':
        # Parse set literal
        inner, new_pos = _parse_terms(tokens, pos + 1, {'RBRACE'})
        if new_pos >= len(tokens) or tokens[new_pos].type != 'RBRACE':
            raise ParseError("Expected '}'", token.line, token.col)
        # Convert to frozenset
        return frozenset(inner), new_pos + 1

    elif token.type == 'WORD':
        value = token.value
        # Handle special words
        if value == 'true':
            return True, pos + 1
        elif value == 'false':
            return False, pos + 1
        elif value == 'nil' or value == 'null':
            return None, pos + 1
        # Look up in WORDS or return as string
        elif value in WORDS:
            return WORDS[value], pos + 1
        else:
            # Return as string - will be looked up at runtime
            return value, pos + 1

    elif token.type in ('SEMICOLON', 'PERIOD'):
        # Statement terminator - skip
        return None, pos + 1

    elif token.type == 'DEFINE':
        # Definition - skip for now
        return None, pos + 1

    else:
        raise ParseError(f"Unexpected token: {token.type}", token.line, token.col)


def parse_and_resolve(source: str) -> List[Any]:
    """
    Parse and resolve all words.

    Unlike parse(), this raises an error for undefined words.
    """
    program = parse(source)
    return _resolve(program)


def _resolve(program: List[Any]) -> List[Any]:
    """Resolve string words to actual functions."""
    result = []
    for item in program:
        if isinstance(item, str):
            if item in WORDS:
                result.append(WORDS[item])
            else:
                raise NameError(f"Undefined word: {item}")
        elif isinstance(item, list):
            result.append(_resolve(item))
        else:
            result.append(item)
    return result

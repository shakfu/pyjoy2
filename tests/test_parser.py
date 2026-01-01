"""
Tests for pyjoy2.parser module.

Tests tokenizer and parser functionality.
"""

import pytest
from pyjoy2.parser import tokenize, parse, Token, ParseError, parse_and_resolve
from pyjoy2.core import WORDS


class TestTokenize:
    """Tests for tokenize function."""

    def test_integer(self):
        tokens = list(tokenize("42"))
        assert len(tokens) == 1
        assert tokens[0].type == "INTEGER"
        assert tokens[0].value == 42

    def test_negative_integer(self):
        tokens = list(tokenize("-17"))
        assert len(tokens) == 1
        assert tokens[0].value == -17

    def test_float(self):
        tokens = list(tokenize("3.14"))
        assert len(tokens) == 1
        assert tokens[0].type == "FLOAT"
        assert tokens[0].value == 3.14

    def test_float_scientific(self):
        tokens = list(tokenize("1.5e10"))
        assert tokens[0].type == "FLOAT"
        assert tokens[0].value == 1.5e10

    def test_string(self):
        tokens = list(tokenize('"hello world"'))
        assert len(tokens) == 1
        assert tokens[0].type == "STRING"
        assert tokens[0].value == "hello world"

    def test_string_with_escape(self):
        tokens = list(tokenize(r'"hello\nworld"'))
        assert tokens[0].value == "hello\nworld"

    def test_word(self):
        tokens = list(tokenize("dup"))
        assert len(tokens) == 1
        assert tokens[0].type == "WORD"
        assert tokens[0].value == "dup"

    def test_word_with_hyphen(self):
        tokens = list(tokenize("my-word"))
        assert tokens[0].type == "WORD"
        assert tokens[0].value == "my-word"

    def test_operator_word(self):
        tokens = list(tokenize("+"))
        assert tokens[0].type == "WORD"
        assert tokens[0].value == "+"

    def test_brackets(self):
        tokens = list(tokenize("[ ]"))
        assert len(tokens) == 2
        assert tokens[0].type == "LBRACKET"
        assert tokens[1].type == "RBRACKET"

    def test_braces(self):
        tokens = list(tokenize("{ }"))
        assert len(tokens) == 2
        assert tokens[0].type == "LBRACE"
        assert tokens[1].type == "RBRACE"

    def test_multiple_tokens(self):
        tokens = list(tokenize("3 4 +"))
        assert len(tokens) == 3
        assert tokens[0].value == 3
        assert tokens[1].value == 4
        assert tokens[2].value == "+"

    def test_comment_parentheses(self):
        tokens = list(tokenize("1 (* comment *) 2"))
        assert len(tokens) == 2
        assert tokens[0].value == 1
        assert tokens[1].value == 2

    def test_comment_hash(self):
        tokens = list(tokenize("1 # comment\n2"))
        assert len(tokens) == 2
        assert tokens[0].value == 1
        assert tokens[1].value == 2

    def test_whitespace_skipped(self):
        tokens = list(tokenize("  1   2  \n  3  "))
        assert len(tokens) == 3
        values = [t.value for t in tokens]
        assert values == [1, 2, 3]

    def test_line_tracking(self):
        tokens = list(tokenize("1\n2\n3"))
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_semicolon(self):
        tokens = list(tokenize("1 ; 2"))
        assert len(tokens) == 3
        assert tokens[1].type == "SEMICOLON"

    def test_period(self):
        tokens = list(tokenize("1 . 2"))
        assert len(tokens) == 3
        assert tokens[1].type == "PERIOD"

    def test_define_operator(self):
        tokens = list(tokenize("square == dup *"))
        assert tokens[1].type == "DEFINE"
        assert tokens[1].value == "=="


class TestParse:
    """Tests for parse function."""

    def test_parse_integer(self):
        program = parse("42")
        assert program == [42]

    def test_parse_float(self):
        program = parse("3.14")
        assert program == [3.14]

    def test_parse_string(self):
        program = parse('"hello"')
        assert program == ["hello"]

    def test_parse_true(self):
        program = parse("true")
        assert program == [True]

    def test_parse_false(self):
        program = parse("false")
        assert program == [False]

    def test_parse_nil(self):
        program = parse("nil")
        assert program == [None]

    def test_parse_null(self):
        program = parse("null")
        assert program == [None]

    def test_parse_quotation(self):
        program = parse("[1 2 3]")
        assert len(program) == 1
        assert program[0] == [1, 2, 3]

    def test_parse_nested_quotation(self):
        program = parse("[[1 2] [3 4]]")
        assert program == [[[1, 2], [3, 4]]]

    def test_parse_empty_quotation(self):
        program = parse("[]")
        assert program == [[]]

    def test_parse_set_literal(self):
        program = parse("{1 2 3}")
        assert len(program) == 1
        assert program[0] == frozenset([1, 2, 3])

    def test_parse_known_word(self):
        """Known words should be resolved to functions."""
        program = parse("dup")
        assert len(program) == 1
        assert callable(program[0])
        assert program[0] is WORDS["dup"]

    def test_parse_unknown_word(self):
        """Unknown words should remain as strings for late binding."""
        program = parse("unknown_word_xyz")
        assert program == ["unknown_word_xyz"]

    def test_parse_multiple_items(self):
        program = parse("1 2 3")
        assert program == [1, 2, 3]

    def test_parse_mixed(self):
        program = parse('42 "hello" [1 2] true')
        assert program[0] == 42
        assert program[1] == "hello"
        assert program[2] == [1, 2]
        assert program[3] is True

    def test_parse_semicolon_skipped(self):
        program = parse("1; 2; 3")
        assert program == [1, 2, 3]

    def test_parse_period_skipped(self):
        program = parse("1. 2. 3")
        assert program == [1, 2, 3]

    def test_parse_unclosed_bracket_raises(self):
        with pytest.raises(ParseError, match="Expected"):
            parse("[1 2 3")

    def test_parse_unclosed_brace_raises(self):
        with pytest.raises(ParseError, match="Expected"):
            parse("{1 2 3")


class TestParseAndResolve:
    """Tests for parse_and_resolve function."""

    def test_resolve_known_words(self):
        program = parse_and_resolve("dup swap")
        assert all(callable(w) for w in program)

    def test_resolve_unknown_word_raises(self):
        with pytest.raises(NameError, match="Undefined word"):
            parse_and_resolve("unknown_xyz")

    def test_resolve_nested_quotation(self):
        program = parse_and_resolve("[dup swap]")
        assert len(program) == 1
        inner = program[0]
        assert all(callable(w) for w in inner)

    def test_resolve_mixed(self):
        program = parse_and_resolve("1 dup 2 swap")
        assert program[0] == 1
        assert callable(program[1])
        assert program[2] == 2
        assert callable(program[3])


class TestParseError:
    """Tests for ParseError exception."""

    def test_parse_error_message(self):
        err = ParseError("test error", line=5, col=10)
        assert "test error" in str(err)
        assert "line 5" in str(err)
        assert "col 10" in str(err)

    def test_parse_error_attributes(self):
        err = ParseError("msg", line=3, col=7)
        assert err.line == 3
        assert err.col == 7


class TestParserEdgeCases:
    """Tests for parser edge cases."""

    def test_parse_define_syntax(self):
        # == should be skipped
        program = parse("foo == bar")
        # 'foo' and 'bar' should be parsed, '==' skipped
        assert "foo" in program or any(
            callable(x) and getattr(x, "joy_word", None) == "foo" for x in program
        )

    def test_resolve_undefined_in_nested_quotation(self):
        # Should raise NameError for undefined word in nested quotation
        with pytest.raises(NameError, match="Undefined word"):
            parse_and_resolve("[undefined_nested_word]")

    def test_parse_empty_input(self):
        program = parse("")
        assert program == []

    def test_parse_only_whitespace(self):
        program = parse("   \n\t  ")
        assert program == []

    def test_parse_only_comments(self):
        program = parse("(* comment *) # another comment")
        assert program == []

    def test_tokenize_multiline(self):
        tokens = list(tokenize("1\n2\n3"))
        values = [t.value for t in tokens]
        assert values == [1, 2, 3]
        # Check line numbers
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3

    def test_unexpected_token_type(self):
        # Create a token with an unexpected type to trigger ParseError
        from pyjoy2.parser import _parse_term

        fake_token = Token(type="UNKNOWN_TYPE", value="?", line=1, col=1)
        with pytest.raises(ParseError, match="Unexpected token"):
            _parse_term([fake_token], 0)

    def test_parse_and_resolve_with_known_word(self):
        # Test that parse_and_resolve correctly resolves known words in quotations
        # This specifically tests line 215 where a string word IS in WORDS
        program = parse_and_resolve("[dup pop swap]")
        assert len(program) == 1
        inner = program[0]
        assert len(inner) == 3
        # All should be resolved to functions
        assert all(callable(w) for w in inner)
        assert inner[0] is WORDS["dup"]
        assert inner[1] is WORDS["pop"]
        assert inner[2] is WORDS["swap"]

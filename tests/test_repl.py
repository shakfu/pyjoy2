"""
Tests for pyjoy2.repl module.

Tests HybridREPL class and run function.
"""

import pytest
from pyjoy2.repl import HybridREPL, run
from pyjoy2.core import Stack, WORDS


class TestRun:
    """Tests for the run() function."""

    def test_run_basic_arithmetic(self):
        result = run("3 4 +")
        assert list(result) == [7]

    def test_run_returns_stack(self):
        result = run("1 2 3")
        assert isinstance(result, Stack)
        assert list(result) == [1, 2, 3]

    def test_run_with_existing_stack(self):
        s = Stack([10])
        result = run("5 +", stack=s)
        assert result is s
        assert list(result) == [15]

    def test_run_multiline(self):
        result = run("""
            1 2 +
            3 *
        """)
        assert list(result) == [9]

    def test_run_comments_ignored(self):
        result = run("""
            # This is a comment
            5 dup *
        """)
        assert list(result) == [25]

    def test_run_quotation(self):
        result = run("[1 2 +] i")
        assert list(result) == [3]

    def test_run_map(self):
        result = run("[1 2 3] [dup *] map")
        assert list(result) == [[1, 4, 9]]

    def test_run_filter(self):
        result = run("[1 2 3 4 5] [2 >] filter")
        assert list(result) == [[3, 4, 5]]

    def test_run_fold(self):
        result = run("[1 2 3 4 5] 0 [+] fold")
        assert list(result) == [15]

    def test_run_nested_quotation(self):
        result = run("[[1 2] [3 4]]")
        assert list(result) == [[[1, 2], [3, 4]]]


class TestHybridREPL:
    """Tests for HybridREPL class."""

    def test_init_creates_stack(self):
        repl = HybridREPL()
        assert isinstance(repl.stack, Stack)
        assert len(repl.stack) == 0

    def test_init_with_stack(self):
        s = Stack([1, 2, 3])
        repl = HybridREPL(stack=s)
        assert repl.stack is s

    def test_init_with_empty_stack(self):
        """Empty stack should not be replaced with new stack."""
        s = Stack()
        repl = HybridREPL(stack=s)
        assert repl.stack is s

    def test_globals_includes_stack(self):
        repl = HybridREPL()
        assert "stack" in repl.globals
        assert "S" in repl.globals
        assert repl.globals["stack"] is repl.stack

    def test_globals_includes_words(self):
        repl = HybridREPL()
        assert "WORDS" in repl.globals
        assert "word" in repl.globals
        assert "define" in repl.globals

    def test_preimported_modules(self):
        repl = HybridREPL()
        assert "math" in repl.globals
        assert "json" in repl.globals
        assert "os" in repl.globals


class TestHybridREPLProcessLine:
    """Tests for HybridREPL.process_line method."""

    def test_process_integer(self):
        repl = HybridREPL()
        repl.process_line("42")
        assert list(repl.stack) == [42]

    def test_process_float(self):
        repl = HybridREPL()
        repl.process_line("3.14")
        assert list(repl.stack) == [3.14]

    def test_process_string(self):
        repl = HybridREPL()
        repl.process_line('"hello"')
        assert list(repl.stack) == ["hello"]

    def test_process_word(self):
        repl = HybridREPL()
        repl.process_line("1 2 +")
        assert list(repl.stack) == [3]

    def test_process_empty_line(self):
        repl = HybridREPL()
        repl.process_line("42")
        repl.process_line("")
        repl.process_line("   ")
        assert list(repl.stack) == [42]

    def test_process_quotation(self):
        repl = HybridREPL()
        repl.process_line("[1 2 3]")
        assert len(repl.stack) == 1
        # The quotation is parsed into an executable list
        assert isinstance(repl.stack.peek(), list)


class TestHybridREPLPythonExpressions:
    """Tests for Python expression syntax in REPL."""

    def test_backtick_expression(self):
        repl = HybridREPL()
        repl.process_line("`2 + 3`")
        assert list(repl.stack) == [5]

    def test_backtick_uses_math(self):
        repl = HybridREPL()
        repl.process_line("`math.sqrt(16)`")
        assert list(repl.stack) == [4.0]

    def test_dollar_expression(self):
        repl = HybridREPL()
        repl.process_line("$(2 ** 10)")
        assert list(repl.stack) == [1024]

    def test_nested_parens_in_dollar(self):
        repl = HybridREPL()
        repl.process_line("$(max(1, min(5, 10)))")
        assert list(repl.stack) == [5]

    def test_mixed_joy_python(self):
        repl = HybridREPL()
        repl.process_line("5 `2 * 3` +")
        assert list(repl.stack) == [11]


class TestHybridREPLPythonStatements:
    """Tests for Python statement syntax in REPL."""

    def test_bang_statement(self):
        repl = HybridREPL()
        repl.process_line("!x = 42")
        assert repl.globals["x"] == 42

    def test_statement_no_push(self):
        repl = HybridREPL()
        repl.process_line("!y = 100")
        assert len(repl.stack) == 0

    def test_use_defined_var(self):
        repl = HybridREPL()
        repl.process_line("!myvar = 7")
        repl.process_line("`myvar * 6`")
        assert list(repl.stack) == [42]


class TestHybridREPLTokenizer:
    """Tests for _tokenize_hybrid method."""

    def test_tokenize_number(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("42")
        assert len(tokens) == 1
        assert tokens[0]["type"] == "literal"
        assert tokens[0]["value"] == 42

    def test_tokenize_float(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("3.14")
        assert tokens[0]["type"] == "literal"
        assert tokens[0]["value"] == 3.14

    def test_tokenize_string(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid('"hello"')
        assert tokens[0]["type"] == "literal"
        assert tokens[0]["value"] == "hello"

    def test_tokenize_word(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("dup")
        assert tokens[0]["type"] == "word"
        assert tokens[0]["value"] == "dup"

    def test_tokenize_quotation(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("[1 2 +]")
        assert tokens[0]["type"] == "quotation"
        assert tokens[0]["value"] == "1 2 +"

    def test_tokenize_nested_quotation(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("[[a] [b]]")
        assert tokens[0]["type"] == "quotation"
        assert tokens[0]["value"] == "[a] [b]"

    def test_tokenize_backtick(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("`x + y`")
        assert tokens[0]["type"] == "python_expr"
        assert tokens[0]["value"] == "x + y"

    def test_tokenize_dollar(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("$(foo())")
        assert tokens[0]["type"] == "python_expr"
        assert tokens[0]["value"] == "foo()"

    def test_tokenize_bang(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("!x = 1")
        assert tokens[0]["type"] == "python_stmt"
        assert tokens[0]["value"] == "x = 1"

    def test_tokenize_boolean_true(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("true")
        assert tokens[0]["type"] == "literal"
        assert tokens[0]["value"] is True

    def test_tokenize_boolean_false(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("false")
        assert tokens[0]["type"] == "literal"
        assert tokens[0]["value"] is False

    def test_tokenize_multiple(self):
        repl = HybridREPL()
        tokens = repl._tokenize_hybrid("1 dup +")
        assert len(tokens) == 3
        assert tokens[0]["value"] == 1
        assert tokens[1]["value"] == "dup"
        assert tokens[2]["value"] == "+"


class TestHybridREPLParseQuotation:
    """Tests for _parse_quotation method."""

    def test_parse_empty(self):
        repl = HybridREPL()
        result = repl._parse_quotation("")
        assert result == []

    def test_parse_literals(self):
        repl = HybridREPL()
        result = repl._parse_quotation("1 2 3")
        assert result == [1, 2, 3]

    def test_parse_known_word(self):
        repl = HybridREPL()
        result = repl._parse_quotation("dup")
        assert len(result) == 1
        assert result[0] is WORDS["dup"]

    def test_parse_unknown_word(self):
        repl = HybridREPL()
        result = repl._parse_quotation("unknown_word_xyz")
        assert result == ["unknown_word_xyz"]

    def test_parse_nested(self):
        repl = HybridREPL()
        result = repl._parse_quotation("[1 2]")
        assert len(result) == 1
        assert result[0] == [1, 2]

    def test_parse_python_expr(self):
        repl = HybridREPL()
        result = repl._parse_quotation("`42`")
        assert len(result) == 1
        # Should be a callable that evaluates the expression
        assert callable(result[0])


class TestHybridREPLCommands:
    """Tests for REPL commands."""

    def test_command_s_empty(self, capsys):
        repl = HybridREPL()
        repl.process_line(".s")
        captured = capsys.readouterr()
        assert "(empty)" in captured.out

    def test_command_s_with_items(self, capsys):
        repl = HybridREPL()
        repl.stack.push(1, 2, 3)
        repl.process_line(".s")
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out

    def test_command_c(self, capsys):
        repl = HybridREPL()
        repl.stack.push(1, 2, 3)
        repl.process_line(".c")
        assert len(repl.stack) == 0
        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower()

    def test_command_w(self, capsys):
        repl = HybridREPL()
        repl.process_line(".w")
        captured = capsys.readouterr()
        assert "words available" in captured.out.lower()
        assert "dup" in captured.out

    def test_command_def(self, capsys):
        repl = HybridREPL()
        repl.process_line(".def square [dup *]")
        captured = capsys.readouterr()
        assert "Defined" in captured.out
        assert "square" in WORDS
        # Test the defined word
        repl.stack.push(5)
        WORDS["square"](repl.stack)
        assert repl.stack.pop() == 25

    def test_command_import(self, capsys):
        repl = HybridREPL()
        repl.process_line(".import collections")
        captured = capsys.readouterr()
        assert "Imported" in captured.out
        assert "collections" in repl.globals


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_factorial_linrec(self):
        result = run("5 [0 =] [pop 1] [dup 1 -] [*] linrec")
        assert list(result) == [120]

    def test_fibonacci_binrec(self):
        # Note: This is an inefficient implementation, just for testing
        result = run("10 [2 <] [] [1 - dup 1 -] [+] binrec")
        assert list(result) == [55]

    def test_quicksort_style(self):
        # Sort using filter
        result = run("[3 1 4 1 5 9 2 6] sort")
        assert list(result) == [[1, 1, 2, 3, 4, 5, 6, 9]]

    def test_python_and_joy_interop(self):
        result = run("`[i**2 for i in range(5)]` [10 <] filter")
        assert list(result) == [[0, 1, 4, 9]]

    def test_custom_word_via_python(self):
        repl = HybridREPL()
        # Define a word using Python in REPL
        # Note: ! statements can't contain [ or ] due to tokenizer limitation
        # So we define a helper function first, then assign it
        repl.process_line("!def _triple(s): s.push(s.pop() * 3)")
        repl.process_line("!WORDS.update({'triple': _triple})")
        repl.process_line("7 triple")
        assert list(repl.stack) == [21]

    def test_complex_pipeline(self):
        # Generate numbers, filter, map, fold
        result = run("""
            10 range
            [2 %] filter
            [dup *] map
            0 [+] fold
        """)
        # Sum of squares of odd numbers 0-9: 1 + 9 + 25 + 49 + 81 = 165
        assert list(result) == [165]

"""
Tests for pyjoy2.core module.

Tests Stack class, word decorator, define decorator, and execute function.
"""

import pytest
from pyjoy2.core import Stack, WORDS, word, define, execute


class TestStack:
    """Tests for Stack class."""

    def test_init_empty(self):
        s = Stack()
        assert len(s) == 0
        assert list(s) == []

    def test_init_from_list(self):
        s = Stack([1, 2, 3])
        assert list(s) == [1, 2, 3]

    def test_push_single(self, stack):
        stack.push(42)
        assert list(stack) == [42]

    def test_push_multiple(self, stack):
        stack.push(1, 2, 3)
        assert list(stack) == [1, 2, 3]

    def test_push_returns_self(self, stack):
        result = stack.push(1)
        assert result is stack

    def test_pop_single(self, populated_stack):
        result = populated_stack.pop()
        assert result == 3
        assert list(populated_stack) == [1, 2]

    def test_pop_multiple(self, populated_stack):
        result = populated_stack.pop(2)
        # Returns tuple (top first, so 3 then 2)
        assert result == (3, 2)
        assert list(populated_stack) == [1]

    def test_pop_empty_raises(self, stack):
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.pop()

    def test_pop_underflow_raises(self, stack):
        stack.push(1)
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.pop(5)

    def test_peek_default(self, populated_stack):
        result = populated_stack.peek()
        assert result == 3
        # Stack unchanged
        assert list(populated_stack) == [1, 2, 3]

    def test_peek_depth(self, populated_stack):
        assert populated_stack.peek(0) == 3
        assert populated_stack.peek(1) == 2
        assert populated_stack.peek(2) == 1

    def test_peek_underflow_raises(self, stack):
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.peek()

    def test_dup(self, stack):
        stack.push(42)
        stack.dup()
        assert list(stack) == [42, 42]

    def test_dup_returns_self(self, stack):
        stack.push(1)
        result = stack.dup()
        assert result is stack

    def test_dup_empty_raises(self, stack):
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.dup()

    def test_swap(self, stack):
        stack.push(1, 2)
        stack.swap()
        assert list(stack) == [2, 1]

    def test_swap_returns_self(self, stack):
        stack.push(1, 2)
        result = stack.swap()
        assert result is stack

    def test_swap_underflow_raises(self, stack):
        stack.push(1)
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.swap()

    def test_over(self, stack):
        stack.push(1, 2)
        stack.over()
        assert list(stack) == [1, 2, 1]

    def test_over_underflow_raises(self, stack):
        stack.push(1)
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.over()

    def test_rot(self, stack):
        stack.push(1, 2, 3)
        stack.rot()
        # a b c -> b c a
        assert list(stack) == [2, 3, 1]

    def test_rot_underflow_raises(self, stack):
        stack.push(1, 2)
        with pytest.raises(IndexError, match="Stack underflow"):
            stack.rot()

    def test_drop(self, populated_stack):
        populated_stack.drop()
        assert list(populated_stack) == [1, 2]

    def test_drop_multiple(self, populated_stack):
        populated_stack.drop(2)
        assert list(populated_stack) == [1]

    def test_depth(self, stack):
        assert stack.depth() == 0
        stack.push(1, 2, 3)
        assert stack.depth() == 3

    def test_repr(self, stack):
        stack.push(1, 2, 3)
        assert repr(stack) == "Stack([1, 2, 3])"

    def test_any_python_object(self, stack):
        """Stack should accept any Python object."""
        obj = {"key": "value"}
        stack.push(obj)
        assert stack.pop() is obj

        stack.push([1, 2, 3])
        stack.push(lambda x: x)
        stack.push(None)
        assert stack.pop() is None


class TestWordDecorator:
    """Tests for @word decorator."""

    def test_word_registers(self, stack):
        @word
        def test_add(a, b):
            return a + b

        assert "test_add" in WORDS
        stack.push(3, 4)
        WORDS["test_add"](stack)
        assert stack.pop() == 7

    def test_word_no_args(self, stack):
        @word
        def push_42():
            return 42

        stack.push(1)
        WORDS["push_42"](stack)
        assert list(stack) == [1, 42]

    def test_word_single_arg(self, stack):
        @word
        def double(x):
            return x * 2

        stack.push(5)
        WORDS["double"](stack)
        assert stack.pop() == 10

    def test_word_none_return(self, stack):
        @word
        def do_nothing(x):
            pass  # Returns None implicitly

        stack.push(5)
        WORDS["do_nothing"](stack)
        # Stack should be empty (5 was popped, nothing pushed)
        assert len(stack) == 0

    def test_word_underflow_raises(self, stack):
        @word
        def needs_two(a, b):
            return a + b

        stack.push(1)  # Only one item
        with pytest.raises(IndexError, match="needs 2 args"):
            WORDS["needs_two"](stack)

    def test_word_preserves_name(self):
        @word
        def my_func(x):
            """My docstring."""
            return x

        assert WORDS["my_func"].__name__ == "my_func"
        assert WORDS["my_func"].__doc__ == "My docstring."


class TestDefineDecorator:
    """Tests for @define decorator."""

    def test_define_registers_with_name(self, stack):
        @define("my-word")
        def _my_word(s):
            s.push(100)

        assert "my-word" in WORDS
        WORDS["my-word"](stack)
        assert stack.pop() == 100

    def test_define_without_name(self, stack):
        @define()
        def another_word(s):
            s.push(200)

        assert "another_word" in WORDS
        WORDS["another_word"](stack)
        assert stack.pop() == 200

    def test_define_receives_stack(self, stack):
        @define("manipulate")
        def _manipulate(s):
            a = s.pop()
            b = s.pop()
            s.push(a + b)
            s.push(a * b)

        stack.push(3, 4)
        WORDS["manipulate"](stack)
        assert list(stack) == [7, 12]

    def test_define_sets_joy_word_attr(self):
        @define("custom-name")
        def _internal(s):
            pass

        assert WORDS["custom-name"].joy_word == "custom-name"


class TestExecute:
    """Tests for execute function."""

    def test_execute_callable(self, stack):
        def push_ten(s):
            s.push(10)

        execute(stack, push_ten)
        assert stack.pop() == 10

    def test_execute_literal(self, stack):
        execute(stack, 42)
        assert stack.pop() == 42

    def test_execute_list_of_literals(self, stack):
        execute(stack, [1, 2, 3])
        assert list(stack) == [1, 2, 3]

    def test_execute_list_with_callable(self, stack):
        def add_one(s):
            s.push(s.pop() + 1)

        execute(stack, [5, add_one])
        assert stack.pop() == 6

    def test_execute_nested_list_pushed(self, stack):
        """Nested lists (quotations) should be pushed, not executed."""
        execute(stack, [[1, 2, 3]])
        result = stack.pop()
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_execute_mixed_program(self, stack):
        @define("inc")
        def _inc(s):
            s.push(s.pop() + 1)

        program = [1, WORDS["inc"], 2, WORDS["inc"]]
        execute(stack, program)
        assert list(stack) == [2, 3]

    def test_execute_string_literal(self, stack):
        execute(stack, "hello")
        assert stack.pop() == "hello"

    def test_execute_none(self, stack):
        execute(stack, None)
        assert stack.pop() is None

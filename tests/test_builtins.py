"""
Tests for pyjoy2.builtins module.

Tests all builtin Joy words.
"""

import pytest
from pyjoy2.core import Stack, WORDS, execute


class TestStackOperations:
    """Tests for stack manipulation words."""

    def test_dup(self, stack):
        stack.push(5)
        WORDS["dup"](stack)
        assert list(stack) == [5, 5]

    def test_pop(self, stack):
        stack.push(1, 2, 3)
        WORDS["pop"](stack)
        assert list(stack) == [1, 2]

    def test_swap(self, stack):
        stack.push(1, 2)
        WORDS["swap"](stack)
        assert list(stack) == [2, 1]

    def test_over(self, stack):
        stack.push(1, 2)
        WORDS["over"](stack)
        assert list(stack) == [1, 2, 1]

    def test_rot(self, stack):
        stack.push(1, 2, 3)
        WORDS["rot"](stack)
        assert list(stack) == [2, 3, 1]

    def test_drop(self, stack):
        stack.push(1, 2, 3)
        WORDS["drop"](stack)
        assert list(stack) == [1, 2]

    def test_nip(self, stack):
        stack.push(1, 2)
        WORDS["nip"](stack)
        assert list(stack) == [2]

    def test_tuck(self, stack):
        stack.push(1, 2)
        WORDS["tuck"](stack)
        assert list(stack) == [2, 1, 2]

    def test_dupd(self, stack):
        stack.push(1, 2)
        WORDS["dupd"](stack)
        assert list(stack) == [1, 1, 2]

    def test_swapd(self, stack):
        stack.push(1, 2, 3)
        WORDS["swapd"](stack)
        assert list(stack) == [2, 1, 3]

    def test_rollup(self, stack):
        stack.push(1, 2, 3)
        WORDS["rollup"](stack)
        assert list(stack) == [3, 1, 2]

    def test_rolldown(self, stack):
        stack.push(1, 2, 3)
        WORDS["rolldown"](stack)
        assert list(stack) == [2, 3, 1]

    def test_clear(self, stack):
        stack.push(1, 2, 3)
        WORDS["clear"](stack)
        assert list(stack) == []

    def test_stack_word(self, stack):
        stack.push(1, 2, 3)
        WORDS["stack"](stack)
        assert list(stack) == [1, 2, 3, [1, 2, 3]]

    def test_unstack(self, stack):
        stack.push([1, 2, 3])
        WORDS["unstack"](stack)
        assert list(stack) == [1, 2, 3]

    def test_depth(self, stack):
        stack.push(1, 2, 3)
        WORDS["depth"](stack)
        assert list(stack) == [1, 2, 3, 3]

    def test_dup2(self, stack):
        stack.push(1, 2)
        WORDS["dup2"](stack)
        assert list(stack) == [1, 2, 1, 2]

    def test_pop2(self, stack):
        stack.push(1, 2, 3)
        WORDS["pop2"](stack)
        assert list(stack) == [1]


class TestArithmetic:
    """Tests for arithmetic words."""

    def test_add(self, stack):
        stack.push(3, 4)
        WORDS["+"](stack)
        assert stack.pop() == 7

    def test_add_alias(self, stack):
        stack.push(3, 4)
        WORDS["add"](stack)
        assert stack.pop() == 7

    def test_sub(self, stack):
        stack.push(10, 3)
        WORDS["-"](stack)
        assert stack.pop() == 7

    def test_mul(self, stack):
        stack.push(6, 7)
        WORDS["*"](stack)
        assert stack.pop() == 42

    def test_div(self, stack):
        stack.push(20, 4)
        WORDS["/"](stack)
        assert stack.pop() == 5.0

    def test_floordiv(self, stack):
        stack.push(7, 2)
        WORDS["//"](stack)
        assert stack.pop() == 3

    def test_mod(self, stack):
        stack.push(7, 3)
        WORDS["%"](stack)
        assert stack.pop() == 1

    def test_neg(self, stack):
        stack.push(5)
        WORDS["neg"](stack)
        assert stack.pop() == -5

    def test_abs(self, stack):
        stack.push(-5)
        WORDS["abs"](stack)
        assert stack.pop() == 5

    def test_sign_positive(self, stack):
        stack.push(42)
        WORDS["sign"](stack)
        assert stack.pop() == 1

    def test_sign_negative(self, stack):
        stack.push(-42)
        WORDS["sign"](stack)
        assert stack.pop() == -1

    def test_sign_zero(self, stack):
        stack.push(0)
        WORDS["sign"](stack)
        assert stack.pop() == 0

    def test_min(self, stack):
        stack.push(5, 3)
        WORDS["min"](stack)
        assert stack.pop() == 3

    def test_max(self, stack):
        stack.push(5, 3)
        WORDS["max"](stack)
        assert stack.pop() == 5

    def test_succ(self, stack):
        stack.push(5)
        WORDS["succ"](stack)
        assert stack.pop() == 6

    def test_pred(self, stack):
        stack.push(5)
        WORDS["pred"](stack)
        assert stack.pop() == 4

    def test_pow(self, stack):
        stack.push(2, 10)
        WORDS["pow"](stack)
        assert stack.pop() == 1024

    def test_pow_operator(self, stack):
        stack.push(2, 10)
        WORDS["**"](stack)
        assert stack.pop() == 1024


class TestComparison:
    """Tests for comparison words."""

    def test_lt_true(self, stack):
        stack.push(3, 5)
        WORDS["<"](stack)
        assert stack.pop() is True

    def test_lt_false(self, stack):
        stack.push(5, 3)
        WORDS["<"](stack)
        assert stack.pop() is False

    def test_le_true(self, stack):
        stack.push(3, 3)
        WORDS["<="](stack)
        assert stack.pop() is True

    def test_gt_true(self, stack):
        stack.push(5, 3)
        WORDS[">"](stack)
        assert stack.pop() is True

    def test_ge_true(self, stack):
        stack.push(5, 5)
        WORDS[">="](stack)
        assert stack.pop() is True

    def test_eq_true(self, stack):
        stack.push(5, 5)
        WORDS["="](stack)
        assert stack.pop() is True

    def test_eq_false(self, stack):
        stack.push(5, 6)
        WORDS["="](stack)
        assert stack.pop() is False

    def test_ne_true(self, stack):
        stack.push(5, 6)
        WORDS["!="](stack)
        assert stack.pop() is True

    def test_cmp_less(self, stack):
        stack.push(3, 5)
        WORDS["cmp"](stack)
        assert stack.pop() == -1

    def test_cmp_greater(self, stack):
        stack.push(5, 3)
        WORDS["cmp"](stack)
        assert stack.pop() == 1

    def test_cmp_equal(self, stack):
        stack.push(5, 5)
        WORDS["cmp"](stack)
        assert stack.pop() == 0


class TestLogic:
    """Tests for logic words."""

    def test_not_true(self, stack):
        stack.push(True)
        WORDS["not"](stack)
        assert stack.pop() is False

    def test_not_false(self, stack):
        stack.push(False)
        WORDS["not"](stack)
        assert stack.pop() is True

    def test_and_true(self, stack):
        stack.push(True, True)
        WORDS["and"](stack)
        assert stack.pop() is True

    def test_and_false(self, stack):
        stack.push(True, False)
        WORDS["and"](stack)
        assert stack.pop() is False

    def test_or_true(self, stack):
        stack.push(False, True)
        WORDS["or"](stack)
        assert stack.pop() is True

    def test_or_false(self, stack):
        stack.push(False, False)
        WORDS["or"](stack)
        assert stack.pop() is False

    def test_xor_true(self, stack):
        stack.push(True, False)
        WORDS["xor"](stack)
        assert stack.pop() is True

    def test_xor_false(self, stack):
        stack.push(True, True)
        WORDS["xor"](stack)
        assert stack.pop() is False


class TestListOperations:
    """Tests for list/sequence words."""

    def test_first(self, stack):
        stack.push([1, 2, 3])
        WORDS["first"](stack)
        assert stack.pop() == 1

    def test_first_string(self, stack):
        stack.push("hello")
        WORDS["first"](stack)
        assert stack.pop() == "h"

    def test_rest(self, stack):
        stack.push([1, 2, 3])
        WORDS["rest"](stack)
        assert stack.pop() == [2, 3]

    def test_rest_string(self, stack):
        stack.push("hello")
        WORDS["rest"](stack)
        assert stack.pop() == "ello"

    def test_cons(self, stack):
        stack.push(0, [1, 2, 3])
        WORDS["cons"](stack)
        assert stack.pop() == [0, 1, 2, 3]

    def test_uncons(self, stack):
        stack.push([1, 2, 3])
        WORDS["uncons"](stack)
        assert list(stack) == [1, [2, 3]]

    def test_concat(self, stack):
        stack.push([1, 2], [3, 4])
        WORDS["concat"](stack)
        assert stack.pop() == [1, 2, 3, 4]

    def test_concat_strings(self, stack):
        stack.push("hello", " world")
        WORDS["concat"](stack)
        assert stack.pop() == "hello world"

    def test_size(self, stack):
        stack.push([1, 2, 3])
        WORDS["size"](stack)
        assert stack.pop() == 3

    def test_null_true(self, stack):
        stack.push([])
        WORDS["null"](stack)
        assert stack.pop() is True

    def test_null_false(self, stack):
        stack.push([1, 2])
        WORDS["null"](stack)
        assert stack.pop() is False

    def test_reverse(self, stack):
        stack.push([1, 2, 3])
        WORDS["reverse"](stack)
        assert stack.pop() == [3, 2, 1]

    def test_reverse_string(self, stack):
        stack.push("hello")
        WORDS["reverse"](stack)
        assert stack.pop() == "olleh"

    def test_at(self, stack):
        stack.push([10, 20, 30], 1)
        WORDS["at"](stack)
        assert stack.pop() == 20

    def test_take(self, stack):
        stack.push(2, [1, 2, 3, 4])
        WORDS["take"](stack)
        assert stack.pop() == [1, 2]

    def test_range(self, stack):
        stack.push(5)
        WORDS["range"](stack)
        assert stack.pop() == [0, 1, 2, 3, 4]

    def test_range2(self, stack):
        stack.push(2, 5)
        WORDS["range2"](stack)
        assert stack.pop() == [2, 3, 4]

    def test_list_wrap(self, stack):
        stack.push(42)
        WORDS["list"](stack)
        assert stack.pop() == [42]

    def test_list_convert(self, stack):
        stack.push((1, 2, 3))
        WORDS["list"](stack)
        assert stack.pop() == [1, 2, 3]

    def test_split(self, stack):
        stack.push("a,b,c", ",")
        WORDS["split"](stack)
        assert stack.pop() == ["a", "b", "c"]

    def test_join(self, stack):
        stack.push(["a", "b", "c"], ",")
        WORDS["join"](stack)
        assert stack.pop() == "a,b,c"

    def test_sort(self, stack):
        stack.push([3, 1, 2])
        WORDS["sort"](stack)
        assert stack.pop() == [1, 2, 3]

    def test_small_empty(self, stack):
        stack.push([])
        WORDS["small"](stack)
        assert stack.pop() is True

    def test_small_one(self, stack):
        stack.push([42])
        WORDS["small"](stack)
        assert stack.pop() is True

    def test_small_two(self, stack):
        stack.push([1, 2])
        WORDS["small"](stack)
        assert stack.pop() is False

    def test_enconcat(self, stack):
        stack.push(5, [1, 2], [8, 9])
        WORDS["enconcat"](stack)
        assert stack.pop() == [1, 2, 5, 8, 9]

    def test_enconcat_empty_lists(self, stack):
        stack.push(42, [], [])
        WORDS["enconcat"](stack)
        assert stack.pop() == [42]

    def test_sum(self, stack):
        stack.push([1, 2, 3, 4])
        WORDS["sum"](stack)
        assert stack.pop() == 10

    def test_prod(self, stack):
        stack.push([1, 2, 3, 4])
        WORDS["prod"](stack)
        assert stack.pop() == 24


class TestStringOperations:
    """Tests for string operation words."""

    def test_chars(self, stack):
        stack.push("hello")
        WORDS["chars"](stack)
        assert stack.pop() == ["h", "e", "l", "l", "o"]

    def test_unchars(self, stack):
        stack.push(["h", "e", "l", "l", "o"])
        WORDS["unchars"](stack)
        assert stack.pop() == "hello"

    def test_upper(self, stack):
        stack.push("hello")
        WORDS["upper"](stack)
        assert stack.pop() == "HELLO"

    def test_lower(self, stack):
        stack.push("HELLO")
        WORDS["lower"](stack)
        assert stack.pop() == "hello"

    def test_trim(self, stack):
        stack.push("  hello  ")
        WORDS["trim"](stack)
        assert stack.pop() == "hello"

    def test_ltrim(self, stack):
        stack.push("  hello  ")
        WORDS["ltrim"](stack)
        assert stack.pop() == "hello  "

    def test_rtrim(self, stack):
        stack.push("  hello  ")
        WORDS["rtrim"](stack)
        assert stack.pop() == "  hello"

    def test_starts_with(self, stack):
        stack.push("hello", "hel")
        WORDS["starts_with"](stack)
        assert stack.pop() is True

    def test_starts_with_false(self, stack):
        stack.push("hello", "xyz")
        WORDS["starts-with?"](stack)
        assert stack.pop() is False

    def test_ends_with(self, stack):
        stack.push("hello", "llo")
        WORDS["ends_with"](stack)
        assert stack.pop() is True

    def test_ends_with_false(self, stack):
        stack.push("hello", "xyz")
        WORDS["ends-with?"](stack)
        assert stack.pop() is False

    def test_replace(self, stack):
        stack.push("hello world", "world", "there")
        WORDS["replace"](stack)
        assert stack.pop() == "hello there"

    def test_words(self, stack):
        stack.push("hello world foo")
        WORDS["words"](stack)
        assert stack.pop() == ["hello", "world", "foo"]

    def test_unwords(self, stack):
        stack.push(["hello", "world", "foo"])
        WORDS["unwords"](stack)
        assert stack.pop() == "hello world foo"


class TestCombinators:
    """Tests for combinator words."""

    def test_i(self, stack):
        stack.push(5, [WORDS["dup"], WORDS["*"]])
        WORDS["i"](stack)
        assert stack.pop() == 25

    def test_x(self, stack):
        """x executes without consuming the quotation."""
        # x peeks the quotation and executes it
        # The quotation stays in place (not moved), results pushed on top
        # Simple test: quotation that pushes a value
        stack.push([42])  # quotation that just pushes 42
        WORDS["x"](stack)
        # After x: 42 is pushed on top, quotation still below
        # Stack: [[42], 42]
        assert stack.pop() == 42
        assert stack.pop() == [42]

    def test_dip(self, stack):
        stack.push(1, 2, [WORDS["dup"]])
        WORDS["dip"](stack)
        assert list(stack) == [1, 1, 2]

    def test_keep(self, stack):
        stack.push(5, [WORDS["dup"], WORDS["*"]])
        WORDS["keep"](stack)
        assert list(stack) == [25, 5]

    def test_bi(self, stack):
        stack.push(5, [WORDS["dup"], WORDS["*"]], [WORDS["dup"], WORDS["+"]])
        WORDS["bi"](stack)
        # 5 squared = 25, 5 doubled = 10
        assert list(stack) == [25, 10]

    def test_ifte_true(self, stack):
        stack.push(10, [5, WORDS[">"]], ["big"], ["small"])
        WORDS["ifte"](stack)
        assert list(stack) == [10, "big"]

    def test_ifte_false(self, stack):
        stack.push(3, [5, WORDS[">"]], ["big"], ["small"])
        WORDS["ifte"](stack)
        assert list(stack) == [3, "small"]

    def test_branch_true(self, stack):
        stack.push(True, ["yes"], ["no"])
        WORDS["branch"](stack)
        assert stack.pop() == "yes"

    def test_branch_false(self, stack):
        stack.push(False, ["yes"], ["no"])
        WORDS["branch"](stack)
        assert stack.pop() == "no"

    def test_when_true(self, stack):
        stack.push(5, True, [WORDS["dup"]])
        WORDS["when"](stack)
        assert list(stack) == [5, 5]

    def test_when_false(self, stack):
        stack.push(5, False, [WORDS["dup"]])
        WORDS["when"](stack)
        assert list(stack) == [5]

    def test_unless_false(self, stack):
        stack.push(5, False, [WORDS["dup"]])
        WORDS["unless"](stack)
        assert list(stack) == [5, 5]

    def test_unless_true(self, stack):
        stack.push(5, True, [WORDS["dup"]])
        WORDS["unless"](stack)
        assert list(stack) == [5]

    def test_cond_first_match(self, stack):
        # Test negative case
        stack.push(-5)
        clauses = [
            [0, WORDS["<"]], ["negative"],
            [0, WORDS["="]], ["zero"],
            [True], ["positive"],
        ]
        stack.push(clauses)
        WORDS["cond"](stack)
        assert list(stack) == [-5, "negative"]

    def test_cond_second_match(self, stack):
        # Test zero case
        stack.push(0)
        clauses = [
            [0, WORDS["<"]], ["negative"],
            [0, WORDS["="]], ["zero"],
            [True], ["positive"],
        ]
        stack.push(clauses)
        WORDS["cond"](stack)
        assert list(stack) == [0, "zero"]

    def test_cond_default(self, stack):
        # Test positive (default) case
        stack.push(5)
        clauses = [
            [0, WORDS["<"]], ["negative"],
            [0, WORDS["="]], ["zero"],
            [True], ["positive"],
        ]
        stack.push(clauses)
        WORDS["cond"](stack)
        assert list(stack) == [5, "positive"]

    def test_cond_no_match(self, stack):
        # No condition matches, stack unchanged
        stack.push(5)
        clauses = [
            [0, WORDS["<"]], ["negative"],
            [0, WORDS["="]], ["zero"],
        ]
        stack.push(clauses)
        WORDS["cond"](stack)
        assert list(stack) == [5]

    def test_case_first_match(self, stack):
        # Match first clause - action consumes value with pop
        stack.push(1)
        clauses = [1, [WORDS["pop"], "one"], 2, [WORDS["pop"], "two"], 3, [WORDS["pop"], "three"]]
        stack.push(clauses)
        WORDS["case"](stack)
        assert list(stack) == ["one"]

    def test_case_second_match(self, stack):
        # Match second clause - action consumes value with pop
        stack.push(2)
        clauses = [1, [WORDS["pop"], "one"], 2, [WORDS["pop"], "two"], 3, [WORDS["pop"], "three"]]
        stack.push(clauses)
        WORDS["case"](stack)
        assert list(stack) == ["two"]

    def test_case_string_match(self, stack):
        # Match on strings - action consumes value with pop
        stack.push("foo")
        clauses = ["bar", [WORDS["pop"], 1], "foo", [WORDS["pop"], 2], "baz", [WORDS["pop"], 3]]
        stack.push(clauses)
        WORDS["case"](stack)
        assert list(stack) == [2]

    def test_case_no_match(self, stack):
        # No match - value stays on stack
        stack.push(99)
        clauses = [1, ["one"], 2, ["two"], 3, ["three"]]
        stack.push(clauses)
        WORDS["case"](stack)
        assert list(stack) == [99]

    def test_case_with_action(self, stack):
        # Action that uses stack
        stack.push(10, 2)
        clauses = [1, [WORDS["pop"]], 2, [WORDS["dup"], WORDS["*"]]]
        stack.push(clauses)
        WORDS["case"](stack)
        assert list(stack) == [10, 4]

    def test_times(self, stack):
        stack.push(0, 5, [1, WORDS["+"]])
        WORDS["times"](stack)
        assert stack.pop() == 5

    def test_each(self, stack):
        results = []

        def capture(s):
            results.append(s.pop())

        stack.push([1, 2, 3], [capture])
        WORDS["each"](stack)
        assert results == [1, 2, 3]

    def test_map(self, stack):
        stack.push([1, 2, 3], [WORDS["dup"], WORDS["*"]])
        WORDS["map"](stack)
        assert stack.pop() == [1, 4, 9]

    def test_filter(self, stack):
        stack.push([1, 2, 3, 4, 5], [2, WORDS[">"]])
        WORDS["filter"](stack)
        assert stack.pop() == [3, 4, 5]

    def test_partition(self, stack):
        stack.push([1, 2, 3, 4, 5], [3, WORDS["<"]])
        WORDS["partition"](stack)
        # Elements < 3 go to first list, elements >= 3 go to second
        assert stack.pop() == [3, 4, 5]
        assert stack.pop() == [1, 2]

    def test_partition_with_pivot(self, stack):
        # Test partition pattern used in qsort: pivot [list] [over <] partition
        stack.push(3, [1, 2, 4, 5], [WORDS["over"], WORDS["<"]])
        WORDS["partition"](stack)
        # Elements < 3 (pivot) go to first list
        assert stack.pop() == [4, 5]
        assert stack.pop() == [1, 2]
        assert stack.pop() == 3  # Pivot is preserved

    def test_fold(self, stack):
        stack.push([1, 2, 3, 4], 0, [WORDS["+"]])
        WORDS["fold"](stack)
        assert stack.pop() == 10

    def test_any_true(self, stack):
        stack.push([1, 2, 3], [2, WORDS["="]])
        WORDS["any"](stack)
        assert stack.pop() is True

    def test_any_false(self, stack):
        stack.push([1, 2, 3], [10, WORDS["="]])
        WORDS["any"](stack)
        assert stack.pop() is False

    def test_all_true(self, stack):
        stack.push([2, 4, 6], [2, WORDS["%"], 0, WORDS["="]])
        WORDS["all"](stack)
        assert stack.pop() is True

    def test_all_false(self, stack):
        stack.push([2, 3, 4], [2, WORDS["%"], 0, WORDS["="]])
        WORDS["all"](stack)
        assert stack.pop() is False

    def test_zip(self, stack):
        stack.push([1, 2, 3], ["a", "b", "c"])
        WORDS["zip"](stack)
        assert stack.pop() == [[1, "a"], [2, "b"], [3, "c"]]

    def test_enumerate(self, stack):
        stack.push(["a", "b", "c"])
        WORDS["enumerate"](stack)
        assert stack.pop() == [[0, "a"], [1, "b"], [2, "c"]]


class TestRecursionCombinators:
    """Tests for recursion combinators."""

    def test_linrec_factorial(self, stack):
        # factorial: n [0 =] [pop 1] [dup 1 -] [*] linrec
        stack.push(5)
        stack.push([0, WORDS["="]])
        stack.push([WORDS["pop"], 1])
        stack.push([WORDS["dup"], 1, WORDS["-"]])
        stack.push([WORDS["*"]])
        WORDS["linrec"](stack)
        assert stack.pop() == 120

    def test_primrec(self, stack):
        # Sum 1 to n: n [0] [+] primrec
        stack.push(5)
        stack.push([0])
        stack.push([WORDS["+"]])
        WORDS["primrec"](stack)
        assert stack.pop() == 15


class TestTypeOperations:
    """Tests for type conversion words."""

    def test_type(self, stack):
        stack.push(42)
        WORDS["type"](stack)
        assert stack.pop() == "int"

    def test_type_list(self, stack):
        stack.push([1, 2, 3])
        WORDS["type"](stack)
        assert stack.pop() == "list"

    def test_int_conversion(self, stack):
        stack.push("42")
        WORDS["int"](stack)
        assert stack.pop() == 42

    def test_float_conversion(self, stack):
        stack.push("3.14")
        WORDS["float"](stack)
        assert stack.pop() == 3.14

    def test_str_conversion(self, stack):
        stack.push(42)
        WORDS["str"](stack)
        assert stack.pop() == "42"

    def test_bool_conversion(self, stack):
        stack.push(0)
        WORDS["bool"](stack)
        assert stack.pop() is False

    def test_repr_conversion(self, stack):
        stack.push("hello")
        WORDS["repr"](stack)
        assert stack.pop() == "'hello'"


class TestMisc:
    """Tests for miscellaneous words."""

    def test_id(self, stack):
        stack.push(42)
        WORDS["id"](stack)
        assert list(stack) == [42]

    def test_apply(self, stack):
        stack.push(5, [WORDS["dup"], WORDS["*"]])
        WORDS["apply"](stack)
        assert stack.pop() == 25

    def test_compose(self, stack):
        stack.push([WORDS["dup"]], [WORDS["*"]])
        WORDS["compose"](stack)
        result = stack.pop()
        assert len(result) == 2

    def test_curry(self, stack):
        stack.push(2, [WORDS["*"]])
        WORDS["curry"](stack)
        result = stack.pop()
        # Result should be [[2] *]
        assert len(result) == 2
        assert result[0] == [2]


class TestEdgeCases:
    """Tests for edge cases and less common paths."""

    def test_first_error_on_non_sequence(self, stack):
        stack.push(42)  # Not a sequence
        with pytest.raises(TypeError, match="not a sequence"):
            WORDS["first"](stack)

    def test_rest_on_iterator(self, stack):
        # Test rest on a general iterator (not list/tuple/str)
        stack.push(iter([1, 2, 3]))
        WORDS["rest"](stack)
        assert stack.pop() == [2, 3]

    def test_cons_on_string(self, stack):
        stack.push("a", "bc")
        WORDS["cons"](stack)
        assert stack.pop() == "abc"

    def test_uncons_on_string(self, stack):
        stack.push("abc")
        WORDS["uncons"](stack)
        assert list(stack) == ["a", "bc"]

    def test_uncons_on_iterator(self, stack):
        # Test uncons on a general iterator
        stack.push(iter([1, 2, 3]))
        WORDS["uncons"](stack)
        assert list(stack) == [1, [2, 3]]

    def test_loop_combinator(self, stack):
        # loop executes quotation repeatedly while top of stack is true
        # This quotation: decrement counter, check if > 0
        # Start with counter 3, decrement until 0
        stack.push(3)
        # Quotation: [1 - dup 0 >] - subtract 1, dup result, check if > 0
        stack.push([1, WORDS["-"], WORDS["dup"], 0, WORDS[">"]])
        WORDS["loop"](stack)
        # After loop: 3 -> 2 -> 1 -> 0, stops when 0 > 0 is False
        assert stack.pop() == 0

    def test_while_combinator(self, stack):
        # while [cond] [body]: increment until >= 5
        stack.push(0)
        stack.push([5, WORDS["<"]])  # condition: n < 5
        stack.push([1, WORDS["+"]])  # body: n + 1
        WORDS["while"](stack)
        assert stack.pop() == 5

    def test_while_false_initially(self, stack):
        # while condition is false from the start
        stack.push(10)
        stack.push([5, WORDS["<"]])  # 10 < 5 is false
        stack.push([1, WORDS["+"]])
        WORDS["while"](stack)
        assert stack.pop() == 10  # unchanged


class TestTypeConversions:
    """Additional type conversion tests."""

    def test_int_from_float(self, stack):
        stack.push(3.7)
        WORDS["int"](stack)
        assert stack.pop() == 3

    def test_float_from_int(self, stack):
        stack.push(42)
        WORDS["float"](stack)
        assert stack.pop() == 42.0

    def test_str_from_list(self, stack):
        stack.push([1, 2, 3])
        WORDS["str"](stack)
        assert stack.pop() == "[1, 2, 3]"

    def test_bool_from_empty_list(self, stack):
        stack.push([])
        WORDS["bool"](stack)
        assert stack.pop() is False

    def test_bool_from_nonempty_list(self, stack):
        stack.push([1])
        WORDS["bool"](stack)
        assert stack.pop() is True

    def test_repr_from_list(self, stack):
        stack.push([1, 2])
        WORDS["repr"](stack)
        assert stack.pop() == "[1, 2]"


class TestUncoveredBuiltins:
    """Tests for builtins with lower coverage."""

    def test_take_on_string(self, stack):
        stack.push(3, "hello")
        WORDS["take"](stack)
        assert stack.pop() == "hel"

    def test_drop_on_string(self, stack):
        stack.push(2, "hello")
        WORDS["drop_"](stack)
        assert stack.pop() == "llo"

    def test_drop_on_list(self, stack):
        stack.push(2, [1, 2, 3, 4])
        WORDS["drop_"](stack)
        assert stack.pop() == [3, 4]

    def test_dipd(self, stack):
        # X Y [P] -> ... X Y
        stack.push(1, 2, 3)
        stack.push([10, WORDS["+"]])  # Add 10 to bottom value
        WORDS["dipd"](stack)
        assert list(stack) == [11, 2, 3]

    def test_tri(self, stack):
        # X [P] [Q] [R] -> ... (apply P, Q, R to X)
        stack.push(5)
        stack.push([WORDS["dup"], WORDS["*"]])  # P: square
        stack.push([2, WORDS["*"]])  # Q: double
        stack.push([1, WORDS["+"]])  # R: increment
        WORDS["tri"](stack)
        # Stack should have: 25 (5*5), 10 (5*2), 6 (5+1)
        assert list(stack) == [25, 10, 6]

    def test_print(self, stack, capsys):
        stack.push("hello")
        WORDS["print"](stack)
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello"
        assert len(list(stack)) == 0  # Consumed

    def test_dot_print(self, stack, capsys):
        stack.push(42)
        WORDS["."](stack)
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert len(list(stack)) == 0

    def test_puts(self, stack, capsys):
        stack.push("no newline")
        WORDS["puts"](stack)
        captured = capsys.readouterr()
        assert captured.out == "no newline"  # No trailing newline

    def test_show(self, stack, capsys):
        stack.push("test")
        WORDS["show"](stack)
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert stack.pop() == "test"  # Value preserved

    def test_input(self, stack, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda: "user input")
        WORDS["input"](stack)
        assert stack.pop() == "user input"

    def test_prompt(self, stack, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda p: f"response to '{p}'")
        stack.push("Enter value: ")
        WORDS["prompt"](stack)
        assert stack.pop() == "response to 'Enter value: '"


class TestQuickWins:
    """Tests for quick win additions."""

    def test_inc(self, stack):
        stack.push(5)
        WORDS["inc"](stack)
        assert stack.pop() == 6

    def test_dec(self, stack):
        stack.push(5)
        WORDS["dec"](stack)
        assert stack.pop() == 4

    def test_zero_predicate(self, stack):
        stack.push(0)
        WORDS["zero?"](stack)
        assert stack.pop() is True

        stack.push(5)
        WORDS["zero?"](stack)
        assert stack.pop() is False

    def test_pos_predicate(self, stack):
        stack.push(5)
        WORDS["pos?"](stack)
        assert stack.pop() is True

        stack.push(-5)
        WORDS["pos?"](stack)
        assert stack.pop() is False

        stack.push(0)
        WORDS["positive?"](stack)
        assert stack.pop() is False

    def test_neg_predicate(self, stack):
        stack.push(-5)
        WORDS["neg?"](stack)
        assert stack.pop() is True

        stack.push(5)
        WORDS["neg?"](stack)
        assert stack.pop() is False

        stack.push(0)
        WORDS["negative?"](stack)
        assert stack.pop() is False

    def test_empty_predicate(self, stack):
        stack.push([])
        WORDS["empty?"](stack)
        assert stack.pop() is True

        stack.push([1, 2])
        WORDS["empty?"](stack)
        assert stack.pop() is False

    def test_id_nop(self, stack):
        stack.push(1, 2, 3)
        WORDS["id"](stack)
        assert list(stack) == [1, 2, 3]

        WORDS["nop"](stack)
        assert list(stack) == [1, 2, 3]

    def test_clr_alias(self, stack):
        stack.push(1, 2, 3)
        WORDS["clr"](stack)
        assert list(stack) == []


class TestAssertions:
    """Tests for assertion words."""

    def test_assert_true(self, stack):
        stack.push(True)
        WORDS["assert"](stack)
        assert list(stack) == []

    def test_assert_false_raises(self, stack):
        stack.push(False)
        with pytest.raises(AssertionError):
            WORDS["assert"](stack)

    def test_assert_truthy(self, stack):
        stack.push(1)
        WORDS["assert"](stack)
        assert list(stack) == []

    def test_assert_eq_equal(self, stack):
        stack.push(5, 5)
        WORDS["assert-eq"](stack)
        assert list(stack) == []

    def test_assert_eq_not_equal_raises(self, stack):
        stack.push(5, 6)
        with pytest.raises(AssertionError) as exc_info:
            WORDS["assert-eq"](stack)
        assert "5" in str(exc_info.value)
        assert "6" in str(exc_info.value)

    def test_assert_eq_alias(self, stack):
        stack.push(3, 3)
        WORDS["assert="](stack)
        assert list(stack) == []

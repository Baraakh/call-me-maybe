"""Unit tests for the character-level grammar primitives."""

from src.constrained_decoding.grammar import (
    Enum,
    GrammarState,
    JSONNumber,
    JSONString,
    Literal,
    Sequence,
)


def _consume(state: GrammarState, text: str) -> GrammarState | None:
    result: GrammarState | None = state
    for char in text:
        assert result is not None
        result = result.step(char)
    return result


class TestLiteral:
    def test_matches_exact_text(self) -> None:
        state = _consume(Literal("abc"), "abc")
        assert state is not None
        assert state.can_exit()

    def test_rejects_wrong_character(self) -> None:
        assert Literal("abc").step("x") is None

    def test_cannot_exit_partway_through(self) -> None:
        state = _consume(Literal("abc"), "ab")
        assert state is not None
        assert not state.can_exit()

    def test_rejects_overshoot_once_complete(self) -> None:
        state = _consume(Literal("ab"), "ab")
        assert state is not None
        assert state.step("c") is None


class TestEnum:
    def test_exact_match_can_exit(self) -> None:
        state = _consume(Enum.from_options(["fn_add", "fn_sub"]), "fn_add")
        assert state is not None
        assert state.can_exit()

    def test_partial_prefix_cannot_exit(self) -> None:
        state = _consume(Enum.from_options(["fn_add", "fn_sub"]), "fn_a")
        assert state is not None
        assert not state.can_exit()

    def test_rejects_character_outside_all_candidates(self) -> None:
        assert Enum.from_options(["abc", "abd"]).step("x") is None

    def test_prefix_of_another_option_can_still_exit(self) -> None:
        # "fn_add" is itself a valid option even though "fn_add_numbers"
        # extends it -- both must remain reachable.
        state = _consume(
            Enum.from_options(["fn_add", "fn_add_numbers"]), "fn_add"
        )
        assert state is not None
        assert state.can_exit()
        longer = state.step("_")
        assert longer is not None
        assert not longer.can_exit()
        completed = _consume(longer, "numbers")
        assert completed is not None
        assert completed.can_exit()


class TestJSONString:
    def test_empty_string_can_exit(self) -> None:
        assert JSONString().can_exit()

    def test_plain_characters_accepted(self) -> None:
        state = _consume(JSONString(), "hello world")
        assert state is not None
        assert state.can_exit()

    def test_unescaped_quote_rejected(self) -> None:
        assert JSONString().step('"') is None

    def test_control_character_rejected(self) -> None:
        assert JSONString().step("\n") is None

    def test_escaped_quote_accepted_and_returns_to_normal(self) -> None:
        state = _consume(JSONString(), '\\"')
        assert state is not None
        assert state.can_exit()

    def test_mid_escape_cannot_exit(self) -> None:
        state = JSONString().step("\\")
        assert state is not None
        assert not state.can_exit()

    def test_invalid_escape_character_rejected(self) -> None:
        state = JSONString().step("\\")
        assert state is not None
        assert state.step("z") is None

    def test_unicode_escape_requires_four_hex_digits(self) -> None:
        state = _consume(JSONString(), "\\u00")
        assert state is not None
        assert not state.can_exit()
        state = _consume(state, "41")
        assert state is not None
        assert state.can_exit()

    def test_unicode_escape_rejects_non_hex(self) -> None:
        state = _consume(JSONString(), "\\u00")
        assert state is not None
        assert state.step("z") is None


class TestJSONNumber:
    def test_single_digit_can_exit(self) -> None:
        state = _consume(JSONNumber(), "3")
        assert state is not None
        assert state.can_exit()

    def test_leading_minus_alone_cannot_exit(self) -> None:
        state = JSONNumber().step("-")
        assert state is not None
        assert not state.can_exit()

    def test_negative_integer(self) -> None:
        state = _consume(JSONNumber(), "-42")
        assert state is not None
        assert state.can_exit()

    def test_decimal_number(self) -> None:
        state = _consume(JSONNumber(), "3.14")
        assert state is not None
        assert state.can_exit()

    def test_trailing_dot_without_digit_cannot_exit(self) -> None:
        state = _consume(JSONNumber(), "3.")
        assert state is not None
        assert not state.can_exit()

    def test_double_dot_rejected(self) -> None:
        state = _consume(JSONNumber(), "3.1")
        assert state is not None
        assert state.step(".") is None

    def test_leading_dot_rejected(self) -> None:
        assert JSONNumber().step(".") is None

    def test_non_digit_rejected_at_start(self) -> None:
        assert JSONNumber().step("a") is None

    def test_integer_only_accepts_plain_digits(self) -> None:
        state = _consume(JSONNumber(integer_only=True), "42")
        assert state is not None
        assert state.can_exit()

    def test_integer_only_accepts_negative(self) -> None:
        state = _consume(JSONNumber(integer_only=True), "-7")
        assert state is not None
        assert state.can_exit()

    def test_integer_only_rejects_decimal_point(self) -> None:
        state = _consume(JSONNumber(integer_only=True), "42")
        assert state is not None
        assert state.step(".") is None


class TestSequence:
    def test_runs_through_fixed_literals(self) -> None:
        seq = Sequence((Literal("{"), Literal("}")))
        state = _consume(seq, "{}")
        assert state is not None
        assert state.can_exit()

    def test_exit_lookahead_hands_off_to_next_primitive(self) -> None:
        seq = Sequence((JSONNumber(), Literal(",")))
        state = _consume(seq, "42,")
        assert state is not None
        assert state.can_exit()

    def test_continues_current_primitive_when_char_extends_it(self) -> None:
        seq = Sequence((JSONNumber(), Literal(",")))
        state = _consume(seq, "4")
        assert state is not None
        # still digits available to extend the number, not the comma yet
        state = state.step("2")
        assert state is not None
        assert not state.can_exit()

    def test_ambiguous_enum_prefix_boundary_via_sequence(self) -> None:
        seq = Sequence((
            Enum.from_options(["fn_add", "fn_add_numbers"]),
            Literal('"'),
        ))
        # closing the shorter name works...
        closed = _consume(seq, 'fn_add"')
        assert closed is not None
        assert closed.can_exit()
        # ...and so does continuing into the longer one.
        extended = _consume(seq, 'fn_add_numbers"')
        assert extended is not None
        assert extended.can_exit()

    def test_rejects_invalid_character_anywhere(self) -> None:
        seq = Sequence((Literal("ab"), Literal("cd")))
        assert seq.step("z") is None

    def test_full_object_grammar_end_to_end(self) -> None:
        seq = Sequence((
            Literal('{"name": "'),
            Enum.from_options(["fn_add_numbers", "fn_greet"]),
            Literal('", "parameters": {"a": '),
            JSONNumber(),
            Literal(', "b": '),
            JSONNumber(),
            Literal("}}"),
        ))
        text = '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}'
        state = _consume(seq, text)
        assert state is not None
        assert state.can_exit()

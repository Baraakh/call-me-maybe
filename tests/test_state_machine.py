"""Unit tests for the parts of StateMachine that don't require loading
the real model."""

import pytest

from src.constrained_decoding.grammar import JSONNumber, JSONString
from src.models import FunctionDefinition, TypeSpec
from src.state_machine import (
    StateMachine,
    StateMachineError,
    _value_primitives,
)


class TestStateMachineEmptyFunctions:
    def test_raises_clear_error_without_loading_the_model(self) -> None:
        with pytest.raises(StateMachineError, match="No functions"):
            StateMachine([])


class TestStateMachineDuplicateFunctionNames:
    def test_raises_clear_error_without_loading_the_model(self) -> None:
        func = FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={"a": TypeSpec(type="number")},
            returns=TypeSpec(type="number"),
        )
        with pytest.raises(StateMachineError, match="same name"):
            StateMachine([func, func])


class TestValuePrimitives:
    def test_integer_type_maps_to_integer_only_number(self) -> None:
        [primitive] = _value_primitives("integer")
        assert isinstance(primitive, JSONNumber)
        state = primitive
        for char in "42":
            stepped = state.step(char)
            assert stepped is not None
            state = stepped
        assert state.can_exit()
        assert state.step(".") is None

    def test_number_type_still_allows_decimals(self) -> None:
        [primitive] = _value_primitives("number")
        assert isinstance(primitive, JSONNumber)
        state = primitive.step("3")
        assert state is not None
        state = state.step(".")
        assert state is not None

    def test_string_type_unchanged(self) -> None:
        primitives = _value_primitives("string")
        assert len(primitives) == 3
        assert isinstance(primitives[1], JSONString)

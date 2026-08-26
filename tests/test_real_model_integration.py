"""Real-model integration test.

Every other test in this suite runs against fakes (see the ``cast(
Small_LLM_Model, ...)`` fakes throughout ``tests/``), so the default
suite stays fast and needs no network access or model download. This
file is the one exception: it loads the actual model and runs the full
``StateMachine`` pipeline end-to-end, to catch anything a fake can't (a
genuinely broken grammar, a tokenizer assumption that doesn't hold in
practice, ...).

Marked ``slow`` and excluded from the default ``pytest`` run (see
``[tool.pytest.ini_options]`` in pyproject.toml). Run it explicitly with:

    uv run pytest -m slow

The first run downloads/caches the model from the Hugging Face Hub if
it isn't already cached locally.
"""

import pytest

from src.io_handling.input_handling import get_funcs_def
from src.models import FunctionDefinition, PromptEntry, TypeSpec
from src.state_machine import StateMachine

pytestmark = pytest.mark.slow


def test_real_model_produces_valid_schema_compliant_results() -> None:
    funcs_def = get_funcs_def("data/input/functions_definition.json")
    state_machine = StateMachine(funcs_def)
    func_names = {func.name for func in funcs_def}
    funcs_by_name = {func.name: func for func in funcs_def}

    prompts = [
        PromptEntry(prompt="What is the sum of 2 and 3?"),
        PromptEntry(prompt="Greet shrek"),
    ]
    results = state_machine.get_func_calls_batch(prompts)

    assert len(results) == len(prompts)
    for prompt, result in zip(prompts, results):
        # structural correctness: this is what's actually guaranteed by
        # constrained decoding, regardless of how good the model's
        # judgement call itself is.
        assert result.prompt == prompt.prompt
        assert result.name in func_names
        chosen = funcs_by_name[result.name]
        assert set(result.parameters) == set(chosen.parameters)

    # decoding is greedy (no sampling), so given the same model and the
    # same sample data, these specific outputs are reproducible -- this
    # is a correctness check on top of the structural one above, tied
    # to this repo's checked-in data/input/functions_definition.json.
    assert results[0].name == "fn_add_numbers"
    assert results[0].parameters == {"a": 2, "b": 3}
    assert results[1].name == "fn_greet"
    assert results[1].parameters == {"name": "shrek"}


def test_real_model_handles_integer_and_boolean_types() -> None:
    # data/input/functions_definition.json only has "string" and "number"
    # parameters, so it never exercises the "integer"/"boolean" grammar
    # paths against the real model -- this function set fills that gap.
    funcs_def = [
        FunctionDefinition(
            name="fn_repeat_string",
            description="Repeat a string a given number of times.",
            parameters={
                "text": TypeSpec(type="string"),
                "times": TypeSpec(type="integer"),
            },
            returns=TypeSpec(type="string"),
        ),
        FunctionDefinition(
            name="fn_set_notifications",
            description="Turn notifications on or off for a user.",
            parameters={
                "username": TypeSpec(type="string"),
                "enabled": TypeSpec(type="boolean"),
            },
            returns=TypeSpec(type="boolean"),
        ),
    ]
    state_machine = StateMachine(funcs_def)

    prompts = [
        PromptEntry(prompt="Repeat the word 'ha' 5 times"),
        PromptEntry(prompt="Turn on notifications for john"),
    ]
    results = state_machine.get_func_calls_batch(prompts)

    assert results[0].name == "fn_repeat_string"
    assert results[0].parameters["text"] == "ha"
    assert isinstance(results[0].parameters["times"], int)
    assert not isinstance(results[0].parameters["times"], bool)
    assert results[0].parameters["times"] == 5

    assert results[1].name == "fn_set_notifications"
    assert results[1].parameters["username"] == "john"
    assert results[1].parameters["enabled"] is True

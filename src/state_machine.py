"""State Machine is the core of the program.

Turns each natural-language prompt into a validated ``FunctionCallResult``
using constrained decoding: the LLM only ever has to generate
``{"name": ..., "parameters": {...}}`` (the ``prompt`` field is filled in
directly from the input, never generated), and every token of that JSON is
chosen from a mask computed by ``constrained_decoding.generator`` against a
grammar built from ``funcs_def`` — see docs/state_machine_plan.md for the
full design.
"""

import json

from llm_sdk import Small_LLM_Model

from .constrained_decoding.generator import (
    ConstrainedDecodingError,
    generate_constrained,
)
from .constrained_decoding.grammar import (
    Enum,
    GrammarState,
    JSONNumber,
    JSONString,
    Literal,
)
from .constrained_decoding.vocab_trie import VocabTrie, build_id_to_text
from .models import FunctionCallResult, FunctionDefinition, PromptEntry

_NAME_PREFIX = '{"name": "'
_NAME_SUFFIX = '", "parameters": {'
_PARAMS_SUFFIX = "}}"


def _describe_functions(funcs_def: list[FunctionDefinition]) -> str:
    """Render the available functions as text for the model's context."""
    lines = ["Available functions:"]
    for func in funcs_def:
        params = ", ".join(
            f"{name}: {spec.type}" for name, spec in func.parameters.items()
        )
        lines.append(
            f"- {func.name}({params}) -> {func.returns.type}: "
            f"{func.description}"
        )
    return "\n".join(lines)


class StateMachineError(Exception):
    """Raised when a function call cannot be generated for a prompt.

    Wraps the lower-level cause (no functions to choose from, or
    ``ConstrainedDecodingError`` — an internal grammar bug that would
    otherwise surface deep inside the generation loop) into a single,
    user-facing error type with a clear message, matching the
    ``InputFileError``/``OutputFileError`` pattern used elsewhere.
    """


def _value_primitives(type_name: str) -> list[GrammarState]:
    if type_name == "string":
        return [Literal('"'), JSONString(), Literal('"')]
    if type_name == "integer":
        return [JSONNumber(integer_only=True)]
    if type_name == "number":
        return [JSONNumber()]
    return [Enum.from_options(["true", "false"])]


class StateMachine:
    """Drives constrained-decoding generation for a fixed set of
    available functions."""

    def __init__(self, funcs_def: list[FunctionDefinition]) -> None:
        if not funcs_def:
            raise StateMachineError(
                "No functions are defined in the functions definition "
                "file; there is nothing to call."
            )
        self._funcs_by_name = {func.name: func for func in funcs_def}
        if len(self._funcs_by_name) != len(funcs_def):
            raise StateMachineError(
                "The functions definition file contains two or more "
                "functions with the same name; function names must be "
                "unique."
            )
        self.funcs_def = funcs_def
        self.llm_client = Small_LLM_Model()
        self.vocab_trie = VocabTrie(build_id_to_text(self.llm_client))
        self._functions_description = _describe_functions(funcs_def)

    def _build_context(self, prompt: PromptEntry) -> list[int]:
        instructions = (
            "You are a function-calling assistant. Read the user request "
            "and pick the single best matching function together with "
            "its argument values.\n"
            f"{self._functions_description}\n\n"
            f"User request: {prompt.prompt}\n"
        )
        input_ids = self.llm_client.encode(instructions)
        return [int(token_id) for token_id in input_ids[0].tolist()]

    def _resolve_function(
        self, context_ids: list[int]
    ) -> tuple[list[int], FunctionDefinition]:
        name_options = list(self._funcs_by_name)
        phase_a: list[GrammarState] = [
            Literal(_NAME_PREFIX),
            Enum.from_options(name_options),
            Literal(_NAME_SUFFIX),
        ]
        context_ids, text = generate_constrained(
            self.llm_client, self.vocab_trie, context_ids, phase_a
        )
        func_name = text[len(_NAME_PREFIX) : -len(_NAME_SUFFIX)]
        return context_ids, self._funcs_by_name[func_name]

    def _generate_parameters(
        self, context_ids: list[int], func_def: FunctionDefinition
    ) -> tuple[list[int], str]:
        phase_b: list[GrammarState] = []
        for index, (param_name, spec) in enumerate(
            func_def.parameters.items()
        ):
            if index > 0:
                phase_b.append(Literal(", "))
            phase_b.append(Literal(f'"{param_name}": '))
            phase_b.extend(_value_primitives(spec.type))
        phase_b.append(Literal(_PARAMS_SUFFIX))
        return generate_constrained(
            self.llm_client, self.vocab_trie, context_ids, phase_b
        )

    def _get_func_call(self, prompt: PromptEntry) -> FunctionCallResult:
        try:
            context_ids = self._build_context(prompt)
            context_ids, func_def = self._resolve_function(context_ids)
            _, params_text = self._generate_parameters(context_ids, func_def)
        except ConstrainedDecodingError as exc:
            raise StateMachineError(
                f"Could not generate a function call for prompt "
                f"{prompt.prompt!r}: {exc}"
            ) from exc

        full_json = f"{_NAME_PREFIX}{func_def.name}{_NAME_SUFFIX}{params_text}"
        parsed = json.loads(full_json)

        return FunctionCallResult(
            prompt=prompt.prompt,
            name=parsed["name"],
            parameters=parsed["parameters"],
        )

    def get_func_calls_batch(
        self, prompts: list[PromptEntry]
    ) -> list[FunctionCallResult]:
        """Generate one validated function call per prompt.

        Args:
            prompts: The natural-language prompts to process.

        Returns:
            One ``FunctionCallResult`` per prompt, in the same order.
        """
        return [self._get_func_call(prompt) for prompt in prompts]

"""Pydantic models for call_me_maybe function-calling project."""

from typing import Literal

from pydantic import BaseModel, TypeAdapter

# ---------------------------------------------------------------------------
# functions_definition.json
# ---------------------------------------------------------------------------


class TypeSpec(BaseModel):
    """Type descriptor — used for both a parameter's type and a function's
    return type, since both are shaped as {"type": "..."} in the schema."""

    type: Literal["string", "number", "boolean"]


class FunctionDefinition(BaseModel):
    """One entry in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


FunctionsDefinitionAdapter = TypeAdapter(list[FunctionDefinition])


# ---------------------------------------------------------------------------
# function_calling_tests.json (input prompts)
# ---------------------------------------------------------------------------


class PromptEntry(BaseModel):
    """One entry in function_calling_tests.json."""

    prompt: str


PromptsAdapter = TypeAdapter(list[PromptEntry])


# ---------------------------------------------------------------------------
# function_calling_results.json (output)
# ---------------------------------------------------------------------------

ParamValue = str | int | float | bool


class FunctionCallResult(BaseModel):
    """One output object — must contain exactly these three keys."""

    prompt: str
    name: str
    parameters: dict[str, ParamValue]


ResultsAdapter = TypeAdapter(list[FunctionCallResult])

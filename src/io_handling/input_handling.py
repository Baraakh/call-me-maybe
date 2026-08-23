"""Handling the input json files.

This module is responsible for reading and validating the two mandatory
input files for the project (``functions_definition.json`` and
``function_calling_tests.json``). Per the project's error-handling
requirements, every failure mode (missing file, unreadable file, file not
encoded as UTF-8, invalid JSON, or JSON that does not match the expected
schema) is caught and re-raised as a single, clear :class:`InputFileError`
with a plain-language message rather than letting the program crash with a
raw traceback.
"""

import pydantic

from ..error_formatting import describe_validation_error
from ..models import (
    FunctionDefinition,
    FunctionsDefinitionAdapter,
    PromptEntry,
    PromptsAdapter,
)


class InputFileError(Exception):
    """Raised when an input file cannot be read or does not match the
    expected schema.

    Wraps the lower-level cause (``OSError`` for missing/unreadable files,
    ``UnicodeDecodeError`` for files that are not valid UTF-8 text,
    ``pydantic.ValidationError`` for malformed or non-conforming JSON) into
    a single, user-facing error type with a clear message.
    """


def get_funcs_def(path: str) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to the ``functions_definition.json`` file.

    Returns:
        The list of parsed and validated function definitions.

    Raises:
        InputFileError: If the file is missing, unreadable, not valid
            UTF-8 text, contains invalid JSON, or does not match the
            expected schema.
    """
    try:
        with open(path, "r", encoding="utf-8") as raw_bytes:
            return FunctionsDefinitionAdapter.validate_json(raw_bytes.read())
    except OSError as exc:
        raise InputFileError(
            f"Could not read functions definition file '{path}': {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise InputFileError(
            f"Could not read functions definition file '{path}': "
            f"the file is not valid UTF-8 text ({exc})"
        ) from exc
    except pydantic.ValidationError as exc:
        raise InputFileError(
            f"Invalid functions definition file '{path}':"
            f"{describe_validation_error(exc)}"
        ) from exc


def get_prompts_entry(path: str) -> list[PromptEntry]:
    """Load and validate the function calling prompts file.

    Args:
        path: Path to the ``function_calling_tests.json`` file.

    Returns:
        The list of parsed and validated prompt entries.

    Raises:
        InputFileError: If the file is missing, unreadable, not valid
            UTF-8 text, contains invalid JSON, or does not match the
            expected schema.
    """
    try:
        with open(path, "r", encoding="utf-8") as raw_bytes:
            return PromptsAdapter.validate_json(raw_bytes.read())
    except OSError as exc:
        raise InputFileError(
            f"Could not read function calling tests file '{path}': {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise InputFileError(
            f"Could not read function calling tests file '{path}': "
            f"the file is not valid UTF-8 text ({exc})"
        ) from exc
    except pydantic.ValidationError as exc:
        raise InputFileError(
            f"Invalid function calling tests file '{path}':"
            f"{describe_validation_error(exc)}"
        ) from exc

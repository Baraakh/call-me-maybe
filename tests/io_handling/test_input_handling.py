"""Tests for :mod:`src.io_handling.input_handling`.

Covers the error-handling contract described in ``docs/call_me_maybe.md``
(Chapter IV.3.1 / V.2): input files may be missing, unreadable, or contain
invalid/non-conforming JSON, and every one of those cases must be surfaced
as a single, clear
:class:`~src.io_handling.input_handling.InputFileError` instead of letting
the program crash with a raw traceback.
"""

import json
from pathlib import Path

import pytest

from src.io_handling.input_handling import (
    InputFileError,
    get_funcs_def,
    get_prompts_entry,
)
from src.models import FunctionDefinition, PromptEntry

VALID_FUNCTIONS_DEFINITION = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
]

VALID_PROMPTS = [
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "Greet shrek"},
]


def write(tmp_path: Path, name: str, content: str) -> Path:
    """Write ``content`` to ``tmp_path / name`` and return the path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# get_funcs_def
# ---------------------------------------------------------------------------


class TestGetFuncsDef:
    def test_valid_file_is_parsed_into_function_definitions(
        self, tmp_path: Path
    ) -> None:
        path = write(
            tmp_path, "functions.json", json.dumps(VALID_FUNCTIONS_DEFINITION)
        )

        result = get_funcs_def(str(path))

        assert len(result) == 2
        assert all(isinstance(item, FunctionDefinition) for item in result)
        assert result[0].name == "fn_add_numbers"
        assert result[0].parameters["a"].type == "number"
        assert result[1].returns.type == "string"

    def test_missing_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(InputFileError) as excinfo:
            get_funcs_def(str(missing))

        # The message should be self-contained: it must tell the user
        # *which* file failed, not just that "something" went wrong.
        assert str(missing) in str(excinfo.value)
        assert excinfo.value.__cause__ is not None

    def test_directory_instead_of_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(InputFileError):
            get_funcs_def(str(tmp_path))

    def test_malformed_json_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = write(tmp_path, "functions.json", "{not valid json")

        with pytest.raises(InputFileError) as excinfo:
            get_funcs_def(str(path))

        assert str(path) in str(excinfo.value)

    def test_empty_file_raises_input_file_error(self, tmp_path: Path) -> None:
        path = write(tmp_path, "functions.json", "")

        with pytest.raises(InputFileError):
            get_funcs_def(str(path))

    def test_json_object_instead_of_array_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = write(
            tmp_path,
            "functions.json",
            json.dumps(VALID_FUNCTIONS_DEFINITION[0]),
        )

        with pytest.raises(InputFileError):
            get_funcs_def(str(path))

    def test_missing_required_field_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        broken = [
            {"name": "fn_add_numbers"}
        ]  # missing description/parameters/returns
        path = write(tmp_path, "functions.json", json.dumps(broken))

        with pytest.raises(InputFileError):
            get_funcs_def(str(path))

    def test_wrong_parameter_type_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        broken = [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {"a": {"type": "not_a_real_type"}},
                "returns": {"type": "number"},
            }
        ]
        path = write(tmp_path, "functions.json", json.dumps(broken))

        with pytest.raises(InputFileError):
            get_funcs_def(str(path))

    def test_non_utf8_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "functions.json"
        path.write_bytes(b"\xff\xfe\x00\x01invalid utf8 \x80\x81")

        with pytest.raises(InputFileError) as excinfo:
            get_funcs_def(str(path))

        assert str(path) in str(excinfo.value)
        assert "utf-8" in str(excinfo.value).lower()
        assert excinfo.value.__cause__ is not None

    def test_validation_error_message_is_human_readable(
        self, tmp_path: Path
    ) -> None:
        broken = [{"name": 123}]  # missing description/parameters/returns
        path = write(tmp_path, "functions.json", json.dumps(broken))

        with pytest.raises(InputFileError) as excinfo:
            get_funcs_def(str(path))

        message = str(excinfo.value)
        # Pydantic's raw ValidationError message links to its own docs and
        # repeats "For further information visit ..." per error; none of
        # that developer-facing noise should leak into the user message.
        assert "pydantic.dev" not in message
        assert "entry 0" in message
        assert "description" in message
        # Each error gets its own bulleted line rather than one long
        # semicolon-joined sentence, so a file with several bad entries
        # stays scannable.
        assert "\n  - entry 0 -> description: " in message


# ---------------------------------------------------------------------------
# get_funcs_prompt
# ---------------------------------------------------------------------------


class TestGetFuncsPrompt:
    def test_valid_file_is_parsed_into_prompt_entries(
        self, tmp_path: Path
    ) -> None:
        path = write(tmp_path, "prompts.json", json.dumps(VALID_PROMPTS))

        result = get_prompts_entry(str(path))

        assert len(result) == 2
        assert all(isinstance(item, PromptEntry) for item in result)
        assert result[0].prompt == "What is the sum of 2 and 3?"

    def test_missing_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(InputFileError) as excinfo:
            get_prompts_entry(str(missing))

        assert str(missing) in str(excinfo.value)
        assert excinfo.value.__cause__ is not None

    def test_directory_instead_of_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(InputFileError):
            get_prompts_entry(str(tmp_path))

    def test_malformed_json_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = write(tmp_path, "prompts.json", '[{"prompt": ]')

        with pytest.raises(InputFileError) as excinfo:
            get_prompts_entry(str(path))

        assert str(path) in str(excinfo.value)

    def test_empty_file_raises_input_file_error(self, tmp_path: Path) -> None:
        path = write(tmp_path, "prompts.json", "")

        with pytest.raises(InputFileError):
            get_prompts_entry(str(path))

    def test_json_object_instead_of_array_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = write(tmp_path, "prompts.json", json.dumps(VALID_PROMPTS[0]))

        with pytest.raises(InputFileError):
            get_prompts_entry(str(path))

    def test_wrong_prompt_type_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        broken = [{"prompt": 12345}]
        path = write(tmp_path, "prompts.json", json.dumps(broken))

        with pytest.raises(InputFileError):
            get_prompts_entry(str(path))

    def test_missing_prompt_key_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        broken = [{"not_prompt": "What is the sum of 2 and 3?"}]
        path = write(tmp_path, "prompts.json", json.dumps(broken))

        with pytest.raises(InputFileError):
            get_prompts_entry(str(path))

    def test_non_utf8_file_raises_input_file_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "prompts.json"
        path.write_bytes(b"\xff\xfe\x00\x01invalid utf8 \x80\x81")

        with pytest.raises(InputFileError) as excinfo:
            get_prompts_entry(str(path))

        assert str(path) in str(excinfo.value)
        assert "utf-8" in str(excinfo.value).lower()
        assert excinfo.value.__cause__ is not None

    def test_validation_error_message_is_human_readable(
        self, tmp_path: Path
    ) -> None:
        broken = [{"prompt": 12345}]
        path = write(tmp_path, "prompts.json", json.dumps(broken))

        with pytest.raises(InputFileError) as excinfo:
            get_prompts_entry(str(path))

        message = str(excinfo.value)
        assert "pydantic.dev" not in message
        assert "entry 0" in message
        assert "prompt" in message
        assert "\n  - entry 0 -> prompt: " in message

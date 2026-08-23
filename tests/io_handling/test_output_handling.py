"""Tests for :mod:`src.io_handling.output_handling`.

Mirrors ``tests/io_handling/test_input_handling.py``: a result value may
fail to serialize to JSON, the output directory may not exist yet (or be
uncreatable), or the file itself may be unwritable — every one of those
cases must be surfaced as a single, clear
:class:`~src.io_handling.output_handling.OutputFileError` instead of
letting the program crash with a raw traceback.
"""

import json
from pathlib import Path

import pytest

from src.io_handling.output_handling import OutputFileError, write_results
from src.models import FunctionCallResult

VALID_RESULTS = [
    FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
    ),
    FunctionCallResult(
        prompt="Greet shrek",
        name="fn_greet",
        parameters={"name": "shrek"},
    ),
]


class TestWriteResults:
    def test_valid_results_are_written_to_file(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"

        write_results(str(path), VALID_RESULTS)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert len(written) == 2
        assert written[0] == {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        }

    def test_missing_output_directory_is_created(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "dir" / "results.json"
        assert not path.parent.exists()

        write_results(str(path), VALID_RESULTS)

        assert path.exists()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert len(written) == 2

    def test_deeply_nested_missing_directories_are_created(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "a" / "b" / "c" / "results.json"

        write_results(str(path), VALID_RESULTS)

        assert path.exists()

    def test_existing_output_directory_is_reused(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        path = out_dir / "results.json"

        write_results(str(path), VALID_RESULTS)

        assert path.exists()

    def test_relative_path_with_no_directory_component_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        write_results("results.json", VALID_RESULTS)

        assert (tmp_path / "results.json").exists()

    def test_empty_results_list_writes_empty_array(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"

        write_results(str(path), [])

        assert json.loads(path.read_text(encoding="utf-8")) == []

    def test_file_path_that_is_a_directory_raises_output_file_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(OutputFileError) as excinfo:
            write_results(str(tmp_path), VALID_RESULTS)

        assert str(tmp_path) in str(excinfo.value)
        assert excinfo.value.__cause__ is not None

    def test_directory_component_that_is_a_file_raises_output_file_error(
        self, tmp_path: Path
    ) -> None:
        blocking_file = tmp_path / "blocking"
        blocking_file.write_text("not a directory", encoding="utf-8")
        path = blocking_file / "results.json"

        with pytest.raises(OutputFileError) as excinfo:
            write_results(str(path), VALID_RESULTS)

        assert str(blocking_file) in str(excinfo.value)
        assert excinfo.value.__cause__ is not None

    def test_unwritable_directory_raises_output_file_error(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "readonly"
        out_dir.mkdir()
        out_dir.chmod(0o500)
        path = out_dir / "results.json"

        try:
            with pytest.raises(OutputFileError) as excinfo:
                write_results(str(path), VALID_RESULTS)
            assert str(path) in str(excinfo.value)
            assert excinfo.value.__cause__ is not None
        finally:
            out_dir.chmod(0o700)

    def test_unserializable_parameter_value_raises_output_file_error(
        self, tmp_path: Path
    ) -> None:
        # model_construct bypasses validation entirely, simulating a
        # FunctionCallResult built by buggy code elsewhere that hands
        # write_results a value none of the ParamValue types can produce.
        path = tmp_path / "results.json"
        broken = [
            FunctionCallResult.model_construct(
                prompt="Greet shrek",
                name="fn_greet",
                parameters={"name": object()},
            )
        ]

        with pytest.raises(OutputFileError) as excinfo:
            write_results(str(path), broken)

        assert excinfo.value.__cause__ is not None
        assert not path.exists()

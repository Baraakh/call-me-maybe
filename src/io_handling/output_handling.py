"""Handling the output json file.

Mirrors the error-handling contract of
:mod:`src.io_handling.input_handling`: every failure mode (a result value
that can't be serialized to JSON, an output directory that doesn't exist
yet or can't be created, a file that can't be written) is caught and
re-raised as a single, clear :class:`OutputFileError` with a
plain-language message rather than letting the program crash with a raw
traceback.
"""

import os

from pydantic_core import PydanticSerializationError

from ..models import FunctionCallResult, ResultsAdapter


class OutputFileError(Exception):
    """Raised when the results cannot be serialized, or the output file/
    directory cannot be written.

    Wraps the lower-level cause (``pydantic_core.PydanticSerializationError``
    for a result containing a value that can't be turned into JSON,
    ``OSError`` for a directory or file that can't be created/written) into
    a single, user-facing error type with a clear message.
    """


def write_results(path: str, results: list[FunctionCallResult]) -> None:
    """Validate and write the function-calling results to disk.

    Args:
        path: Destination path for the output file (e.g.
            ``data/output/function_calling_results.json``). Its parent
            directory is created automatically if it doesn't exist yet.
        results: The list of function-call results to write.

    Raises:
        OutputFileError: If the results cannot be serialized to JSON, the
            output directory cannot be created, or the file cannot be
            written.
    """
    try:
        json_bytes = ResultsAdapter.dump_json(results, indent=2)
    except PydanticSerializationError as exc:
        raise OutputFileError(
            f"Could not serialize results data to JSON: {exc}"
        ) from exc

    output_dir = os.path.dirname(path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            raise OutputFileError(
                f"Could not create output directory '{output_dir}': {exc}"
            ) from exc

    try:
        with open(path, "wb") as f:
            f.write(json_bytes)
    except OSError as exc:
        raise OutputFileError(
            f"Could not write results file '{path}': {exc}"
        ) from exc

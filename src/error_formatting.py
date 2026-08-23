"""Shared helpers for turning pydantic errors into user-facing messages."""

import pydantic


def describe_validation_error(exc: pydantic.ValidationError) -> str:
    """Render a ``pydantic.ValidationError`` as a short, plain-language
    summary.

    ``str(exc)`` is developer-facing: a stack of entries with internal type
    codes and links to pydantic's own documentation. This instead collapses
    each underlying error down to a ``"- <location>: <message>"`` bullet,
    one per line, so someone without Python/pydantic knowledge can tell
    what is wrong and where — and can still scan several bad entries at a
    glance.
    """
    errors = exc.errors()

    if len(errors) == 1 and errors[0]["type"] == "json_invalid":
        return f" the file is not valid JSON ({errors[0]['msg']})"

    lines = []
    for error in errors:
        loc = error["loc"]
        if loc and isinstance(loc[0], int):
            where = f"entry {loc[0]}"
            if len(loc) > 1:
                where += " -> " + ".".join(str(part) for part in loc[1:])
        else:
            where = ".".join(str(part) for part in loc) or "top level"

        msg = (
            "this field is required"
            if error["type"] == "missing"
            else error["msg"]
        )
        lines.append(f"  - {where}: {msg}")

    return "\n" + "\n".join(lines)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a 42 school project ("call_me_maybe"): a function-calling tool that turns natural-language
prompts into structured function calls (`{"name": ..., "parameters": {...}}`) for a small local LLM
(default: `Qwen/Qwen3-0.6B`). The full assignment spec lives in `docs/call_me_maybe_organized.md` —
read it before making architectural decisions, since it defines hard grading requirements, not just
suggestions.

The core technical requirement: output JSON must be produced via **constrained decoding** (masking
invalid tokens in the logits returned by the LLM SDK before selecting the next token), not by
prompting the model and hoping it emits valid JSON. Function selection must also be done by the LLM
itself, not by heuristics/keyword matching.

## Commands

Dependency management is via `uv`, with `llm_sdk` as a workspace member (`pyproject.toml`
`[tool.uv.workspace]`).

```bash
uv sync                 # install dependencies (what the grader/moulinette runs)
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

- Defaults: reads `data/input/functions_definition.json` and `data/input/function_calling_tests.json`,
  writes `data/output/function_calling_results.json`.
- The project's `Makefile` and `README.md` are currently empty placeholders. Per the assignment spec
  (Chapter IV.2), the `Makefile` is required to define `install`, `run`, `debug` (via `pdb`), `clean`
  (removes `__pycache__`, `.mypy_cache`), `lint`, and optionally `lint-strict`:
  - `lint`: `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
  - `lint-strict`: `flake8 .` and `mypy . --strict`
- `ruff.toml` sets `line-length = 79` (flake8-compatible), even though flake8 is the mandated linter.

No test runner is configured yet (`tests/__init__.py` is empty); `pytest` is listed as a dev dependency.

## Architecture

- `src/models.py` — Pydantic models and `TypeAdapter`s for all three JSON contracts:
  `FunctionDefinition` (functions_definition.json), `PromptEntry` (function_calling_tests.json), and
  `FunctionCallResult` (the output). `TypeSpec` restricts parameter/return types to
  `"string" | "number" | "boolean"`. The assignment requires **every** class to use Pydantic.
- `src/input_handling.py` — loads and validates the two input files, translating both `OSError`
  (missing/unreadable file) and `pydantic.ValidationError` (malformed/non-conforming JSON) into a
  single `InputFileError` so `src/__main__.py` never has to deal with raw exceptions. Follow this
  pattern for any new I/O: catch at the boundary, re-raise as one project-specific error type with a
  clear message — the program must never crash with an unhandled traceback.
- `src/__main__.py` — CLI entry point (currently a minimal skeleton wiring up input loading; the
  constrained-decoding generation pipeline and output writing are not yet implemented).
- `src/sandbox.py` — scratch/experimental script (not part of the pipeline) demonstrating raw
  `llm_sdk` usage: encoding, greedy next-token selection from `get_logits_from_input_ids`, and
  decoding — useful as a reference for how the SDK's primitives compose, but not something `__main__`
  should import.
- `llm_sdk/` — a separate uv workspace package (its own `pyproject.toml`) providing `Small_LLM_Model`,
  a thin wrapper around a Hugging Face causal LM. Public surface: `encode`, `decode` (optional per
  spec), `get_logits_from_input_ids`, `get_path_to_vocab_file`, `get_path_to_merges_file`,
  `get_path_to_tokenizer_file`. **Only public methods/attributes of this package may be used** — the
  assignment explicitly forbids reaching into its private members. The vocab file (via
  `get_path_to_vocab_file`) is what maps token IDs to strings for building the constrained-decoding
  token mask.

## Constraints from the assignment spec

These are grading requirements, not stylistic preferences — deviating from them will fail review:

- Python ≥3.10; must pass `flake8` and `mypy` (see lint flags above) with full type hints on
  parameters, returns, and variables.
- Constrained decoding must enforce both JSON structural validity *and* schema compliance
  (correct keys, correct types per `functions_definition.json`) at the token level — not just
  validate/retry after generation.
- Forbidden dependencies: `dspy`, `pytorch`-based structured-output libs (`outlines`), or similar.
  `numpy`, `json`, and `pydantic` are the expected tools beyond `llm_sdk` itself.
- Output file must contain exactly the keys `prompt`, `name`, `parameters` per entry — no extras.
- `data/output/` must not be committed (already gitignored).
- Don't hardcode against the example `data/input/` files — the grader swaps them in for review.

*This project has been created as part of the 42 curriculum by bkhilo.*

# call_me_maybe

## Description

`call_me_maybe` is a function-calling tool that turns natural-language
prompts into structured, machine-executable function calls. Given a
prompt like *"What is the sum of 2 and 3?"* and a set of available
functions described in `data/input/functions_definition.json`, it
produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

Both the function name and its arguments are chosen by a small local
LLM (`Qwen/Qwen3-0.6B`, via the `llm_sdk` package). The distinguishing
part of the project is *how* the output is guaranteed to always be
valid: instead of prompting the model and hoping it emits well-formed
JSON, every token the model is allowed to produce is restricted, at
the logit level, to only the tokens that keep the output both
structurally valid JSON and compliant with the function's schema —
**constrained decoding**. The model never gets the chance to write
something invalid in the first place.

## Instructions

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
make install   # uv sync
make run       # uv run python -m src
make debug     # run under pdb
make lint      # flake8 + mypy
make clean     # remove __pycache__ and .mypy_cache
```

Or directly:

```bash
uv sync
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

By default the program reads `data/input/functions_definition.json`
and `data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. All three paths can be
overridden with the flags above.

## Algorithm Explanation

Text generation with an LLM works one token at a time: at every step
the model scores every entry in its ~150,000-token vocabulary, and
normally the highest-scoring one is picked. Constrained decoding
intervenes in that scoring step — before a token is picked, the set of
vocabulary entries that would keep the output legal (given what's
already been written and the schema still to satisfy) is computed, and
every other entry's score is forced to `-infinity`. The model still
chooses among the remaining options, but it is mathematically
incapable of producing an invalid token.

The output the model has to generate for each prompt is just
`{"name": ..., "parameters": {...}}` — the `prompt` field is copied
from the input directly, never generated, since there's no reason to
let free-form text generation anywhere near a field that's already
known.

Generation runs in two phases, driven by `src/state_machine.py`:

1. **Phase A — which function?** The grammar allows only
   `{"name": "` + one of the known function names + `", "parameters": {`.
   The model picks among the available function names one token at a
   time (ambiguous prefixes, e.g. `fn_add` vs. `fn_add_numbers`, are
   kept open until the model's own choice resolves them).
2. **Phase B — which arguments?** Once the function name is fully
   written, its exact parameter list and types are known, so a new,
   narrow grammar is built for just those fields (`string` values are
   wrapped in quotes and restricted to legal JSON string content,
   `number`/`integer` values are restricted to legal JSON number
   syntax, `boolean` values are restricted to `true`/`false`). There is
   no point in this phase where an unexpected key, a missing key, or a
   wrong type is reachable — the shape is fixed by construction.

Two supporting pieces make this practical:

- `src/constrained_decoding/grammar.py` — four composable primitives
  (`Literal`, `Enum`, `JSONString`, `JSONNumber`) that express "what's
  legal to write next" and chain into the two grammars above.
- `src/constrained_decoding/vocab_trie.py` — the model's ~150,000-entry
  vocabulary, organized once into a trie so that "which vocabulary
  entries are legal right now" can be answered by walking only the
  matching branches, instead of scanning the whole vocabulary at every
  step.

`src/constrained_decoding/generator.py` ties these together: at each
step it asks the grammar which tokens are legal, skips the model
entirely when only one token is legal (most JSON punctuation and key
names have exactly one legal continuation), and otherwise asks the
model to score the legal set and picks the winner.

## Design Decisions

- **Two independent grammar phases instead of one combined grammar.**
  The set of legal parameters isn't known until the function name is
  fully resolved, so building one grammar upfront that already knows
  every function's parameters would either be wrong (parameters from
  the *wrong* function would be reachable) or need to be pruned live.
  Building Phase B fresh, from the resolved function definition, keeps
  each phase simple and impossible to get wrong.
- **A vocabulary trie instead of scanning the vocabulary per token.**
  With ~150,000 tokens and dozens of generation steps per prompt, a
  linear scan at every step is wasteful; a trie lets most steps only
  visit the handful of branches that could possibly be legal.
- **Skipping model inference when only one token is legal.** Most
  structural tokens (braces, colons, quotes, the fixed parts of key
  names) have exactly one legal continuation. Calling the model for
  those would be correct but pointlessly slow, so the generator writes
  them directly.
- **Pydantic for every data contract.** `functions_definition.json`,
  `function_calling_tests.json`, and the output file are each modeled
  as pydantic classes (`src/models.py`), so malformed input is caught
  as a validation error at the boundary rather than surfacing later as
  a confusing runtime failure deep inside generation.

## Performance Analysis

Constrained decoding guarantees 100% valid, schema-compliant JSON by
construction — there is no retry loop, and no output can fail to
parse, because invalid tokens are never reachable in the first place.

On the bundled sample data (11 prompts, 5 functions), a full run
completes in under a minute on standard hardware and produces a
correct function name and argument set for every prompt, including
prompts that require inferring a non-trivial value (e.g. a regex
pattern) rather than copying a literal from the prompt text.

The main performance lever is the "skip the model when only one token
is legal" optimization described above: a meaningful fraction of the
~20-30 tokens per output (JSON punctuation, fixed key names) costs no
model inference at all.

## Challenges Faced

- **A hung generation loop.** An early version of the string grammar
  allowed any legal-in-a-JSON-string character indefinitely, without
  ever telling the model a closing `"` was expected — since a comma or
  `}` is ordinary text *inside* a string, there was nothing to signal
  "stop here." The fix was to wrap every string-typed value in its own
  literal `"` on each side, the same way a human writing JSON would.
- **Ambiguous function-name prefixes.** When one function name is a
  prefix of another (e.g. `fn_add` vs. `fn_add_numbers`), the grammar
  has to keep both "stop here" and "keep going" legal until the model's
  own next token resolves the ambiguity, rather than guessing early.
- **Premature performance optimization.** The vocabulary-to-text
  lookup (`build_id_to_text`) was suspected of being a bottleneck and
  rewritten to avoid calling the SDK's `decode()` per token id — before
  actually measuring it. Once measured, decoding the entire ~150,000
  entry vocabulary this way took well under half a second, so the
  rewrite was reverted in favor of the simpler implementation.

## Testing Strategy

During development, the grammar primitives, vocabulary trie, and
generator were covered by unit tests using fabricated vocabularies and
mocked model responses (no model download required), alongside a
small integration suite that exercises the full pipeline against the
real `Qwen/Qwen3-0.6B` model end-to-end. Per the assignment
guidelines, test code is not included in this submission.

Correctness of the full pipeline can be verified by running
`make run` against the sample files in `data/input/` and inspecting
`data/output/function_calling_results.json` — every entry must be
valid JSON with exactly the `prompt`, `name`, and `parameters` keys,
matching the schema in `functions_definition.json`.

## Example Usage

```bash
uv sync
uv run python -m src
cat data/output/function_calling_results.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Hugging Face — How do Transformers work? (tokenization, logits, generation)](https://huggingface.co/learn/nlp-course)
- [Guidance / Outlines papers and blog posts on constrained/structured generation](https://github.com/dottxt-ai/outlines)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Trie data structure — CLRS / standard algorithms references](https://en.wikipedia.org/wiki/Trie)

**AI usage:** Claude (Anthropic's Claude Code) was used throughout
this project as a pair-programming assistant: designing the grammar
state machine and constrained-decoding generator, implementing the
vocabulary trie, writing and debugging the pydantic models and I/O
error handling, drafting and iterating on unit/integration tests, and
diagnosing runtime issues (including the hung-generation bug described
above under Challenges Faced). All generated code was reviewed, run,
and understood before being kept.

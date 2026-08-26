# How constrained decoding actually works in this project

`docs/state_machine_plan.md` is the technical design doc (useful as a
reference while working on the code). This one is the plain-language
version: what the finished code actually does, walked through with real
examples captured from an actual run against the real model.

## The problem in one sentence

A 0.6B-parameter model, just asked nicely to "output JSON," gets it wrong
a lot of the time — a stray comma, a missing brace, a made-up field. We
don't fix that by checking its answer afterwards and retrying. We fix it
by never letting it produce a wrong character in the first place.

## The trick: a strict multiple-choice form, not an open essay

Normally, generating text with an LLM works one **token** at a time (a
token is a chunk of text — sometimes a whole word, sometimes a few
letters, sometimes just one character). At every step, the model scores
every entry in its ~150,000-word vocabulary and the highest-scoring one
gets picked, then the process repeats for the next token.

Constrained decoding intervenes in that scoring step. Before the model's
scores are used to pick a token, we work out — from the JSON we still
need to produce — which vocabulary entries are even *possible* right now.
Every other entry gets its score set to "never, under any circumstances"
(`-infinity`), so it mathematically cannot be picked. The model still gets
to use its judgement — but only among the choices that are already
guaranteed not to break the JSON.

That's the whole idea. Everything else in this project is machinery to
answer one question, fast, at every single step: **"given what's been
written so far, which vocabulary entries are legal right now?"**

## What the model actually has to write

The output file needs three fields per prompt — `prompt`, `name`,
`parameters` — but `prompt` is just the question we were asked, copied
verbatim. There's no reason to make the model retype it (and every
reason not to: retyping is exactly the kind of free-form text generation
we're trying to avoid). So the model's entire job, for every prompt, is
to fill in:

```json
{"name": "...", "parameters": {...}}
```

`state_machine.py` glues the original `prompt` back on afterwards.

## A real, complete example

Prompt: **"What is the sum of 2 and 3?"**
Available functions (abbreviated): `fn_add_numbers(a: number, b: number)`,
`fn_greet(name: string)`.

Here is the *actual* sequence of tokens the model produced, captured by
instrumenting the real generation loop against the real model. Nothing
here is invented — this is one real run:

| # | token written | how it was decided |
|---|---|---|
| 1 | `{` | model chose between 2 legal options |
| 2 | `"` | model chose between 2 legal options |
| 3 | `name` | model chose between 4 legal options |
| 4 | `":` | model chose between 2 legal options |
| 5 | ` "` | model chose between 2 legal options |
| 6 | `f` | model chose between 2 legal options |
| 7 | `n` | **forced** — only 1 legal option existed |
| 8 | `_add` | model chose between 6 legal options |
| 9 | `_numbers` | model chose between 5 legal options |
| 10 | `",` | model chose between 2 legal options |
| 11 | ` "` | model chose between 2 legal options |
| 12 | `parameters` | model chose between 7 legal options |
| 13 | `":` | model chose between 2 legal options |
| 14 | ` {` | model chose between 2 legal options |

At step 9, the moment the model finishes writing `fn_add_numbers`, the
system now knows *exactly* which function was picked — and therefore
exactly which parameters (`a: number`, `b: number`) must come next. It
builds a brand-new, tiny "form" for just those two fields and continues:

| # | token written | how it was decided |
|---|---|---|
| 1 | `"a` | model chose between 2 legal options |
| 2 | `":` | model chose between 2 legal options |
| 3 | ` ` | model chose between 2 legal options |
| 4 | `2` | model chose between 11 legal options (any digit) |
| 5 | `,` | model chose between 12 legal options |
| 6 | ` "` | model chose between 2 legal options |
| 7 | `b` | **forced** — only 1 legal option existed |
| 8 | `":` | model chose between 2 legal options |
| 9 | ` ` | model chose between 2 legal options |
| 10 | `3` | model chose between 11 legal options |
| 11 | `}}` | model chose between 13 legal options |

Final text, concatenated: `{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`
— valid JSON, matching the schema, on the first and only attempt. No
retry loop, no "try to parse it and hope."

Notice two things:
- **Steps aren't characters.** `_add`, `_numbers`, `parameters`, `":` each
  came out as a single token. The model's vocabulary is built from
  common chunks of text, not single letters, so most of the "obviously
  forced" scaffolding (braces, colons, quotes) still costs a real step,
  but a cheap one.
- **"Forced" steps skip the model entirely.** At step 7 of each phase,
  there was only one legal continuation (`n` to keep spelling `fn_...`;
  `b` because `a` was already used), so the code doesn't even bother
  asking the model — it just writes it in. This is a real speed
  optimization, not just a simplification: most of a JSON object's
  punctuation and key names have exactly one legal continuation, so a
  large fraction of the ~20–30 tokens per output cost nothing at all.

## The four building blocks

Everything the grammar needs to express is built from four small,
reusable pieces (`src/constrained_decoding/grammar.py`):

- **`Literal("text")`** — "this exact text and nothing else." Used for
  all the fixed JSON punctuation: `{`, `", "parameters": {`, the quotes
  around a key name, `}}`, etc.
- **`Enum([option1, option2, ...])`** — "exactly one of these fixed
  choices." Used for the function name itself (options = every function
  name in `functions_definition.json`) and for `true`/`false`.
- **`JSONString()`** — "any text, as long as it's legal *inside* a JSON
  string" (no raw newlines, an unescaped `"` isn't allowed, etc.). Used
  for `string`-typed parameter values.
- **`JSONNumber()`** — "digits, optionally with a leading `-` and a
  decimal point, but nothing else." Used for `number`-typed parameter
  values.

These snap together into a sequence, and the sequence itself behaves like
one of these pieces (it can be asked "what's legal next?" too) — that's
what lets the whole `{"name": ..., "parameters": {...}}` shape be built
by just chaining these four pieces in the right order.

### The tricky bit: knowing when to stop

`Enum` has to handle an ambiguous case correctly: what if one function
name is the start of another, e.g. `fn_add` and `fn_add_numbers`? After
the model has written `fn_add`, is that the complete answer, or the start
of the longer name? Both must stay possible — the model needs to be free
to close the name right there (write the closing `"`) *or* keep going
(write `_numbers`). The grammar always keeps both options open until the
model's own token choice resolves it, so this never produces something
invalid or forces a premature guess.

## The two phases

1. **Phase A — which function?** The grammar only knows `{"name": "` +
   *one of the function names* + `", "parameters": {`. Once the function
   name is fully written, we look it up in `functions_definition.json` —
   we now know its exact parameter list and types.
2. **Phase B — which arguments?** *Built fresh, using the answer from
   Phase A.* For `fn_add_numbers(a: number, b: number)` that's
   `"a": <number>, "b": <number>}}`; for `fn_greet(name: string)` it
   would instead be `"name": <string>}}`. There is never a point where an
   unexpected key, a missing key, or a wrong type is reachable — the
   shape is fixed by construction the moment Phase A resolves.

## Turning "what's legal" into "which vocabulary entries"

Knowing that "only a digit is legal here" is one thing; knowing *which of
the ~150,000 vocabulary entries* are digits (or start with a digit, since
tokens can be multiple characters like `"23"`) is another. Scanning all
150,000 on every single token would be slow. Instead, `vocab_trie.py`
organizes the whole vocabulary once, up front, into a **trie** — a
tree where you spell out a token letter by letter and each fork is a
branch to the next possible letter, similar to how a phone's predictive
keyboard narrows suggestions as you type.

Finding the legal tokens for the current grammar state means walking that
tree and only going down branches whose next letter is currently allowed
— so for something narrow like "the next character must be `f`," almost
none of the tree gets visited at all.

## A real bug, found by actually running it

The plan looked right on paper. Running it against the real model on the
real sample data turned up something the design review hadn't: the very
first working version generated `"name": shrek` instead of
`"name": "shrek"`. The grammar for "any legal string content" was never
told a closing `"` was coming — and since a comma or a `}` is perfectly
ordinary text *inside* a string, without that closing quote in place
there was nothing to ever tell the model "you're done." Generation just
ran on indefinitely for any prompt needing a string-typed argument (the
run hung for over five minutes and had to be killed). The fix was one
line: wrap every string-typed value in its own `Literal('"')` on each
side, exactly like a human writing JSON would.

(Building the vocabulary lookup table — turning each of the ~150,000
vocabulary ids into its real text via the model's own `decode()` — was
briefly suspected as a second source of slowness and rewritten to avoid
it, before actually measuring it and finding `decode()` was never the
problem: decoding the entire vocabulary this way takes well under half a
second. That rewrite was reverted; `build_id_to_text()` just calls
`decode()` once per id, which is simple and already fast enough. The
lesson stuck: measure before rewriting for performance, even when a fix
seems to work.)

## Does it actually work? (real output, real run)

`uv run python -m src` on the 11 sample prompts and 5 sample functions
finished in **53 seconds** and got all 11 right — including the ones that
require inferring an actual regex pattern from a description:

```json
{
  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "([0-9]+)",
    "replacement": "NUMBERS"
  }
}
```

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "a|e|i|o|u",
    "replacement": "*"
  }
}
```

## Where everything lives

| File | What it's responsible for |
|---|---|
| `src/constrained_decoding/grammar.py` | The four building blocks (`Literal`, `Enum`, `JSONString`, `JSONNumber`) and how they chain into a sequence. Pure Python — no model involved, easy to test on its own. |
| `src/constrained_decoding/vocab_trie.py` | Turns the model's vocabulary file into a fast lookup tree, and answers "which token ids are legal right now?" |
| `src/constrained_decoding/generator.py` | The actual token-by-token loop: ask what's legal → skip the model if there's only one option, otherwise ask the model to score the legal options → write down the winner → repeat until done. |
| `src/state_machine.py` | Ties it together per prompt: build the model's context, run Phase A to pick the function, build Phase B from that function's real parameter list, run it, parse the result. |
| `tests/constrained_decoding/` | Tests for all of the above using fake/fabricated data — no model download needed, so they run in under two seconds. |

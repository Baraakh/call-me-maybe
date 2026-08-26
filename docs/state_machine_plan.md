# `state_machine.py` — Constrained Decoding Design Plan

This document plans the finite state machine (FSM) that drives constrained
decoding, the core technical requirement of the project (spec §III.3, §V.3.3).
No code yet — this is the architecture to implement against.

## 1. What must actually be generated

The output contract (`FunctionCallResult`) needs three keys: `prompt`,
`name`, `parameters`. We already have `prompt` verbatim from the input file —
there is no reason to make the LLM regenerate it. So the LLM's job is
reduced to producing exactly:

```json
{"name": "<one of the function names>", "parameters": {<schema-typed args>}}
```

`src/state_machine.py` fills in `prompt` itself once the LLM finishes. This
keeps the grammar small and removes an entire class of failure (the model
mangling the echoed prompt string).

Two things must be schema-driven, per spec §V.3.3:
- **Function selection** (`name`) must be chosen by the LLM, constrained to
  the exact set of names in `functions_definition.json` — not picked by
  heuristics.
- **Argument generation** (`parameters`) must be constrained to the chosen
  function's parameter names and JSON types (`string | number | boolean`,
  per `TypeSpec`).

## 2. Two-layer architecture

Constrained decoding needs two independent concerns that should not be
tangled together:

1. **Grammar layer** — a character/text-level automaton that knows nothing
   about tokens. Given "what has been generated so far," it can answer
   "which characters are legal next, and does the string so far already
   form a complete, exit-able value?"
2. **Token-masking layer** — bridges the grammar to the LLM's actual
   vocabulary. Given the grammar's current state, it answers "which token
   ids are legal next?" so the corresponding logits can be kept and
   everything else set to `-inf` before argmax/sampling.

Splitting these means the grammar is pure Python with no LLM dependency
(easy to unit test), and the token-masking layer is a thin, reusable
adapter (easy to test against a fake vocabulary).

```
src/state_machine.py            orchestration: per-prompt generation loop
src/constrained_decoding/
    grammar.py                  character-level FSM primitives + composition
    vocab_trie.py                token-id <-> text table + trie for masking
    generator.py                 the actual token-by-token decode loop
```

(Names are suggestions; finalize during implementation.)

## 3. Grammar primitives

Because every parameter type in this project is a flat scalar
(`string | number | boolean` — no arrays/objects per `TypeSpec`), the whole
grammar can be built by composing a handful of reusable primitives in
sequence. Each primitive exposes the same interface: given the text
generated so far *within that primitive's span*, report (a) the set of
legal next characters (or "any character except quote", etc.), and (b)
whether the current text is already a valid, complete value that may be
exited to the next primitive.

- **`Literal(text)`** — must match exactly, one character at a time
  (`{`, `}`, `:`, `,`, `"name"`, `"parameters"`, the quote marks around
  a parameter's key, etc.). No branching.
- **`Enum(options)`** — trie of candidate strings; narrows as characters
  are matched; exits (closing quote allowed) only when the accumulated
  text exactly equals one full candidate. Used for:
  - the function `name` value (`options` = every `FunctionDefinition.name`)
  - each parameter's key literal (`options` = that one key, effectively a
    `Literal`, but implemented via the same code path)
  - JSON booleans (`Enum(["true", "false"])`)
- **`JSONString()`** — arbitrary string content per JSON escaping rules
  (reject raw control characters and an unescaped `"`; accept `\"`, `\\`,
  `\n`, etc.); exits on an unescaped closing quote.
- **`JSONNumber()`** — `-?[0-9]+(\.[0-9]+)?` (extend to exponents only if
  needed later); exits as soon as the digits-so-far form a complete
  number, but stays open for more digits — this is a "greedy but
  exitable" primitive, same pattern as string enums.

A `Sequence([...])` combinator chains primitives: the FSM is in exactly one
primitive at a time; when that primitive signals "exited," control moves to
the next one in the sequence.

## 4. Two-phase composition per prompt

The parameter grammar cannot be built until we know which function was
picked, so generation happens in two phases against one continuously
extended token context:

**Phase A — structural preamble + name selection**
```
Sequence([
    Literal('{"name": "'),
    Enum(all function names),
    Literal('", "parameters": {'),
])
```
Once the `Enum` primitive exits, the matched candidate *is* the selected
`FunctionDefinition` — no separate lookup/parsing step needed, the grammar
state transition doubles as function selection.

**Phase B — parameters object, built dynamically**

Given the resolved `FunctionDefinition.parameters` (a `dict[str, TypeSpec]`,
order-preserving), build:
```
Sequence([
    Literal('"' + key1 + '": '), value_primitive(type1), Literal(', '),
    Literal('"' + key2 + '": '), value_primitive(type2), Literal(', '),
    ...
    Literal('"' + keyN + '": '), value_primitive(typeN),
    Literal('}}'),
])
```
where `value_primitive(type)` is `JSONString()`, `JSONNumber()`, or
`Enum(["true","false"])` depending on the parameter's `TypeSpec.type`.

If a function has zero parameters, Phase B collapses to `Literal('}}')`.

This two-phase design means the *entire* output shape is known and fixed by
construction the moment the name is resolved — there is never a point where
an invalid key, wrong type, or extra key is reachable.

## 5. Token masking: vocab trie walk

Naively, masking would mean iterating every vocabulary entry (~150k for
Qwen3) at every generation step and testing each candidate string against
the grammar. That's wasteful. Instead:

1. **Build a vocab trie once**, at `StateMachine.__init__`, mapping decoded
   token text → token id, using `llm_client.get_path_to_vocab_file()` (per
   the spec's explicit hint in §V.3.3). Loading this file gives raw
   BPE-encoded token strings (byte-level tokens use marker characters like
   `Ġ` for a leading space); convert each to real text via the standard
   GPT-2/BPE byte-to-unicode inverse table before inserting into the trie.
   (Fallback/cross-check option: call `llm_client.decode([token_id])` per
   token instead of reimplementing the byte table — simpler, slightly
   slower, but only paid once at startup. Decide during implementation
   which to use first; the from-scratch byte-table route is also what
   unlocks the bonus "recode the tokenizer" item, see §9.)
2. **Per generation step**, instead of scanning the whole vocabulary, walk
   the vocab trie and the grammar's character-acceptance function in
   lockstep starting from the grammar's current state: descend a trie edge
   only if that character is currently legal; whenever the trie walk lands
   on a node that is a complete token, record that token id as valid
   (continuing further if the grammar isn't exited, or also if the grammar
   would accept exiting there). This visits only the region of the
   vocabulary that's actually reachable under the current constraint
   (tiny for `Enum`/`Literal` states, larger but still bounded for
   `JSONString`/`JSONNumber`), instead of the full vocabulary every time.
3. **Apply the mask**: call `get_logits_from_input_ids`, set every logit not
   in the valid-token set to `-inf`, pick a token (argmax, greedy — see
   §7 for why sampling isn't needed here), append it to the input ids and
   to the grammar's accumulated text, advance the grammar state.

**Optimization worth calling out explicitly**: whenever the valid-token set
computed by the trie walk has exactly one member (true for every `Literal`
span, and for large stretches of `Enum` once only one candidate remains),
skip the model call entirely — there's nothing to choose, so append that
token directly. This cuts the number of forward passes substantially since
most of the JSON scaffolding (`{"name": "`, `", "parameters": {`, key
literals, `, `, `}}`) is fixed text with no real decision, and only the
function-name choice and each parameter value are actual LLM decisions.
This directly serves the "under 5 minutes for all prompts" requirement
(§V.5).

## 6. Generation loop (per prompt)

```
1. Build the LLM context: system/instruction text (briefly explaining the
   task + listing available functions with name/description/params) +
   the user's natural-language prompt, encoded via llm_client.encode().
2. Append the fixed opening literal '{"name": "' directly (no model call
   needed — see §5's optimization).
3. Run the trie-walk/mask/select loop through Phase A until the Enum
   exits → resolves `chosen_function: FunctionDefinition`.
4. Build the Phase B sequence from chosen_function.parameters.
5. Continue the same loop through Phase B until the final Literal('}}')
   is fully emitted.
6. The accumulated text since step 2 is now, by construction, exactly
   valid JSON matching the schema — parse it directly (no try/except
   needed for malformed JSON, since it's structurally guaranteed) and
   attach the original `prompt` to build a `FunctionCallResult`.
```

`get_func_calls_batch` (already stubbed in `state_machine.py`) runs this
loop once per `PromptEntry` and collects the results. "Batch" here can
start out as a plain Python loop over prompts (simplest, matches the
current SDK surface which has no native batching); revisit only if the
5-minute budget is at risk.

## 7. Why greedy, not sampling

Constrained decoding already guarantees validity; greedy argmax over the
masked logits gives deterministic, reproducible output and is the simplest
thing that can hit the "90%+ correct function selection" bar (§V.5) — the
mask removes the *invalid* options, greedy picks the model's *best guess*
among what's left. No temperature/top-p machinery needed. (If accuracy
testing later shows greedy gets stuck picking a wrong-but-plausible
function name, revisit — but that's a tuning question, not a structural
one.)

## 8. Error handling

Because the grammar makes every state's valid-token set provably
non-empty (there is always at least one legal continuation — the earliest
point a `JSONString`/`JSONNumber` can exit is well-defined, and `Enum`
always has ≥1 remaining candidate by construction), the decode loop itself
should never hit a dead end. Two things still need explicit handling,
consistent with the existing `InputFileError`/`OutputFileError` pattern in
`src/io_handling/`:

- **Defensive assertion**: if the computed valid-token set is ever empty
  (would indicate a bug in the grammar/trie code, not bad input), fail
  loudly with a clear internal error rather than silently producing
  garbage — this is a program-correctness bug, not a runtime condition to
  recover from.
- **`functions_definition.json` has zero functions**: nothing for `Enum`
  to select from. Detect this before starting generation and raise a
  clear, project-specific error (following the existing error-handling
  convention) rather than letting the trie walk fail deep inside the loop.

No other error handling is needed inside the loop — the entire point of
this design is that "malformed output" becomes structurally unreachable.

## 9. Testing strategy

- **Grammar unit tests** (no LLM, no model download): feed hand-written
  character sequences into `Literal`/`Enum`/`JSONString`/`JSONNumber` and
  assert accept/reject and exit-ability at each step. Fast, deterministic,
  the bulk of the test suite.
- **Vocab trie tests**: build a trie from a small fake vocab dict (not the
  real 150k-entry file) and verify the lockstep walk produces the expected
  valid-token-id set for known grammar states.
- **Generator loop tests**: stub `Small_LLM_Model.get_logits_from_input_ids`
  with a fake that returns deterministic logits, so the full
  mask → argmax → append loop can be exercised end-to-end without loading
  the real Qwen model — keeps this fast enough for routine `pytest` runs.
- **One real-model integration test** (slow, likely `@pytest.mark.slow` or
  similar, not part of the default fast suite): run an actual prompt
  through `StateMachine` against the real `functions_definition.json`
  example and assert the output parses and matches the schema. This is the
  only test that touches the real model/download.

## 10. Suggested build order

1. `grammar.py` primitives + `Sequence` combinator, fully unit-tested with
   fake character streams — no LLM involved yet.
2. `vocab_trie.py`: load+decode the vocab file, build the trie, implement
   the lockstep walk against a grammar state — unit-tested with a small
   fake vocab first, then sanity-checked against the real vocab file.
3. `generator.py`: the mask/select/append loop, tested against a stubbed
   LLM client first.
4. Wire Phase A (name selection) end-to-end against the real model on one
   prompt; confirm the `Enum` resolves to a real `FunctionDefinition`.
5. Wire Phase B (dynamic parameter grammar) end-to-end; confirm full valid
   JSON is produced and parses.
6. Fill in `StateMachine.get_func_calls_batch` to run the full loop over
   all prompts and build `FunctionCallResult` list, then confirm
   `write_results` (already implemented) produces a spec-compliant output
   file.
7. Prompt-template tuning pass (the instruction/system text shown to the
   model before generation starts) to hit the 90%+ correctness bar —
   this is the only part likely to need iteration once the mechanism
   itself is proven correct.

## 11. Open questions to confirm before/while implementing

- **Byte-table vs. `decode()`-per-token** for building the vocab trie
  (§5, point 1) — start with whichever is faster to get correct; the
  byte-table route also happens to satisfy the bonus item "recoding the
  tokenizer... using `get_logits_from_input_ids` and
  `get_path_to_vocab_file`" (spec §VII) if `encode`/`decode` are avoided
  entirely later.
- **Prompt/system-text wording** shown to the model to describe the
  available functions — affects accuracy, not structural validity; tune
  empirically once the mechanism works.
- **Numeric type fidelity**: `TypeSpec` only says `"number"`, not
  int-vs-float. `JSONNumber()` as specified (`-?[0-9]+(\.[0-9]+)?`)
  naturally produces `int` when there's no decimal point and `float`
  otherwise (via `json.loads`), which matches `ParamValue = str | int |
  float | bool` in `models.py` without extra work — confirm this is the
  desired behavior rather than always coercing to `float`.

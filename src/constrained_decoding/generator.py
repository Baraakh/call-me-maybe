"""The actual token-by-token constrained decoding loop.

At every step, the current grammar state is turned into the set of
currently-valid token ids via the vocabulary trie. When exactly one token
is valid, it is appended directly with no model call — most of a JSON
object's structural scaffolding (punctuation, key literals) has only one
legal continuation, so this is a meaningful part of what keeps generation
inside the "reasonable speed" budget (docs/call_me_maybe.md §V.5). Only
real decisions (which function, what value) cost a forward pass, and even
then the choice is a plain argmax over the masked logits: validity is
already guaranteed by construction, so there is nothing sampling would buy
here that greedy selection doesn't already give more deterministically.
"""

from collections.abc import Hashable

from llm_sdk import Small_LLM_Model

from .grammar import GrammarState, Sequence
from .vocab_trie import VocabTrie


class ConstrainedDecodingError(Exception):
    """Raised when the grammar reaches a state with no valid next token.

    This should never happen for a well-formed grammar — every primitive
    always has at least one legal continuation — so this indicates a bug
    in the grammar construction, not a runtime/input condition.
    """


def generate_constrained(
    llm_client: Small_LLM_Model,
    vocab_trie: VocabTrie,
    context_ids: list[int],
    primitives: list[GrammarState],
) -> tuple[list[int], str]:
    """Extend ``context_ids`` with tokens satisfying ``primitives`` in
    order, stopping as soon as the last primitive is complete.

    Args:
        llm_client: The model wrapper, used only for its public
            ``get_logits_from_input_ids`` method.
        vocab_trie: Vocabulary trie built once per ``StateMachine``.
        context_ids: Token ids generated so far; not mutated in place.
        primitives: The grammar, as a sequence of primitives to satisfy.

    Returns:
        A tuple of the extended token id list and the text generated
        during this call (not including ``context_ids``' prior text).
    """
    context_ids = list(context_ids)
    state: GrammarState = Sequence(tuple(primitives))
    cache: dict[Hashable, dict[int, GrammarState]] = {}
    text_parts: list[str] = []

    while not state.can_exit():
        valid = vocab_trie.valid_next_tokens(state, cache)
        if not valid:
            raise ConstrainedDecodingError(
                "constrained decoding reached a dead end: no vocabulary "
                "token is a valid continuation of the current grammar "
                "state; this indicates a bug in the grammar definition"
            )
        if len(valid) == 1:
            token_id, next_state = next(iter(valid.items()))
        else:
            logits = llm_client.get_logits_from_input_ids(context_ids)
            token_id = max(valid, key=lambda tid: logits[tid])
            next_state = valid[token_id]

        context_ids.append(token_id)
        text_parts.append(vocab_trie.text_of(token_id))
        state = next_state

    return context_ids, "".join(text_parts)

"""Maps the LLM's vocabulary to text and lets a grammar state be turned
into the set of currently-valid token ids.

Building the mapping uses only public ``Small_LLM_Model`` methods, per the
project's constraint against reaching into ``llm_sdk`` internals:
``get_path_to_vocab_file`` (as the spec explicitly hints, see
docs/call_me_maybe.md §V.3.3) gives the exact set of ordinary vocabulary
token ids, and ``decode`` turns each one into its real text. Special/added
tokens (end-of-text, chat-template markers, ...) are not present in the
vocab file and are therefore never candidates for generation, which is the
behaviour we want: the grammar defines its own stopping point, not the
model's EOS token.

Measured directly: decoding every one of the ~150k ordinary vocabulary
ids this way takes well under half a second (no model forward pass is
involved, just the tokenizer) — negligible next to the couple of seconds
it takes to load the model itself, so there is no need to reimplement the
tokenizer's own byte-level decoding to speed this up.
"""

from __future__ import annotations

import json
from collections.abc import Hashable

from llm_sdk import Small_LLM_Model

from .grammar import GrammarState


def load_vocab_ids(llm_client: Small_LLM_Model) -> list[int]:
    """Return every ordinary vocabulary token id, from the vocab file."""
    vocab_path = llm_client.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        token_to_id: dict[str, int] = json.load(vocab_file)
    return list(set(token_to_id.values()))


def build_id_to_text(llm_client: Small_LLM_Model) -> dict[int, str]:
    """Decode every ordinary vocabulary token id to its text."""
    return {
        token_id: llm_client.decode([token_id])
        for token_id in load_vocab_ids(llm_client)
    }


class _VocabTrieNode:
    __slots__ = ("children", "token_ids")

    def __init__(self) -> None:
        self.children: dict[str, _VocabTrieNode] = {}
        self.token_ids: list[int] = []


class VocabTrie:
    """A trie over every vocabulary token's decoded text, used to find
    which token ids are legal continuations of a given grammar state
    without scanning the whole vocabulary at every generation step."""

    def __init__(self, id_to_text: dict[int, str]) -> None:
        self._id_to_text = id_to_text
        self._root = _VocabTrieNode()
        for token_id, text in id_to_text.items():
            if not text:
                continue
            node = self._root
            for char in text:
                node = node.children.setdefault(char, _VocabTrieNode())
            node.token_ids.append(token_id)

    def text_of(self, token_id: int) -> str:
        return self._id_to_text[token_id]

    def valid_next_tokens(
        self,
        state: GrammarState,
        cache: dict[Hashable, dict[int, GrammarState]] | None = None,
    ) -> dict[int, GrammarState]:
        """Every token id that is a legal next token from ``state``,
        mapped to the grammar state reached after consuming it.

        ``cache`` (scoped to a single generation call by the caller) lets
        repeated occurrences of the same grammar state — very common for
        ``JSONString``/``JSONNumber``, whose internal state does not grow
        with how much has already been generated — skip re-walking the
        trie entirely.
        """
        key = state.state_key()
        if cache is not None and key in cache:
            return cache[key]
        result: dict[int, GrammarState] = {}
        self._walk(self._root, state, result)
        if cache is not None:
            cache[key] = result
        return result

    def _walk(
        self,
        node: _VocabTrieNode,
        state: GrammarState,
        result: dict[int, GrammarState],
    ) -> None:
        for char, child in node.children.items():
            next_state = state.step(char)
            if next_state is None:
                continue
            if child.token_ids:
                for token_id in child.token_ids:
                    result[token_id] = next_state
            self._walk(child, next_state, result)

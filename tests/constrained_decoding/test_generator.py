"""Unit tests for the constrained decoding generation loop, against a
fake LLM client so no real model download is required."""

from typing import cast

import pytest
from llm_sdk import Small_LLM_Model

from src.constrained_decoding.generator import (
    ConstrainedDecodingError,
    generate_constrained,
)
from src.constrained_decoding.grammar import Enum, GrammarState, Literal
from src.constrained_decoding.vocab_trie import VocabTrie


class _FakeLLM:
    """Always scores the token with the highest id highest, so tests can
    predict exactly which of several valid tokens gets picked."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self.calls = 0

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[float]:
        self.calls += 1
        return [float(i) for i in range(self._vocab_size)]


def _fake_llm(vocab_size: int) -> tuple[Small_LLM_Model, _FakeLLM]:
    fake = _FakeLLM(vocab_size)
    return cast(Small_LLM_Model, fake), fake


class TestGenerateConstrained:
    def test_single_valid_token_needs_no_model_call(self) -> None:
        llm, fake = _fake_llm(vocab_size=1)
        trie = VocabTrie({0: "hi"})
        primitives: list[GrammarState] = [Literal("hi")]

        ids, text = generate_constrained(llm, trie, [], primitives)

        assert text == "hi"
        assert ids == [0]
        assert fake.calls == 0

    def test_picks_highest_scoring_token_among_valid_options(self) -> None:
        llm, fake = _fake_llm(vocab_size=3)
        trie = VocabTrie({0: "fn_add", 1: "fn_sub", 2: "irrelevant"})
        primitives: list[GrammarState] = [
            Enum.from_options(["fn_add", "fn_sub"])
        ]

        ids, text = generate_constrained(llm, trie, [], primitives)

        # token id 1 ("fn_sub") scores higher than id 0 ("fn_add"), and
        # the irrelevant token (id 2) is masked out entirely.
        assert text == "fn_sub"
        assert ids == [1]
        assert fake.calls == 1

    def test_multi_token_generation_extends_context(self) -> None:
        llm, _ = _fake_llm(vocab_size=2)
        trie = VocabTrie({0: "{", 1: "}"})
        primitives: list[GrammarState] = [Literal("{"), Literal("}")]

        ids, text = generate_constrained(llm, trie, [99], primitives)

        assert ids == [99, 0, 1]
        assert text == "{}"

    def test_dead_end_raises(self) -> None:
        llm, _ = _fake_llm(vocab_size=1)
        trie = VocabTrie({0: "z"})
        primitives: list[GrammarState] = [Literal("a")]

        with pytest.raises(ConstrainedDecodingError):
            generate_constrained(llm, trie, [], primitives)

    def test_does_not_mutate_input_context_list(self) -> None:
        llm, _ = _fake_llm(vocab_size=1)
        trie = VocabTrie({0: "x"})
        original = [1, 2, 3]

        ids, _ = generate_constrained(
            llm, trie, original, [Literal("x")]
        )

        assert original == [1, 2, 3]
        assert ids == [1, 2, 3, 0]

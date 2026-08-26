"""Unit tests for the vocabulary trie and token masking."""

import json
from typing import cast

from llm_sdk import Small_LLM_Model

from src.constrained_decoding.grammar import Enum, JSONNumber, Literal
from src.constrained_decoding.vocab_trie import (
    VocabTrie,
    build_id_to_text,
    load_vocab_ids,
)


class TestLoadVocabIds:
    def test_reads_ids_from_vocab_file(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        vocab = {"a": 0, "b": 1, "c": 2}
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

        class _FakePathLLM:
            def get_path_to_vocab_file(self) -> str:
                return str(vocab_path)

        ids = load_vocab_ids(cast(Small_LLM_Model, _FakePathLLM()))
        assert set(ids) == {0, 1, 2}

    def test_deduplicates_ids(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        # two different token spellings sharing an id shouldn't happen in
        # a well-formed vocab file, but is cheap to guard against anyway.
        vocab = {"a": 0, "b": 0, "c": 1}
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

        class _FakePathLLM:
            def get_path_to_vocab_file(self) -> str:
                return str(vocab_path)

        ids = load_vocab_ids(cast(Small_LLM_Model, _FakePathLLM()))
        assert set(ids) == {0, 1}
        assert len(ids) == 2


class TestBuildIdToText:
    def _llm_for_vocab(
        self,
        tmp_path: object,
        vocab: dict[str, int],
        decoded: dict[int, str],
    ) -> Small_LLM_Model:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

        class _FakeLLM:
            def get_path_to_vocab_file(self) -> str:
                return str(vocab_path)

            def decode(self, ids: list[int]) -> str:
                return "".join(decoded[i] for i in ids)

        return cast(Small_LLM_Model, _FakeLLM())

    def test_decodes_every_id(self, tmp_path: object) -> None:
        llm = self._llm_for_vocab(
            tmp_path,
            vocab={"ab": 0, "cd": 1},
            decoded={0: "ab", 1: "cd"},
        )
        assert build_id_to_text(llm) == {0: "ab", 1: "cd"}

    def test_uses_the_tokenizer_decoded_text_not_the_raw_key(
        self, tmp_path: object
    ) -> None:
        # decode() is the source of truth for a token's real text, which
        # can differ from its raw vocab-file spelling (e.g. a
        # byte-level-BPE marker like 'Ġ' decoding to a real space).
        llm = self._llm_for_vocab(
            tmp_path,
            vocab={"Ġhello": 0},
            decoded={0: " hello"},
        )
        assert build_id_to_text(llm) == {0: " hello"}


class TestVocabTrie:
    def _trie(self) -> VocabTrie:
        return VocabTrie({
            0: "fn_add",
            1: "fn_sub",
            2: "_numbers",
            3: '"',
            4: "1",
            5: "23",
            6: ",",
        })

    def test_finds_single_matching_token(self) -> None:
        trie = self._trie()
        state = Literal("fn_add")
        valid = trie.valid_next_tokens(state)
        assert set(valid) == {0}

    def test_finds_all_enum_branches(self) -> None:
        trie = self._trie()
        state = Enum.from_options(["fn_add", "fn_sub"])
        valid = trie.valid_next_tokens(state)
        assert set(valid) == {0, 1}

    def test_enum_prefix_extends_across_multiple_tokens(self) -> None:
        trie = VocabTrie({
            0: "fn_add",
            1: "_numbers",
        })
        state = Enum.from_options(["fn_add", "fn_add_numbers"])
        first = trie.valid_next_tokens(state)
        assert set(first) == {0}
        after_add = first[0]
        second = trie.valid_next_tokens(after_add)
        assert set(second) == {1}

    def test_rejects_tokens_outside_grammar(self) -> None:
        trie = self._trie()
        state = Literal("fn_add")
        valid = trie.valid_next_tokens(state)
        assert 3 not in valid  # the quote token doesn't match "fn_add"

    def test_number_token_variants_all_valid(self) -> None:
        trie = self._trie()
        state = JSONNumber()
        valid = trie.valid_next_tokens(state)
        assert 4 in valid  # "1"
        assert 5 in valid  # "23"
        assert 3 not in valid  # quote isn't a digit

    def test_cache_returns_same_result_object(self) -> None:
        trie = self._trie()
        state = Literal("fn_add")
        cache: dict = {}
        first = trie.valid_next_tokens(state, cache)
        second = trie.valid_next_tokens(state, cache)
        assert first is second

    def test_empty_result_for_dead_end_state(self) -> None:
        trie = VocabTrie({0: "x"})
        state = Literal("y")
        assert trie.valid_next_tokens(state) == {}

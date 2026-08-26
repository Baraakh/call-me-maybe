"""Character-level grammar primitives for constrained decoding.

Each primitive is an immutable, tiny state machine over a *single* piece
of the output grammar (a fixed literal, an enumerated set of strings, a
JSON string body, or a JSON number body). ``step`` never mutates the
receiver: it returns a brand-new state (or ``None`` if the character is
rejected), which is what lets :class:`Sequence` and the vocabulary trie
walk in ``vocab_trie.py`` explore several candidate continuations from the
same starting point without any explicit backtracking/cloning logic.

``Sequence`` composes a fixed list of primitives (built once the shape of
the expected output is known, e.g. once a function has been chosen and its
parameter schema is fixed) and itself satisfies :class:`GrammarState`, so
it can be driven by the same generation loop as a single primitive.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol


class GrammarState(Protocol):
    """Structural interface every grammar primitive (and ``Sequence``)
    implements."""

    def step(self, char: str) -> GrammarState | None:
        """Return the state after consuming ``char``, or ``None`` if
        ``char`` is not a legal continuation from this state."""
        ...

    def can_exit(self) -> bool:
        """Whether this state represents a complete, valid value that may
        be followed by whatever comes next in the grammar."""
        ...

    def state_key(self) -> Hashable:
        """A hashable key identifying this exact state, for memoizing the
        (expensive) vocabulary-trie walk across repeated occurrences of
        the same state."""
        ...


class Literal:
    """Matches one fixed piece of text, character for character."""

    def __init__(self, text: str, index: int = 0) -> None:
        self._text = text
        self._index = index

    def step(self, char: str) -> Literal | None:
        if self._index < len(self._text) and self._text[self._index] == char:
            return Literal(self._text, self._index + 1)
        return None

    def can_exit(self) -> bool:
        return self._index == len(self._text)

    def state_key(self) -> Hashable:
        return (self._text, self._index)


class _EnumTrieNode:
    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[str, _EnumTrieNode] = {}
        self.is_terminal = False


def _build_enum_trie(options: list[str]) -> _EnumTrieNode:
    root = _EnumTrieNode()
    for option in options:
        node = root
        for char in option:
            node = node.children.setdefault(char, _EnumTrieNode())
        node.is_terminal = True
    return root


class Enum:
    """Matches exactly one of a fixed set of candidate strings.

    Used both for real enumerations (the function name, JSON booleans)
    and for literal object keys, which are just a one-option enum.
    """

    def __init__(self, node: _EnumTrieNode) -> None:
        self._node = node

    @classmethod
    def from_options(cls, options: list[str]) -> Enum:
        return cls(_build_enum_trie(options))

    def step(self, char: str) -> Enum | None:
        child = self._node.children.get(char)
        if child is None:
            return None
        return Enum(child)

    def can_exit(self) -> bool:
        return self._node.is_terminal

    def state_key(self) -> Hashable:
        return id(self._node)


class JSONString:
    """Matches the *content* of a JSON string (the surrounding quotes are
    ordinary :class:`Literal` primitives in the composed sequence)."""

    _NORMAL = 0
    _ESCAPE = 1
    _UNICODE1 = 2
    _UNICODE2 = 3
    _UNICODE3 = 4
    _UNICODE4 = 5

    _SIMPLE_ESCAPES = frozenset('"\\/bfnrt')

    def __init__(self, mode: int = _NORMAL) -> None:
        self._mode = mode

    def step(self, char: str) -> JSONString | None:
        if self._mode == self._NORMAL:
            if char == "\\":
                return JSONString(self._ESCAPE)
            if char == '"' or ord(char) < 0x20:
                return None
            return JSONString(self._NORMAL)
        if self._mode == self._ESCAPE:
            if char in self._SIMPLE_ESCAPES:
                return JSONString(self._NORMAL)
            if char == "u":
                return JSONString(self._UNICODE1)
            return None
        if self._mode in (
            self._UNICODE1,
            self._UNICODE2,
            self._UNICODE3,
            self._UNICODE4,
        ):
            if char not in "0123456789abcdefABCDEF":
                return None
            if self._mode == self._UNICODE4:
                return JSONString(self._NORMAL)
            return JSONString(self._mode + 1)
        return None

    def can_exit(self) -> bool:
        return self._mode == self._NORMAL

    def state_key(self) -> Hashable:
        return ("JSONString", self._mode)


class JSONNumber:
    """Matches ``-?[0-9]+(\\.[0-9]+)?``, or just ``-?[0-9]+`` (no decimal
    point ever allowed) when ``integer_only`` is set — used for the
    ``integer`` parameter type, as opposed to ``number`` which allows
    either an integer or a float."""

    _START = 0
    _NEG = 1
    _INT = 2
    _DOT = 3
    _FRAC = 4

    def __init__(self, mode: int = _START, integer_only: bool = False) -> None:
        self._mode = mode
        self._integer_only = integer_only

    def step(self, char: str) -> JSONNumber | None:
        is_digit = char.isdigit() and char.isascii()
        if self._mode == self._START:
            if char == "-":
                return JSONNumber(self._NEG, self._integer_only)
            if is_digit:
                return JSONNumber(self._INT, self._integer_only)
            return None
        if self._mode == self._NEG:
            if is_digit:
                return JSONNumber(self._INT, self._integer_only)
            return None
        if self._mode == self._INT:
            if is_digit:
                return JSONNumber(self._INT, self._integer_only)
            if char == "." and not self._integer_only:
                return JSONNumber(self._DOT, self._integer_only)
            return None
        if self._mode == self._DOT:
            if is_digit:
                return JSONNumber(self._FRAC, self._integer_only)
            return None
        if self._mode == self._FRAC:
            if is_digit:
                return JSONNumber(self._FRAC, self._integer_only)
            return None
        return None

    def can_exit(self) -> bool:
        return self._mode in (self._INT, self._FRAC)

    def state_key(self) -> Hashable:
        return ("JSONNumber", self._mode, self._integer_only)


class Sequence:
    """Chains a fixed list of primitives, advancing through them in
    order.

    A character is first offered to the current primitive; if rejected
    but the current primitive is in a state where it may legally end,
    the character is instead offered to the next primitive (and so on).
    This lookahead is what lets ambiguous boundaries — e.g. one function
    name being a prefix of another — resolve correctly: the model
    remains free, at that exact point, to either keep extending the name
    or to close it and move on to the next literal, and both options are
    surfaced to the token-masking layer.
    """

    def __init__(
        self, primitives: tuple[GrammarState, ...], index: int = 0
    ) -> None:
        self._primitives = primitives
        self._index = index

    def _current(self) -> GrammarState:
        return self._primitives[self._index]

    def step(self, char: str) -> Sequence | None:
        current = self._current()
        stepped = current.step(char)
        if stepped is not None:
            new_primitives = (
                self._primitives[: self._index]
                + (stepped,)
                + self._primitives[self._index + 1:]
            )
            return Sequence(new_primitives, self._index)
        if current.can_exit() and self._index + 1 < len(self._primitives):
            return Sequence(self._primitives, self._index + 1).step(char)
        return None

    def can_exit(self) -> bool:
        return (
            self._index == len(self._primitives) - 1
            and self._current().can_exit()
        )

    def state_key(self) -> Hashable:
        return (self._index, self._current().state_key())

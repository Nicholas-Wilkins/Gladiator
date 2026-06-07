"""
Transposition table for negamax search.

Stores previously-evaluated positions keyed by Zobrist hash
(provided by python-chess's built-in __hash__).  Cached entries
include the search depth, score, type flag (exact / lower bound /
upper bound), and the best move for move ordering.
"""

from __future__ import annotations

import chess

HASH_EXACT = 0
HASH_ALPHA = 1
HASH_BETA = 2


class _Entry:
    __slots__ = ("depth", "score", "flag", "move")

    def __init__(self, depth: int, score: float, flag: int, move: chess.Move | None = None):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move


class TranspositionTable:
    def __init__(self, max_size: int = 500_000):
        self._table: dict[int, _Entry] = {}
        self._max_size = max_size

    def probe(self, key: int, depth: int, alpha: float, beta: float) -> tuple[float | None, chess.Move | None]:
        entry = self._table.get(key)
        if entry is None or entry.depth < depth:
            return None, None

        if entry.flag == HASH_EXACT:
            return entry.score, entry.move
        if entry.flag == HASH_BETA and entry.score >= beta:
            return entry.score, entry.move
        if entry.flag == HASH_ALPHA and entry.score <= alpha:
            return entry.score, entry.move

        return None, entry.move

    def store(self, key: int, depth: int, score: float, flag: int, move: chess.Move | None = None) -> None:
        prev = self._table.get(key)
        if prev is not None and prev.depth > depth:
            return
        self._table[key] = _Entry(depth, score, flag, move)
        if len(self._table) > self._max_size:
            self._prune()

    def _prune(self) -> None:
        items = sorted(self._table.items(), key=lambda x: -x[1].depth)
        self._table = dict(items[: len(items) // 2])

    def clear(self) -> None:
        self._table.clear()

    def size(self) -> int:
        return len(self._table)

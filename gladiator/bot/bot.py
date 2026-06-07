"""Bot: wraps BotParams + search into a callable that picks moves."""

from __future__ import annotations

import threading

import chess
import numpy as np

from gladiator.bot.params import BotParams
from gladiator.bot.search import best_move
from gladiator.bot.transposition import TranspositionTable


class Bot:
    def __init__(self, params: BotParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(abs(hash(params.bot_id)) % (2 ** 32))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board, depth: int | None = None, stop_event: threading.Event | None = None) -> chess.Move:
        """Return the best legal move for the current position."""
        tt = TranspositionTable()
        return best_move(board, self.params, self.rng, depth=depth, tt=tt, stop_event=stop_event)

    @property
    def bot_id(self) -> str:
        return self.params.bot_id

    @property
    def generation(self) -> int:
        return self.params.generation

    def __repr__(self) -> str:
        return f"Bot(id={self.bot_id[:8]}, gen={self.generation}, depth={self.params.search_depth})"

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int, rng: np.random.Generator) -> "Bot":
        return cls(BotParams.random(generation, rng))

    def mutate(self, generation: int, rng: np.random.Generator) -> "Bot":
        return Bot(self.params.mutate(generation, rng))

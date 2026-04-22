"""Bot: wraps BotParams + search into a callable that picks moves."""

from __future__ import annotations

import chess
import numpy as np

from gladiator_mini.bot.params import BotParams
from gladiator_mini.bot.search import best_move


class Bot:
    def __init__(self, params: BotParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(abs(hash(params.bot_id)) % (2 ** 32))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return the best legal move for the current position."""
        return best_move(board, self.params, self.rng)

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

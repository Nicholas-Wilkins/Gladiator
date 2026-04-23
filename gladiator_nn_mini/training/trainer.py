"""
Training loop for Gladiator-NN-Mini.

Identical to the full NN trainer except RANDOM_WIN_THRESHOLD is 15
instead of 100, so mutation mode is reached much faster.

RANDOM mode   – challengers have freshly-initialised random network weights
MUTATION mode – challengers are noise-perturbed copies of the champion

The mode switches RANDOM → MUTATION once the champion wins 15 consecutive
matches against random-weight opponents.  Once MUTATION mode is entered it
is permanent — a new champion resets the counter but stays in MUTATION.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import torch

from gladiator_nn_mini.bot.bot import NNBot
from gladiator_nn_mini.storage.db import DB
from gladiator_nn_mini.training.match import MatchResult, play_match

logger = logging.getLogger(__name__)

RANDOM_WIN_THRESHOLD = 15


class Mode(Enum):
    RANDOM = auto()
    MUTATION = auto()


@dataclass
class TrainerState:
    champion: NNBot
    generation: int = 1
    consecutive_wins: int = 0
    mode: Mode = Mode.RANDOM
    total_matches: int = 0
    total_champion_changes: int = 0
    start_time: float = field(default_factory=time.time)


ProgressCallback = Callable[[TrainerState, MatchResult, NNBot], None]


class Trainer:
    def __init__(
        self,
        db: DB,
        device: torch.device,
        seed: Optional[int] = None,
        progress_cb: Optional[ProgressCallback] = None,
        move_cb=None,
        max_games: Optional[int] = None,
    ) -> None:
        self.db = db
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.progress_cb = progress_cb
        self.move_cb = move_cb
        self.max_games = max_games
        self._games_played = 0
        self._stop = False

        signal.signal(signal.SIGINT, self._handle_sigint)

    def run(self) -> None:
        state = self._load_or_bootstrap()
        logger.info("NN-Mini training started. Champion: %s", state.champion)

        while not self._stop:
            try:
                self._step(state)
            except Exception as exc:
                logger.error("Error during training step: %s", exc, exc_info=True)
                self.db.save_champion(state.champion, state)
                raise

        logger.info("NN-Mini training stopped gracefully.")

    def _step(self, state: TrainerState) -> None:
        state.generation += 1
        challenger = self._make_challenger(state)

        logger.info(
            "Gen %d | Mode: %s | Champion: %s | Challenger: %s",
            state.generation, state.mode.name, state.champion, challenger,
        )

        games_before = self._games_played

        def _pair_cb(partial: MatchResult) -> None:
            self._games_played = games_before + partial.total_games
            if self.progress_cb:
                self.progress_cb(state, partial, challenger)
            if self.max_games and self._games_played >= self.max_games:
                self._stop = True

        match_result = play_match(
            state.champion,
            challenger,
            rng=self.rng,
            progress_cb=_pair_cb,
            move_cb=self.move_cb,
        )
        self._games_played = games_before + match_result.total_games
        state.total_matches += 1

        champion_won = match_result.winner_id == state.champion.bot_id
        self.db.save_match(match_result, state)

        if champion_won:
            state.consecutive_wins += 1
            logger.info("Champion RETAINED. Streak: %d", state.consecutive_wins)
            if state.mode == Mode.RANDOM and state.consecutive_wins >= RANDOM_WIN_THRESHOLD:
                state.mode = Mode.MUTATION
                logger.info("*** MODE SWITCH: RANDOM → MUTATION ***")
        else:
            logger.info("Champion DETHRONED. New champion: %s", challenger)
            state.champion = challenger
            state.consecutive_wins = 0
            # mode is intentionally NOT reset — once MUTATION, always MUTATION
            state.total_champion_changes += 1

        self.db.save_champion(state.champion, state)

        if self.progress_cb:
            self.progress_cb(state, match_result, challenger)

    def _make_challenger(self, state: TrainerState) -> NNBot:
        if state.mode == Mode.MUTATION:
            return state.champion.mutate(state.generation, self.rng)
        return NNBot.random(state.generation, self.device, self.rng)

    def _load_or_bootstrap(self) -> TrainerState:
        saved = self.db.load_champion()
        if saved is not None:
            params, meta = saved
            champion = NNBot(params, self.device)
            state = TrainerState(
                champion=champion,
                generation=meta.get("generation", 1),
                consecutive_wins=meta.get("consecutive_wins", 0),
                mode=Mode[meta.get("mode", "RANDOM")],
                total_matches=meta.get("total_matches", 0),
                total_champion_changes=meta.get("total_champion_changes", 0),
            )
            logger.info("Resuming from saved champion (gen %d).", state.generation)
            return state

        logger.info("No saved champion. Running inaugural match...")
        bot_a = NNBot.random(generation=1, device=self.device, rng=self.rng)
        bot_b = NNBot.random(generation=1, device=self.device, rng=self.rng)
        result = play_match(bot_a, bot_b, rng=self.rng)

        champion = bot_a if result.winner_id == bot_a.bot_id else bot_b
        state = TrainerState(champion=champion, generation=1)
        self.db.save_champion(champion, state)
        self.db.save_match(result, state)
        logger.info("Inaugural champion: %s", champion)
        return state

    def _handle_sigint(self, signum, frame) -> None:
        logger.warning("Interrupt received — stopping after current match...")
        self._stop = True

"""
Main training loop for Gladiator.

Modes
-----
RANDOM    – challengers are generated completely from RNG.
MUTATION  – challengers are mutations of the reigning champion.

The mode switches from RANDOM to MUTATION once the current champion
accumulates 100 consecutive wins against random challengers.
Each time a new champion is crowned the consecutive-win counter resets
and the mode reverts to RANDOM so every champion must prove itself.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from gladiator.bot.bot import Bot
from gladiator.storage.db import DB
from gladiator.training.match import MatchResult, play_match

logger = logging.getLogger(__name__)

RANDOM_WIN_THRESHOLD = 100  # consecutive random wins before switching to mutation


class Mode(Enum):
    RANDOM = auto()
    MUTATION = auto()


@dataclass
class TrainerState:
    champion: Bot
    generation: int = 1
    consecutive_wins: int = 0
    mode: Mode = Mode.RANDOM
    total_matches: int = 0
    total_champion_changes: int = 0
    start_time: float = field(default_factory=time.time)


# Callback type: receives (state, match_result, challenger) after each match
ProgressCallback = Callable[[TrainerState, MatchResult, Bot], None]


class Trainer:
    def __init__(
        self,
        db: DB,
        seed: Optional[int] = None,
        progress_cb: Optional[ProgressCallback] = None,
        move_cb=None,
        max_games: Optional[int] = None,
    ) -> None:
        self.db = db
        self.rng = np.random.default_rng(seed)
        self.progress_cb = progress_cb
        self.move_cb = move_cb
        self.max_games = max_games
        self._games_played = 0
        self._stop = False

        # Intercept SIGINT for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_sigint)

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Run the training loop indefinitely (until interrupted)."""
        state = self._load_or_bootstrap()
        logger.info("Training started. Champion: %s", state.champion)

        while not self._stop:
            try:
                self._step(state)
            except Exception as exc:
                logger.error("Error during training step: %s", exc, exc_info=True)
                # Save state and re-raise so the caller can handle it
                self.db.save_champion(state.champion, state)
                raise

        logger.info("Training stopped gracefully.")

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

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

        # Save match before updating state.champion so the correct champion_id is recorded
        self.db.save_match(match_result, state)

        if champion_won:
            state.consecutive_wins += 1
            logger.info(
                "Champion RETAINED after %d game(s). Streak: %d",
                match_result.total_games, state.consecutive_wins,
            )
            # Mode upgrade: random-win streak hit threshold
            if state.mode == Mode.RANDOM and state.consecutive_wins >= RANDOM_WIN_THRESHOLD:
                state.mode = Mode.MUTATION
                logger.info(
                    "*** MODE SWITCH: RANDOM → MUTATION after %d consecutive wins ***",
                    state.consecutive_wins,
                )
        else:
            logger.info(
                "Champion DETHRONED after %d game(s). New champion: %s",
                match_result.total_games, challenger,
            )
            state.champion = challenger
            state.consecutive_wins = 0
            state.mode = Mode.RANDOM        # every new champion starts fresh
            state.total_champion_changes += 1

        self.db.save_champion(state.champion, state)

        if self.progress_cb:
            self.progress_cb(state, match_result, challenger)

    def _make_challenger(self, state: TrainerState) -> Bot:
        if state.mode == Mode.MUTATION:
            return state.champion.mutate(state.generation, self.rng)
        return Bot.random(state.generation, self.rng)

    def _load_or_bootstrap(self) -> TrainerState:
        """Load existing champion from DB, or run the very first match to crown one."""
        saved = self.db.load_champion()
        if saved is not None:
            champion, meta = saved
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

        # First run: generate two random bots and crown the winner
        logger.info("No saved champion found. Running inaugural match...")
        bot_a = Bot.random(generation=1, rng=self.rng)
        bot_b = Bot.random(generation=1, rng=self.rng)
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

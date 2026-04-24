"""
SQLite persistence for Gladiator-Mini.

Tables
------
champion      – current champion params + trainer state
champion_log  – every champion ever crowned (lineage history)
matches       – one row per match (summary)
games         – one row per game (PGN + metadata)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple

from gladiator_mini.bot.bot import Bot
from gladiator_mini.bot.params import BotParams
from gladiator_mini.training.match import MatchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS champion (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    bot_id              TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    generation          INTEGER NOT NULL,
    consecutive_wins    INTEGER NOT NULL DEFAULT 0,
    mode                TEXT NOT NULL DEFAULT 'RANDOM',
    total_matches       INTEGER NOT NULL DEFAULT 0,
    total_champion_changes INTEGER NOT NULL DEFAULT 0,
    saved_at            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS champion_log (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id              TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    generation          INTEGER NOT NULL,
    crowned_at          REAL NOT NULL,
    total_matches_at_crowning INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS matches (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    champion_id         TEXT NOT NULL,
    challenger_id       TEXT NOT NULL,
    champion_won        INTEGER NOT NULL,   -- 1 = champion kept title, 0 = dethroned
    score_champion      REAL NOT NULL,
    score_challenger    REAL NOT NULL,
    pairs_played        INTEGER NOT NULL,
    total_games         INTEGER NOT NULL,
    generation          INTEGER NOT NULL,
    mode                TEXT NOT NULL,
    played_at           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_rowid         INTEGER NOT NULL REFERENCES matches(rowid),
    pair_number         INTEGER NOT NULL,
    game_in_pair        INTEGER NOT NULL,   -- 1 or 2
    white_id            TEXT NOT NULL,
    black_id            TEXT NOT NULL,
    result              TEXT NOT NULL,
    termination         TEXT NOT NULL,
    halfmoves           INTEGER NOT NULL,
    duration_s          REAL NOT NULL,
    pgn                 TEXT NOT NULL
);
"""


class DB:
    def __init__(self, path: str | Path = "gladiator_mini.db") -> None:
        self.path = Path(path)
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_SCHEMA)
        self._con.commit()

    # ------------------------------------------------------------------ #
    # Champion                                                             #
    # ------------------------------------------------------------------ #

    def save_champion(self, bot: Bot, state) -> None:
        """Upsert the current champion row and append to champion_log."""
        now = time.time()
        params_json = bot.params.to_json()

        self._con.execute(
            """
            INSERT INTO champion
                (id, bot_id, params_json, generation, consecutive_wins, mode,
                 total_matches, total_champion_changes, saved_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                bot_id = excluded.bot_id,
                params_json = excluded.params_json,
                generation = excluded.generation,
                consecutive_wins = excluded.consecutive_wins,
                mode = excluded.mode,
                total_matches = excluded.total_matches,
                total_champion_changes = excluded.total_champion_changes,
                saved_at = excluded.saved_at
            """,
            (
                bot.bot_id, params_json, state.generation,
                state.consecutive_wins, state.mode.name,
                state.total_matches, state.total_champion_changes, now,
            ),
        )

        existing = self._con.execute(
            "SELECT bot_id FROM champion_log WHERE bot_id = ?", (bot.bot_id,)
        ).fetchone()
        if existing is None:
            self._con.execute(
                """
                INSERT INTO champion_log
                    (bot_id, params_json, generation, crowned_at, total_matches_at_crowning)
                VALUES (?, '{}', ?, ?, ?)
                """,
                (bot.bot_id, state.generation, now, state.total_matches),
            )

        self._con.commit()

    def load_champion(self) -> Optional[Tuple[Bot, dict]]:
        """Return (Bot, meta_dict) for the saved champion, or None."""
        row = self._con.execute("SELECT * FROM champion WHERE id = 1").fetchone()
        if row is None:
            return None
        params = BotParams.from_json(row["params_json"])
        bot = Bot(params)
        meta = {
            "generation": row["generation"],
            "consecutive_wins": row["consecutive_wins"],
            "mode": row["mode"],
            "total_matches": row["total_matches"],
            "total_champion_changes": row["total_champion_changes"],
        }
        return bot, meta

    # ------------------------------------------------------------------ #
    # Match + game logging                                                 #
    # ------------------------------------------------------------------ #

    def save_match(self, result: MatchResult, state) -> None:
        """Persist a completed match and all its games."""
        champion_id = state.champion.bot_id

        if result.winner_id == champion_id:
            challenger_id = result.loser_id
            champion_won = 1
            score_champion = result.total_score_a if result.pairs[0].game1.winner == champion_id or result.pairs[0].game1.winner is None else result.total_score_b
        else:
            challenger_id = result.winner_id
            champion_won = 0
            score_champion = result.total_score_a if _is_bot_a(result, champion_id) else result.total_score_b

        score_challenger = result.total_score_a + result.total_score_b - score_champion

        cur = self._con.execute(
            """
            INSERT INTO matches
                (champion_id, challenger_id, champion_won, score_champion, score_challenger,
                 pairs_played, total_games, generation, mode, played_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                champion_id, challenger_id, champion_won,
                score_champion, score_challenger,
                len(result.pairs), result.total_games,
                state.generation, state.mode.name, time.time(),
            ),
        )
        match_rowid = cur.lastrowid

        for pair_num, pair in enumerate(result.pairs, start=1):
            for game_in_pair, game in enumerate([pair.game1, pair.game2], start=1):
                white_id = _white_id_in_game(game_in_pair, champion_id, challenger_id)
                black_id = challenger_id if white_id == champion_id else champion_id
                self._con.execute(
                    """
                    INSERT INTO games
                        (match_rowid, pair_number, game_in_pair, white_id, black_id,
                         result, termination, halfmoves, duration_s, pgn)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_rowid, pair_num, game_in_pair,
                        white_id, black_id,
                        game.result_str, game.termination,
                        game.halfmoves, game.duration_s, game.pgn,
                    ),
                )

        self._con.commit()

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def recent_matches(self, n: int = 20) -> list[dict]:
        rows = self._con.execute(
            """
            SELECT champion_id, challenger_id, champion_won,
                   score_champion, score_challenger, pairs_played, total_games,
                   generation, mode, played_at
            FROM matches
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    def champion_lineage(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT bot_id, generation, crowned_at, total_matches_at_crowning FROM champion_log ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._con.close()


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _is_bot_a(result: MatchResult, bot_id: str) -> bool:
    if not result.pairs:
        return True
    g1 = result.pairs[0].game1
    return g1.winner == bot_id or (g1.winner is None and result.total_score_a >= result.total_score_b)


def _white_id_in_game(game_in_pair: int, champion_id: str, challenger_id: str) -> str:
    """game 1: champion is white; game 2: challenger is white."""
    return champion_id if game_in_pair == 1 else challenger_id

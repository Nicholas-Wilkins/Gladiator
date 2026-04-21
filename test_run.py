"""
Diagnostic test run: plays up to MAX_GAMES total games, exports each as a PGN,
and writes a detailed log. Run with:

    python test_run.py

Outputs:
    test_pgns/         -- one .pgn file per game
    test_run.log       -- full debug log
    test_summary.jsonl -- one JSON line per game (machine-readable)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

MAX_GAMES = 100
PGN_DIR = Path("test_pgns")
LOG_FILE = Path("test_run.log")
SUMMARY_FILE = Path("test_summary.jsonl")
DB_PATH = "test_gladiator.db"
SEED = 42

# ------------------------------------------------------------------ #
# Logging setup (file + stdout, DEBUG level)                          #
# ------------------------------------------------------------------ #

PGN_DIR.mkdir(exist_ok=True)
LOG_FILE.unlink(missing_ok=True)
SUMMARY_FILE.unlink(missing_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Game counter + PGN export via monkey-patch                          #
# ------------------------------------------------------------------ #

game_counter = 0
match_counter = 0


class GameLimitReached(Exception):
    pass


import gladiator.training.match as _match_module

_original_play_game = _match_module.play_game


def _instrumented_play_game(white, black):
    global game_counter

    t0 = time.time()
    result = _original_play_game(white, black)
    elapsed = time.time() - t0

    game_counter += 1

    pgn_name = (
        f"game_{game_counter:04d}"
        f"_w{white.bot_id[:8]}_d{white.params.search_depth}"
        f"_b{black.bot_id[:8]}_d{black.params.search_depth}"
        f"_{result.termination}.pgn"
    )
    (PGN_DIR / pgn_name).write_text(result.pgn)

    logger.info(
        "Game %3d | white=...%s(gen=%d,depth=%d) black=...%s(gen=%d,depth=%d)"
        " → %s (%s) | %d halfmoves | %.2fs",
        game_counter,
        white.bot_id[-8:], white.generation, white.params.search_depth,
        black.bot_id[-8:], black.generation, black.params.search_depth,
        result.result_str,
        result.termination,
        result.halfmoves,
        result.duration_s,
    )

    record = {
        "game_num": game_counter,
        "white_id": white.bot_id,
        "white_gen": white.generation,
        "white_depth": white.params.search_depth,
        "black_id": black.bot_id,
        "black_gen": black.generation,
        "black_depth": black.params.search_depth,
        "result": result.result_str,
        "termination": result.termination,
        "halfmoves": result.halfmoves,
        "duration_s": round(result.duration_s, 3),
        "winner_id": result.winner,
    }
    with SUMMARY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    if game_counter >= MAX_GAMES:
        raise GameLimitReached(f"Reached {MAX_GAMES} games — stopping.")

    return result


_match_module.play_game = _instrumented_play_game

# ------------------------------------------------------------------ #
# Also log each match outcome                                          #
# ------------------------------------------------------------------ #

_original_play_match = _match_module.play_match


def _instrumented_play_match(bot_a, bot_b, rng=None, progress_cb=None):
    global match_counter
    match_counter += 1
    logger.info(
        "--- Match %d start: A=...%s(gen=%d,depth=%d) vs B=...%s(gen=%d,depth=%d) ---",
        match_counter,
        bot_a.bot_id[-8:], bot_a.generation, bot_a.params.search_depth,
        bot_b.bot_id[-8:], bot_b.generation, bot_b.params.search_depth,
    )
    result = _original_play_match(bot_a, bot_b, rng=rng, progress_cb=progress_cb)
    logger.info(
        "--- Match %d end: winner=...%s | score %.1f-%.1f | %d pairs ---",
        match_counter,
        result.winner_id[-8:],
        result.total_score_a,
        result.total_score_b,
        len(result.pairs),
    )
    return result


_match_module.play_match = _instrumented_play_match

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #

from gladiator.storage.db import DB
from gladiator.training.trainer import Trainer

logger.info("=== Gladiator diagnostic test run ===")
logger.info("MAX_GAMES=%d  SEED=%d  DB=%s", MAX_GAMES, SEED, DB_PATH)

db = DB(DB_PATH)
trainer = Trainer(db=db, seed=SEED)

try:
    trainer.run()
except GameLimitReached as e:
    logger.info("DONE: %s", e)
except Exception as e:
    logger.error("Unexpected error: %s", e, exc_info=True)
finally:
    db.close()

logger.info("Games played: %d | Matches played: %d", game_counter, match_counter)
logger.info("PGNs written to: %s/", PGN_DIR)
logger.info("Summary:         %s", SUMMARY_FILE)
logger.info("Full log:        %s", LOG_FILE)

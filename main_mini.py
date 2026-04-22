"""
Gladiator-Mini — self-improving chess engine with a lower mutation threshold.

Identical to the full Gladiator engine but enters MUTATION mode after 15
consecutive wins instead of 100, allowing mutation phase to be reached faster.

Usage
-----
    python main_mini.py                  # start / resume training with dashboard
    python main_mini.py --no-ui          # headless (logs to stdout only)
    python main_mini.py --seed 42        # deterministic RNG seed
    python main_mini.py --db path/to.db  # custom database path
    python main_mini.py --history        # print champion lineage and exit
    python main_mini.py --stats          # print recent match stats and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gladiator-Mini chess engine trainer")
    p.add_argument("--db", default="gladiator_mini.db", help="SQLite database path")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--no-ui", action="store_true", help="Disable the Rich dashboard")
    p.add_argument("--history", action="store_true", help="Print champion lineage and exit")
    p.add_argument("--stats", action="store_true", help="Print recent match stats and exit")
    p.add_argument("--max-games", type=int, default=None, help="Stop after this many games (for testing)")
    return p.parse_args()


def _print_history(db) -> None:
    from rich.table import Table
    from rich import print as rprint
    import datetime

    lineage = db.champion_lineage()
    table = Table(title="Champion Lineage (Mini)", show_lines=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Bot ID")
    table.add_column("Generation", justify="right")
    table.add_column("Crowned At")
    table.add_column("Matches At Crowning", justify="right")

    for i, row in enumerate(lineage, 1):
        crowned = datetime.datetime.fromtimestamp(row["crowned_at"]).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(
            str(i),
            row["bot_id"][:16] + "…",
            str(row["generation"]),
            crowned,
            str(row["total_matches_at_crowning"]),
        )
    rprint(table)


def _print_stats(db) -> None:
    from rich.table import Table
    from rich import print as rprint

    matches = db.recent_matches(30)
    table = Table(title="Recent Match Results — Mini (last 30)", show_lines=False)
    table.add_column("Gen", justify="right")
    table.add_column("Mode")
    table.add_column("Outcome")
    table.add_column("Champion score", justify="center")
    table.add_column("Challenger score", justify="center")
    table.add_column("Pairs", justify="right")

    for r in matches:
        outcome = "[green]KEPT[/green]" if r["champion_won"] else "[red]DETHRONED[/red]"
        table.add_row(
            str(r["generation"]),
            r["mode"],
            outcome,
            f"{r['score_champion']:.1f}",
            f"{r['score_challenger']:.1f}",
            str(r["pairs_played"]),
        )
    rprint(table)


def main() -> None:
    args = _parse_args()

    from gladiator_mini.storage.db import DB
    db = DB(args.db)

    if args.history:
        _print_history(db)
        db.close()
        return

    if args.stats:
        _print_stats(db)
        db.close()
        return

    from gladiator_mini.training.trainer import Trainer

    if args.no_ui:
        trainer = Trainer(db=db, seed=args.seed, max_games=args.max_games)
        try:
            trainer.run()
        finally:
            db.close()
        return

    from gladiator_mini.ui.dashboard import Dashboard

    with Dashboard(db) as dash:
        post_match_cb, _ = dash.make_progress_cb()
        move_cb = dash.make_move_cb()

        trainer = Trainer(
            db=db,
            seed=args.seed,
            progress_cb=post_match_cb,
            move_cb=move_cb,
            max_games=args.max_games,
        )

        try:
            trainer.run()
        finally:
            db.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export a Gladiator bot as a self-contained UCI engine script.

The output script embeds the bot's parameters directly — no .db file needed.
Recipients only need Python, python-chess, numpy (and torch for NN engines)
plus the gladiator package installed from this repo.

Usage
-----
  python export_bot.py                          # auto-select from gladiator.db
  python export_bot.py --engine nn              # auto-select from gladiator_nn.db
  python export_bot.py --db gladiator_worker_1.db
  python export_bot.py --db gladiator_worker_1.db --bot-id 981432a4-...
  python export_bot.py --engine nn --db gladiator_nn_worker_0.db --output my_bot.py

Auto-selection rules
--------------------
  RANDOM mode  → bot with highest historical streak whose params are stored
  MUTATION mode → current reigning champion
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

_CPU_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Self-contained Gladiator UCI engine — no database file required.

Bot:         {bot_id}
Generation:  {generation}
Best streak: {streak} consecutive wins
Engine:      {engine_label}

Requirements
------------
  1. Clone the Gladiator repo
  2. cd into it and run: python install.py
  3. Point Nibbler (or any UCI GUI) at this script
"""

from __future__ import annotations

import sys
import threading

import chess

_PARAMS_JSON = {params_json_repr}
_BOT_NAME = "{bot_name}"


def _load_bot():
    from gladiator{pkg_suffix}.bot.params import BotParams
    from gladiator{pkg_suffix}.bot.bot import Bot
    return Bot(BotParams.from_json(_PARAMS_JSON))


def _parse_position(tokens):
    board = chess.Board()
    idx = 1
    if idx >= len(tokens):
        return board
    if tokens[idx] == "startpos":
        idx += 1
    elif tokens[idx] == "fen":
        idx += 1
        fen_parts = []
        while idx < len(tokens) and tokens[idx] != "moves":
            fen_parts.append(tokens[idx])
            idx += 1
        board = chess.Board(" ".join(fen_parts))
    if idx < len(tokens) and tokens[idx] == "moves":
        idx += 1
        for uci_move in tokens[idx:]:
            board.push_uci(uci_move)
    return board


class _Engine:
    def __init__(self):
        self._bot = _load_bot()
        self._board = chess.Board()
        self._thread = None
        self._stop = threading.Event()

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            if cmd == "uci":
                print(f"id name {{_BOT_NAME}}", flush=True)
                print("id author Gladiator", flush=True)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self._board = chess.Board()
            elif cmd == "position":
                self._board = _parse_position(tokens)
            elif cmd == "go":
                depth = 5
                if "depth" in tokens:
                    try:
                        depth = int(tokens[tokens.index("depth") + 1])
                    except (IndexError, ValueError):
                        pass
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._search, args=(self._board.copy(), depth), daemon=True
                )
                self._thread.start()
            elif cmd == "stop":
                self._stop.set()
                if self._thread:
                    self._thread.join()
            elif cmd == "quit":
                self._stop.set()
                if self._thread:
                    self._thread.join()
                break

    def _search(self, board, depth):
        move = self._bot.choose_move(board, depth=depth)
        print(f"info depth {{depth}}", flush=True)
        print(f"bestmove {{move.uci()}}", flush=True)


if __name__ == "__main__":
    _Engine().run()
'''

_NN_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Self-contained Gladiator NN UCI engine — no database file required.

Bot:         {bot_id}
Generation:  {generation}
Best streak: {streak} consecutive wins
Engine:      {engine_label}

Requirements
------------
  1. Clone the Gladiator repo
  2. cd into it and run: python install.py
  3. Point Nibbler (or any UCI GUI) at this script
"""

from __future__ import annotations

import sys
import threading

import chess

_PARAMS_JSON = {params_json_repr}
_BOT_NAME = "{bot_name}"


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def _load_bot(device):
    from gladiator{pkg_suffix}.bot.bot import NNBot, NNBotParams
    return NNBot(NNBotParams.from_json(_PARAMS_JSON), device)


def _parse_position(tokens):
    board = chess.Board()
    idx = 1
    if idx >= len(tokens):
        return board
    if tokens[idx] == "startpos":
        idx += 1
    elif tokens[idx] == "fen":
        idx += 1
        fen_parts = []
        while idx < len(tokens) and tokens[idx] != "moves":
            fen_parts.append(tokens[idx])
            idx += 1
        board = chess.Board(" ".join(fen_parts))
    if idx < len(tokens) and tokens[idx] == "moves":
        idx += 1
        for uci_move in tokens[idx:]:
            board.push_uci(uci_move)
    return board


class _Engine:
    def __init__(self):
        device = _pick_device()
        self._bot = _load_bot(device)
        self._board = chess.Board()
        self._thread = None
        self._stop = threading.Event()

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            if cmd == "uci":
                print(f"id name {{_BOT_NAME}}", flush=True)
                print("id author Gladiator", flush=True)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self._board = chess.Board()
            elif cmd == "position":
                self._board = _parse_position(tokens)
            elif cmd == "go":
                depth = 5
                if "depth" in tokens:
                    try:
                        depth = int(tokens[tokens.index("depth") + 1])
                    except (IndexError, ValueError):
                        pass
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._search, args=(self._board.copy(), depth), daemon=True
                )
                self._thread.start()
            elif cmd == "stop":
                self._stop.set()
                if self._thread:
                    self._thread.join()
            elif cmd == "quit":
                self._stop.set()
                if self._thread:
                    self._thread.join()
                break

    def _search(self, board, depth):
        move = self._bot.choose_move(board)
        print(f"info depth {{depth}}", flush=True)
        print(f"bestmove {{move.uci()}}", flush=True)


if __name__ == "__main__":
    _Engine().run()
'''

# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------

_ENGINES = {
    "cpu":     {"default_db": "gladiator.db",        "pkg_suffix": "",       "nn": False, "label": "CPU heuristic"},
    "nn":      {"default_db": "gladiator_nn.db",      "pkg_suffix": "_nn",    "nn": True,  "label": "Neural network (GPU/CPU)"},
    "mini":    {"default_db": "gladiator_mini.db",    "pkg_suffix": "_mini",  "nn": False, "label": "CPU mini"},
    "nn-mini": {"default_db": "gladiator_nn_mini.db", "pkg_suffix": "_nn_mini","nn": True, "label": "Neural network mini"},
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _best_streak_bot(con: sqlite3.Connection) -> tuple[str, str, int, int] | None:
    """
    Return (bot_id, params_json, generation, streak) for the bot with the
    highest historical streak whose params are actually stored in champion_log.

    Streak is derived from gaps between champion_log entries (gap method):
        streak[i] = total_matches_at_crowning[i+1] - total_matches_at_crowning[i] - 1
    The -1 accounts for the losing match that crowned the next champion.
    """
    log = con.execute(
        "SELECT bot_id, params_json, generation, total_matches_at_crowning "
        "FROM champion_log ORDER BY rowid"
    ).fetchall()

    best = None
    best_streak = -1

    for i in range(len(log) - 1):
        streak = max(0, log[i + 1][3] - log[i][3] - 1)
        params = log[i][1]
        if streak > best_streak and params and params != "{}":
            best_streak = streak
            best = (log[i][0], params, log[i][2], streak)

    return best


def _current_champion(con: sqlite3.Connection) -> tuple[str, str, int, int, str]:
    """Return (bot_id, params_json, generation, consecutive_wins, mode)."""
    row = con.execute(
        "SELECT bot_id, params_json, generation, consecutive_wins, mode "
        "FROM champion WHERE id = 1"
    ).fetchone()
    if row is None:
        raise SystemExit("No champion found in database.")
    return row[0], row[1], row[2], row[3], row[4]


def _bot_by_id(con: sqlite3.Connection, bot_id: str) -> tuple[str, str, int] | None:
    """Look up a bot by ID in champion or champion_log."""
    row = con.execute(
        "SELECT bot_id, params_json, generation FROM champion WHERE id=1 AND bot_id=?",
        (bot_id,),
    ).fetchone()
    if row:
        return row[0], row[1], row[2]

    row = con.execute(
        "SELECT bot_id, params_json, generation FROM champion_log WHERE bot_id=?",
        (bot_id,),
    ).fetchone()
    if row and row[1] and row[1] != "{}":
        return row[0], row[1], row[2]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Gladiator bot as a self-contained UCI engine script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Auto-selection rules (when --bot-id is not given):
              RANDOM mode  → bot with the highest historical streak that has stored params
              MUTATION mode → current reigning champion
        """),
    )
    parser.add_argument(
        "--engine", choices=list(_ENGINES), default="cpu",
        help="Engine type (default: cpu)",
    )
    parser.add_argument(
        "--db", metavar="PATH",
        help="Path to the SQLite database (default: engine's standard DB name)",
    )
    parser.add_argument(
        "--bot-id", metavar="UUID",
        help="Specific bot UUID to export (default: auto-select)",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Output file path (default: auto-generated name, printed to stderr)",
    )
    args = parser.parse_args()

    cfg = _ENGINES[args.engine]
    db_path = Path(args.db) if args.db else Path(cfg["default_db"])

    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    con = sqlite3.connect(str(db_path))

    # Select bot
    if args.bot_id:
        result = _bot_by_id(con, args.bot_id)
        if result is None:
            sys.exit(
                f"Bot {args.bot_id} not found or params not stored in {db_path}.\n"
                "Note: champion_log only stores params for bots crowned after the params-fix was applied."
            )
        bot_id, params_json, generation = result
        # Estimate streak via gap method
        log = con.execute(
            "SELECT bot_id, total_matches_at_crowning FROM champion_log ORDER BY rowid"
        ).fetchall()
        streak = 0
        for i, (lid, lm) in enumerate(log):
            if lid == bot_id and i + 1 < len(log):
                streak = max(0, log[i + 1][1] - lm - 1)
                break
        # If still reigning, use consecutive_wins
        cur = con.execute(
            "SELECT consecutive_wins FROM champion WHERE id=1 AND bot_id=?", (bot_id,)
        ).fetchone()
        if cur:
            streak = cur[0]
    else:
        # Auto-select based on mode
        bot_id, params_json, generation, consecutive_wins, mode = _current_champion(con)

        if mode == "MUTATION":
            streak = consecutive_wins
            print(f"[export_bot] MUTATION mode — exporting current champion {bot_id[:8]} (streak={streak})", file=sys.stderr)
        else:
            best = _best_streak_bot(con)
            if best and best[3] >= consecutive_wins:
                bot_id, params_json, generation, streak = best
                print(f"[export_bot] RANDOM mode — best historical streak={streak}, bot={bot_id[:8]}", file=sys.stderr)
            else:
                streak = consecutive_wins
                print(f"[export_bot] RANDOM mode — no stored historical params; using current champion {bot_id[:8]} (streak={streak})", file=sys.stderr)

    con.close()

    if not params_json or params_json == "{}":
        sys.exit(
            f"Bot {bot_id} has no stored params (params_json is empty).\n"
            "This can happen for bots crowned before the params-fix was applied.\n"
            "Only the current champion and bots crowned after the fix have recoverable params."
        )

    # Build output filename
    streak_label = f"streak{streak}" if streak else f"gen{generation}"
    engine_label = cfg["label"]
    bot_name = f"Gladiator-{args.engine.upper()} {streak_label} gen={generation} id={bot_id[:8]}"

    safe_id = bot_id.replace("-", "")[:8]
    auto_name = f"uci_{args.engine}_{streak_label}_{safe_id}.py"

    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir():
            out_path = out_path / auto_name
    else:
        out_dir = Path(__file__).parent / "exported_bots"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / auto_name

    # Render template
    params_json_repr = repr(params_json)
    template = _NN_TEMPLATE if cfg["nn"] else _CPU_TEMPLATE
    script = template.format(
        bot_id=bot_id,
        generation=generation,
        streak=streak,
        engine_label=engine_label,
        bot_name=bot_name,
        params_json_repr=params_json_repr,
        pkg_suffix=cfg["pkg_suffix"],
    )

    out_path.write_text(script)
    out_path.chmod(0o755)
    print(str(out_path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export a Gladiator bot as a self-contained UCI engine script.

The output script has zero external dependencies beyond pip packages — no repo
clone, no .db file, no venv hunting. Recipients just install deps and run.

  CPU/mini:  pip install python-chess numpy
  NN/nn-mini: pip install python-chess numpy torch

Usage
-----
  python export_bot.py                          # auto-select from gladiator.db
  python export_bot.py --engine nn              # auto-select from gladiator_nn.db
  python export_bot.py --db gladiator_worker_1.db
  python export_bot.py --db gladiator_worker_1.db --bot-id 981432a4-...
  python export_bot.py --engine nn --db gladiator_nn_worker_0.db --output my_bot.py
  python export_bot.py -o /some/directory/     # saves with auto-generated name

Auto-selection rules
--------------------
  RANDOM mode  → bot with highest historical streak whose params are stored
  MUTATION mode → current reigning champion
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Engine source inlining
# ---------------------------------------------------------------------------

def _inline_sources(paths: list[Path]) -> str:
    """
    Read source files, strip internal gladiator imports and __future__ imports,
    and concatenate. Duplicate stdlib/third-party imports are harmless in Python.
    """
    parts = []
    for path in paths:
        lines = []
        for line in path.read_text().splitlines():
            # Drop intra-package imports — we're inlining everything
            if re.match(r"from gladiator", line):
                continue
            if line.startswith("from __future__"):
                continue
            lines.append(line)
        parts.append("\n".join(lines).strip())
    return "\n\n\n".join(parts)


def _build_cpu_engine(repo_root: Path) -> str:
    base = repo_root / "gladiator" / "bot"
    return _inline_sources([
        base / "tables.py",
        base / "params.py",
        base / "evaluator.py",
        base / "search.py",
        base / "bot.py",
    ])


def _build_mini_engine(repo_root: Path) -> str:
    base = repo_root / "gladiator_mini" / "bot"
    return _inline_sources([
        base / "tables.py",
        base / "params.py",
        base / "evaluator.py",
        base / "search.py",
        base / "bot.py",
    ])


def _build_nn_engine(repo_root: Path) -> str:
    return _inline_sources([
        repo_root / "gladiator_nn" / "bot" / "network.py",
        repo_root / "gladiator_nn" / "training" / "search.py",
        repo_root / "gladiator_nn" / "bot" / "bot.py",
    ])


def _build_nn_mini_engine(repo_root: Path) -> str:
    return _inline_sources([
        repo_root / "gladiator_nn_mini" / "bot" / "network.py",
        repo_root / "gladiator_nn_mini" / "training" / "search.py",
        repo_root / "gladiator_nn_mini" / "bot" / "bot.py",
    ])


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

_BOOTSTRAP = '''\
import sys as _sys, os as _os, subprocess as _sp, pathlib as _pl

def _bootstrap(pkgs):
    """Ensure packages are available, creating a private venv if needed."""
    try:
        for p in pkgs:
            __import__(p)
        return  # all deps present
    except ImportError:
        pass

    _venv = _pl.Path(_os.environ.get("GLADIATOR_VENV", "")) or (
        _pl.Path.home() / ".local" / "share" / "gladiator_engines" / "venv"
    )
    _py = _venv / ("Scripts/python.exe" if _sys.platform == "win32" else "bin/python")

    if not _py.exists():
        _sp.check_call([_sys.executable, "-m", "venv", str(_venv)])
        _sp.check_call([str(_py), "-m", "pip", "install", "--quiet", *[p.replace("_", "-") for p in pkgs]])

    _os.execv(str(_py), [str(_py)] + _sys.argv)

'''

_CPU_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Self-contained Gladiator UCI engine.

Bot:         {bot_id}
Generation:  {generation}
Best streak: {streak} consecutive wins
Engine:      {engine_label}
"""

''' + _BOOTSTRAP + '''\
_bootstrap(["chess", "numpy"])

import sys
import threading
import json
import uuid
import numpy as np
import chess

# ---------------------------------------------------------------------------
# Inlined engine code
# ---------------------------------------------------------------------------

{engine_code}

# ---------------------------------------------------------------------------
# Bot parameters (embedded at export time)
# ---------------------------------------------------------------------------

_PARAMS_JSON = {params_json_repr}
_BOT_NAME = "{bot_name}"


def _load_bot():
    return Bot(BotParams.from_json(_PARAMS_JSON))


# ---------------------------------------------------------------------------
# UCI loop
# ---------------------------------------------------------------------------

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
Self-contained Gladiator NN UCI engine.

Bot:         {bot_id}
Generation:  {generation}
Best streak: {streak} consecutive wins
Engine:      {engine_label}
"""

''' + _BOOTSTRAP + '''\
_bootstrap(["chess", "numpy", "torch"])

import sys
import threading
import base64
import io
import json
import uuid
import numpy as np
import chess
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Inlined engine code
# ---------------------------------------------------------------------------

{engine_code}

# ---------------------------------------------------------------------------
# Bot parameters (embedded at export time)
# ---------------------------------------------------------------------------

_PARAMS_JSON = {params_json_repr}
_BOT_NAME = "{bot_name}"


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def _load_bot():
    device = _pick_device()
    return NNBot(NNBotParams.from_json(_PARAMS_JSON), device)


# ---------------------------------------------------------------------------
# UCI loop
# ---------------------------------------------------------------------------

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
    "cpu":     {"default_db": "gladiator.db",        "nn": False, "label": "CPU heuristic"},
    "nn":      {"default_db": "gladiator_nn.db",      "nn": True,  "label": "Neural network (GPU/CPU)"},
    "mini":    {"default_db": "gladiator_mini.db",    "nn": False, "label": "CPU mini"},
    "nn-mini": {"default_db": "gladiator_nn_mini.db", "nn": True,  "label": "Neural network mini"},
}

_ENGINE_CODE_BUILDERS = {
    "cpu":     _build_cpu_engine,
    "nn":      _build_nn_engine,
    "mini":    _build_mini_engine,
    "nn-mini": _build_nn_mini_engine,
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _best_streak_bot(con: sqlite3.Connection):
    """
    Return (bot_id, params_json, generation, streak) for the highest-streak
    bot in champion_log that has stored params, or None.
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


def _current_champion(con: sqlite3.Connection):
    row = con.execute(
        "SELECT bot_id, params_json, generation, consecutive_wins, mode "
        "FROM champion WHERE id = 1"
    ).fetchone()
    if row is None:
        sys.exit("No champion found in database.")
    return row[0], row[1], row[2], row[3], row[4]


def _bot_by_id(con: sqlite3.Connection, bot_id: str):
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
            The output script is fully standalone — no repo or venv needed.
            Just: pip install python-chess numpy   (add torch for NN engines)

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
        help="Output file path or directory (default: exported_bots/ inside repo)",
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
                "Only bots crowned after the params-fix have recoverable params."
            )
        bot_id, params_json, generation = result
        log = con.execute(
            "SELECT bot_id, total_matches_at_crowning FROM champion_log ORDER BY rowid"
        ).fetchall()
        streak = 0
        for i, (lid, lm) in enumerate(log):
            if lid == bot_id and i + 1 < len(log):
                streak = max(0, log[i + 1][1] - lm - 1)
                break
        cur = con.execute(
            "SELECT consecutive_wins FROM champion WHERE id=1 AND bot_id=?", (bot_id,)
        ).fetchone()
        if cur:
            streak = cur[0]
    else:
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
                print(f"[export_bot] RANDOM mode — using current champion {bot_id[:8]} (streak={streak})", file=sys.stderr)

    con.close()

    if not params_json or params_json == "{}":
        sys.exit(
            f"Bot {bot_id} has no stored params.\n"
            "Only bots crowned after the params-fix have recoverable params."
        )

    # Build engine code by inlining source files
    repo_root = Path(__file__).parent
    engine_code = _ENGINE_CODE_BUILDERS[args.engine](repo_root)

    # Build names / paths
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
        out_dir = repo_root / "exported_bots"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / auto_name

    # Render template — escape braces in inlined source before .format()
    template = _NN_TEMPLATE if cfg["nn"] else _CPU_TEMPLATE
    script = template.format(
        bot_id=bot_id,
        generation=generation,
        streak=streak,
        engine_label=engine_label,
        bot_name=bot_name,
        params_json_repr=repr(params_json),
        engine_code=engine_code,
    )

    out_path.write_text(script)
    out_path.chmod(0o755)
    print(str(out_path))


if __name__ == "__main__":
    main()

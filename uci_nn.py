"""
UCI engine wrapper for the Gladiator-NN champion bot.

Usage
-----
    .venv/bin/python uci_nn.py
    .venv/bin/python uci_nn.py --db path/to/gladiator_nn.db
    .venv/bin/python uci_nn.py --db path/to/gladiator_nn.db --bot-id <uuid>
    .venv/bin/python uci_nn.py --cpu

Without --bot-id, loads the current champion from the database.
With --bot-id, loads that specific bot from champion_log by UUID.

Point any UCI-compatible GUI (Nibbler, Arena, ChessBase, etc.) at this script
as a custom engine.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import threading
from typing import Optional

import chess
import numpy as np


def _send(line: str) -> None:
    print(line, flush=True)


def _pick_device(force_cpu: bool):
    import torch
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def _parse_position(tokens: list[str]) -> chess.Board:
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


def _load_nn_bot_by_id(db_path: str, bot_id: str, device):
    from gladiator_nn.bot.bot import NNBot, NNBotParams

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT params_json, generation FROM champion_log WHERE bot_id = ?", (bot_id,)
    ).fetchone()
    con.close()

    if row is None:
        raise ValueError(f"Bot ID '{bot_id}' not found in champion_log of '{db_path}'")

    params = NNBotParams.from_json(row[0])
    return NNBot(params, device), row[1]


class UCINNEngine:
    def __init__(self, db_path: str, device, bot_id: Optional[str] = None) -> None:
        from gladiator_nn.bot.bot import NNBot

        if bot_id:
            self._bot, gen = _load_nn_bot_by_id(db_path, bot_id, device)
            self._name = f"Gladiator-NN gen={gen} id={bot_id[:8]}"
        else:
            from gladiator_nn.storage.db import DB
            db = DB(db_path)
            saved = db.load_champion()
            db.close()

            if saved is None:
                self._bot = NNBot.random(generation=0, device=device,
                                         rng=np.random.default_rng())
                self._name = "Gladiator-NN (untrained)"
            else:
                params, meta = saved
                self._bot = NNBot(params, device)
                gen = meta.get("generation", params.generation)
                self._name = f"Gladiator-NN gen={gen}"

        self._board = chess.Board()
        self._search_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                _send(f"id name {self._name}")
                _send("id author Gladiator")
                _send("uciok")

            elif cmd == "isready":
                _send("readyok")

            elif cmd == "ucinewgame":
                self._board = chess.Board()

            elif cmd == "position":
                self._board = _parse_position(tokens)

            elif cmd == "go":
                self._stop_event.clear()
                self._search_thread = threading.Thread(
                    target=self._search, args=(self._board.copy(),), daemon=True
                )
                self._search_thread.start()

            elif cmd == "stop":
                self._stop_event.set()
                if self._search_thread:
                    self._search_thread.join()

            elif cmd == "quit":
                self._stop_event.set()
                if self._search_thread:
                    self._search_thread.join()
                break

    def _search(self, board: chess.Board) -> None:
        move = self._bot.choose_move(board)
        if not self._stop_event.is_set():
            _send(f"info depth {self._bot.params.search_depth}")
            _send(f"bestmove {move.uci()}")
        else:
            _send(f"bestmove {move.uci()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gladiator-NN UCI engine")
    parser.add_argument("--db", default="gladiator_nn.db", help="SQLite database path")
    parser.add_argument("--bot-id", default=None, help="Load a specific past champion by UUID from champion_log")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = _pick_device(args.cpu)
    UCINNEngine(args.db, device, bot_id=args.bot_id).run()


if __name__ == "__main__":
    main()

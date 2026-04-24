#!/home/nick/Projects/consortium/consortium-projects/colosseum/gladiator/.venv/bin/python
"""
Plug-and-play UCI wrapper for the CPU streak champion from gladiator_worker_2.db.

Bot:        e6282976-02f7-4ea9-bdf1-571e1141287c
Generation: 7869
Best streak: 27 consecutive wins

Point Nibbler (or any UCI GUI) at this script — no arguments needed.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import chess

_DB = str(Path(__file__).parent / "gladiator_worker_2.db")
_BOT_ID = "e6282976-02f7-4ea9-bdf1-571e1141287c"


def _send(line: str) -> None:
    print(line, flush=True)


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


def _load_bot():
    from gladiator.bot.params import BotParams
    from gladiator.bot.bot import Bot

    con = sqlite3.connect(_DB)
    row = con.execute(
        "SELECT params_json, generation FROM champion_log WHERE bot_id = ?", (_BOT_ID,)
    ).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Bot {_BOT_ID} not found in {_DB}")

    params = BotParams.from_json(row[0])
    return Bot(params), row[1]


class _Engine:
    def __init__(self) -> None:
        self._bot, gen = _load_bot()
        self._name = f"Gladiator CPU streak=27 gen={gen} id={_BOT_ID[:8]}"
        self._board = chess.Board()
        self._thread = None
        self._stop = threading.Event()

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

    def _search(self, board: chess.Board, depth: int) -> None:
        move = self._bot.choose_move(board, depth=depth)
        _send(f"info depth {depth}")
        _send(f"bestmove {move.uci()}")


if __name__ == "__main__":
    _Engine().run()

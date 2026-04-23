"""
Plug-and-play UCI wrapper for the NN streak champion from gladiator_nn_worker_0.db.

Bot:        87193391-3ba1-41a7-b42a-c6921e340f2f
Generation: 2031
Best streak: 19 consecutive wins

Point Nibbler (or any UCI GUI) at this script — no arguments needed.
Runs on CUDA if available, DirectML (AMD/Windows) next, then CPU.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import chess

_DB = str(Path(__file__).parent / "gladiator_nn_worker_0.db")
_BOT_ID = "87193391-3ba1-41a7-b42a-c6921e340f2f"


def _send(line: str) -> None:
    print(line, flush=True)


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


def _load_bot(device):
    from gladiator_nn.bot.bot import NNBot, NNBotParams

    con = sqlite3.connect(_DB)
    row = con.execute(
        "SELECT params_json, generation FROM champion_log WHERE bot_id = ?", (_BOT_ID,)
    ).fetchone()
    con.close()

    if row is None:
        raise RuntimeError(f"Bot {_BOT_ID} not found in {_DB}")

    params = NNBotParams.from_json(row[0])
    return NNBot(params, device), row[1]


class _Engine:
    def __init__(self) -> None:
        device = _pick_device()
        self._bot, gen = _load_bot(device)
        self._name = f"Gladiator-NN streak=19 gen={gen} id={_BOT_ID[:8]}"
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
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._search, args=(self._board.copy(),), daemon=True
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

    def _search(self, board: chess.Board) -> None:
        move = self._bot.choose_move(board)
        _send(f"info depth {self._bot.params.search_depth}")
        _send(f"bestmove {move.uci()}")


if __name__ == "__main__":
    _Engine().run()

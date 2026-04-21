"""
Match logic for the NN engine — identical structure to the CPU version.

Scoring:
  win  = 1.0 point
  draw = 0.5 points
  loss = 0.0 points

A match is decided by cumulative score over game pairs (each bot plays once as
white and once as black).  Tie → play another pair, up to _MAX_PAIRS, then
random tiebreak.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import chess.pgn

from gladiator_nn.bot.bot import NNBot


@dataclass
class GameResult:
    pgn: str
    winner: Optional[str]
    result_str: str
    termination: str
    halfmoves: int
    duration_s: float


@dataclass
class PairResult:
    game1: GameResult
    game2: GameResult
    score_a: float
    score_b: float


@dataclass
class MatchResult:
    pairs: List[PairResult] = field(default_factory=list)
    total_score_a: float = 0.0
    total_score_b: float = 0.0
    winner_id: str = ""
    loser_id: str = ""

    @property
    def total_games(self) -> int:
        return len(self.pairs) * 2


def play_game(white: NNBot, black: NNBot, move_cb=None) -> GameResult:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white.bot_id
    game.headers["WhiteDepth"] = str(white.params.search_depth)
    game.headers["WhiteGen"] = str(white.generation)
    game.headers["Black"] = black.bot_id
    game.headers["BlackDepth"] = str(black.params.search_depth)
    game.headers["BlackGen"] = str(black.generation)

    node = game
    t0 = time.time()

    while not (
        board.is_checkmate()
        or board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition()
    ):
        current = white if board.turn == chess.WHITE else black
        move = current.choose_move(board)

        if move not in board.legal_moves:
            raise RuntimeError(
                f"NNBot {current.bot_id} produced illegal move {move} in {board.fen()}"
            )

        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)

        if move_cb:
            move_cb(board.copy(), san, white.bot_id, black.bot_id)

    duration = time.time() - t0

    if board.is_checkmate():
        termination = "checkmate"
        winner_id = white.bot_id if board.turn == chess.BLACK else black.bot_id
        result_str = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate():
        termination, winner_id, result_str = "stalemate", None, "1/2-1/2"
    elif board.is_fifty_moves():
        termination, winner_id, result_str = "50-move", None, "1/2-1/2"
    elif board.is_repetition():
        termination, winner_id, result_str = "repetition", None, "1/2-1/2"
    else:
        termination, winner_id, result_str = "material", None, "1/2-1/2"

    game.headers["Result"] = result_str
    return GameResult(
        pgn=str(game),
        winner=winner_id,
        result_str=result_str,
        termination=termination,
        halfmoves=len(board.move_stack),
        duration_s=duration,
    )


def play_pair(bot_a: NNBot, bot_b: NNBot, move_cb=None) -> PairResult:
    game1 = play_game(white=bot_a, black=bot_b, move_cb=move_cb)
    game2 = play_game(white=bot_b, black=bot_a, move_cb=move_cb)
    score_a = _score_for(bot_a.bot_id, game1) + _score_for(bot_a.bot_id, game2)
    score_b = _score_for(bot_b.bot_id, game1) + _score_for(bot_b.bot_id, game2)
    return PairResult(game1=game1, game2=game2, score_a=score_a, score_b=score_b)


def _score_for(bot_id: str, game: GameResult) -> float:
    if game.winner == bot_id:
        return 1.0
    if game.winner is None:
        return 0.5
    return 0.0


_MAX_PAIRS = 6


def play_match(
    bot_a: NNBot,
    bot_b: NNBot,
    rng=None,
    progress_cb=None,
    move_cb=None,
) -> MatchResult:
    import random as _random
    result = MatchResult()
    pair_num = 0

    while True:
        pair_num += 1
        pair = play_pair(bot_a, bot_b, move_cb=move_cb)
        result.pairs.append(pair)
        result.total_score_a += pair.score_a
        result.total_score_b += pair.score_b

        if progress_cb:
            progress_cb(result)

        if result.total_score_a != result.total_score_b:
            break

        if pair_num >= _MAX_PAIRS:
            if (rng.random() if rng is not None else _random.random()) < 0.5:
                result.total_score_a += 0.001
            else:
                result.total_score_b += 0.001
            break

    if result.total_score_a > result.total_score_b:
        result.winner_id, result.loser_id = bot_a.bot_id, bot_b.bot_id
    else:
        result.winner_id, result.loser_id = bot_b.bot_id, bot_a.bot_id

    return result

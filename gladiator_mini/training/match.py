"""
Match logic: single game, game pair, and full match series.

Scoring:
  win  = 1.0 point
  draw = 0.5 points
  loss = 0.0 points

A match is decided by cumulative score over game pairs (each bot plays
once as white and once as black).  If the score is tied after a pair,
another pair is played.  This continues until one bot leads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess
import chess.pgn

from gladiator_mini.bot.bot import Bot


@dataclass
class GameResult:
    pgn: str
    winner: Optional[str]   # bot_id of winner, or None for draw
    result_str: str          # "1-0", "0-1", "1/2-1/2"
    termination: str         # "checkmate", "stalemate", "material"
    halfmoves: int
    duration_s: float


@dataclass
class PairResult:
    game1: GameResult        # bot_a plays white
    game2: GameResult        # bot_b plays white
    score_a: float
    score_b: float


@dataclass
class MatchResult:
    pairs: List[PairResult] = field(default_factory=list)
    total_score_a: float = 0.0
    total_score_b: float = 0.0
    winner_id: str = ""      # bot_id of match winner
    loser_id: str = ""

    @property
    def total_games(self) -> int:
        return len(self.pairs) * 2


def play_game(white: Bot, black: Bot, move_cb=None) -> GameResult:
    """Play a single game; returns a GameResult."""
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
        current_bot = white if board.turn == chess.WHITE else black
        move = current_bot.choose_move(board)

        if move not in board.legal_moves:
            raise RuntimeError(
                f"Bot {current_bot.bot_id} produced illegal move {move} "
                f"in position {board.fen()}"
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
        termination = "stalemate"
        winner_id = None
        result_str = "1/2-1/2"
    elif board.is_fifty_moves():
        termination = "50-move"
        winner_id = None
        result_str = "1/2-1/2"
    elif board.is_repetition():
        termination = "repetition"
        winner_id = None
        result_str = "1/2-1/2"
    else:  # insufficient material
        termination = "material"
        winner_id = None
        result_str = "1/2-1/2"

    game.headers["Result"] = result_str
    pgn_str = str(game)

    return GameResult(
        pgn=pgn_str,
        winner=winner_id,
        result_str=result_str,
        termination=termination,
        halfmoves=len(board.move_stack),
        duration_s=duration,
    )


def play_pair(bot_a: Bot, bot_b: Bot, move_cb=None) -> PairResult:
    """
    Play one game pair: bot_a as white in game 1, bot_b as white in game 2.
    Returns scores for bot_a and bot_b.
    """
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


_MAX_PAIRS = 6  # after this many tied pairs, toss a coin


def play_match(bot_a: Bot, bot_b: Bot, rng=None, progress_cb=None, move_cb=None) -> MatchResult:
    """
    Play game pairs until one bot leads.  Calls progress_cb(match_result)
    after each pair if provided.  If still tied after _MAX_PAIRS pairs,
    the winner is chosen randomly.
    """
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
        result.winner_id = bot_a.bot_id
        result.loser_id = bot_b.bot_id
    else:
        result.winner_id = bot_b.bot_id
        result.loser_id = bot_a.bot_id

    return result

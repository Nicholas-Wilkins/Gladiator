"""
Negamax search with alpha-beta pruning.

Negamax property: the value of a position to the side to move is the
negation of the value to the opponent, so we never branch on "whose turn".
"""

from __future__ import annotations

import chess

from gladiator.bot.evaluator import evaluate
from gladiator.bot.params import BotParams

_INF = float("inf")
_CONTEMPT = -25  # centipawns: claimable draws are slightly bad for the side to move


def best_move(board: chess.Board, params: BotParams, rng=None) -> chess.Move:
    """Return the best move for the side to move according to this bot."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
    if len(legal) == 1:
        return legal[0]

    best: chess.Move = legal[0]
    best_score = -_INF
    alpha = -_INF
    beta = _INF

    for move in _order_moves(board, legal):
        board.push(move)
        score = -_negamax(board, params, params.search_depth - 1, -beta, -alpha)
        board.pop()

        # Sub-centipawn jitter breaks ties without affecting real score differences
        if rng is not None:
            score += rng.uniform(-0.5, 0.5)

        if score > best_score:
            best_score = score
            best = move
        alpha = max(alpha, score)

    return best


def _negamax(
    board: chess.Board,
    params: BotParams,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    if board.is_repetition() or board.is_fifty_moves():
        return _CONTEMPT
    if depth == 0 or board.is_game_over():
        return evaluate(board, params)

    best_score = -_INF
    for move in _order_moves(board, list(board.legal_moves)):
        board.push(move)
        score = -_negamax(board, params, depth - 1, -beta, -alpha)
        board.pop()

        if score > best_score:
            best_score = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # beta cutoff

    return best_score


# ------------------------------------------------------------------ #
# Move ordering                                                        #
# ------------------------------------------------------------------ #

_PIECE_VALUE_ROUGH = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """
    Score moves for ordering:
      captures first (MVV-LVA: most-valuable victim, least-valuable attacker),
      then promotions,
      then quiet moves.
    Higher score → examined first → better alpha-beta pruning.
    """
    def _key(move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_val = _PIECE_VALUE_ROUGH.get(victim.piece_type, 0) if victim else 0
            attacker_val = _PIECE_VALUE_ROUGH.get(attacker.piece_type, 0) if attacker else 0
            score += 10 * victim_val - attacker_val + 1000
        if move.promotion:
            score += 800
        return score

    return sorted(moves, key=_key, reverse=True)

"""
Negamax search with alpha-beta pruning.

Negamax property: the value of a position to the side to move is the
negation of the value to the opponent, so we never branch on "whose turn".
"""

from __future__ import annotations

import threading

import chess

from gladiator.bot.evaluator import evaluate
from gladiator.bot.params import BotParams
from gladiator.bot.transposition import HASH_ALPHA, HASH_BETA, HASH_EXACT, TranspositionTable

_INF = float("inf")
_CONTEMPT = -25  # centipawns: claimable draws are slightly bad for the side to move


def best_move(
    board: chess.Board,
    params: BotParams,
    rng=None,
    depth: int | None = None,
    tt: TranspositionTable | None = None,
    stop_event: threading.Event | None = None,
) -> chess.Move:
    """Return the best move for the side to move according to this bot."""
    search_depth = depth if depth is not None else params.search_depth
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
    if len(legal) == 1:
        return legal[0]

    key = board._transposition_key()
    _, tt_move = tt.probe(key, search_depth, -_INF, _INF) if tt is not None else (None, None)

    best: chess.Move = legal[0]
    best_score = -_INF
    alpha = -_INF
    beta = _INF

    for move in _order_moves(board, legal, tt_move):
        if stop_event and stop_event.is_set():
            if best_score == -_INF:
                best = move
            break
        board.push(move)
        score = -_negamax(board, params, search_depth - 1, -beta, -alpha, tt, stop_event)
        board.pop()

        # Sub-centipawn jitter breaks ties without affecting real score differences
        if rng is not None:
            score += rng.uniform(-0.5, 0.5)

        if score > best_score:
            best_score = score
            best = move
        alpha = max(alpha, score)

    if tt is not None:
        flag = HASH_EXACT
        if best_score <= -_INF + 1:
            flag = HASH_ALPHA
        tt.store(key, search_depth, best_score, flag, best)

    return best


def _negamax(
    board: chess.Board,
    params: BotParams,
    depth: int,
    alpha: float,
    beta: float,
    tt: TranspositionTable | None = None,
    stop_event: threading.Event | None = None,
) -> float:
    if stop_event and stop_event.is_set():
        return 0.0

    key = board._transposition_key()

    if board.is_repetition() or board.is_fifty_moves():
        return _CONTEMPT

    if tt is not None:
        tt_score, _ = tt.probe(key, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

    if depth == 0 or board.is_game_over():
        sc = evaluate(board, params)
        if tt is not None:
            tt.store(key, depth, sc, HASH_EXACT)
        return sc

    best_score = -_INF
    best_move_inner = None
    orig_alpha = alpha
    for move in _order_moves(board, list(board.legal_moves)):
        if stop_event and stop_event.is_set():
            break
        board.push(move)
        score = -_negamax(board, params, depth - 1, -beta, -alpha, tt, stop_event)
        board.pop()

        if score > best_score:
            best_score = score
            best_move_inner = move
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # beta cutoff

    if tt is not None:
        flag = HASH_ALPHA if best_score <= orig_alpha else (HASH_BETA if best_score >= beta else HASH_EXACT)
        tt.store(key, depth, best_score, flag, best_move_inner)

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


def _order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None = None,
) -> list[chess.Move]:
    """
    Score moves for ordering:
      captures first (MVV-LVA: most-valuable victim, least-valuable attacker),
      then promotions,
      then quiet moves.
    Higher score → examined first → better alpha-beta pruning.
    A TT-best move (if given) is always examined first.
    """
    if tt_move and tt_move in moves:
        moves = [m for m in moves if m != tt_move]

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

    rest = sorted(moves, key=_key, reverse=True)
    return [tt_move] + rest if tt_move else rest

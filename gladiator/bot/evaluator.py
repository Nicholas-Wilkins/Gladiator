"""
Position evaluation from the perspective of the side to move.
Returns positive values when the side to move is winning.
All scores are in centipawns.
"""

from __future__ import annotations

import chess
import numpy as np

from gladiator.bot.params import BotParams

# Squares used for centre control bonus
_CENTRE = frozenset([chess.D4, chess.D5, chess.E4, chess.E5])
_EXTENDED_CENTRE = frozenset([
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
])

_CHECKMATE_SCORE = 100_000
_DRAW_SCORE = 0


def evaluate(board: chess.Board, params: BotParams) -> float:
    """
    Evaluate the position from the perspective of board.turn.
    The caller (negamax) is responsible for negating the value at each ply.
    """
    if board.is_checkmate():
        # Side to move is in checkmate → very bad for the side to move
        return -_CHECKMATE_SCORE

    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()):
        return _DRAW_SCORE

    us = board.turn
    them = not us

    score = 0.0

    # ------------------------------------------------------------------ #
    # Material + piece-square tables                                       #
    # ------------------------------------------------------------------ #
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        pv = params.piece_values[piece.piece_type]
        # PST index: white uses rank-flipped index, black uses raw index
        if piece.color == chess.WHITE:
            pst_idx = sq ^ 56
        else:
            pst_idx = sq
        pst_val = params.pst[piece.piece_type][pst_idx]

        total = pv + pst_val
        if piece.color == us:
            score += total
        else:
            score -= total

    # ------------------------------------------------------------------ #
    # Mobility (number of legal moves available to each side)             #
    # ------------------------------------------------------------------ #
    if params.mobility_weight != 0.0:
        our_mobility = _count_mobility(board, us)
        their_mobility = _count_mobility(board, them)
        score += params.mobility_weight * (our_mobility - their_mobility)

    # ------------------------------------------------------------------ #
    # Pawn structure                                                       #
    # ------------------------------------------------------------------ #
    if params.pawn_structure_weight != 0.0:
        score += params.pawn_structure_weight * _pawn_structure(board, us, them)

    # ------------------------------------------------------------------ #
    # King safety                                                          #
    # ------------------------------------------------------------------ #
    if params.king_safety_weight != 0.0:
        score += params.king_safety_weight * _king_safety(board, us, them)

    return score


# ------------------------------------------------------------------ #
# Helper: mobility                                                    #
# ------------------------------------------------------------------ #

def _count_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count pseudo-legal moves for a color (fast; no null-move push needed)."""
    return sum(1 for m in board.pseudo_legal_moves if board.color_at(m.from_square) == color)


# ------------------------------------------------------------------ #
# Helper: pawn structure                                              #
# ------------------------------------------------------------------ #

def _pawn_structure(board: chess.Board, us: chess.Color, them: chess.Color) -> float:
    score = 0.0
    us_pawns = board.pieces(chess.PAWN, us)
    them_pawns = board.pieces(chess.PAWN, them)

    us_files = [chess.square_file(sq) for sq in us_pawns]
    them_files = [chess.square_file(sq) for sq in them_pawns]

    for sq in us_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        # Doubled pawn penalty
        if us_files.count(f) > 1:
            score -= 15

        # Isolated pawn penalty
        adjacent = [f - 1, f + 1]
        if not any(af in us_files for af in adjacent if 0 <= af <= 7):
            score -= 20

        # Passed pawn bonus — no enemy pawns on same or adjacent files ahead
        direction = 1 if us == chess.WHITE else -1
        ranks_ahead = range(r + direction, 8 if direction == 1 else -1, direction)
        blocking = False
        for rf in ranks_ahead:
            for df in [f - 1, f, f + 1]:
                if 0 <= df <= 7 and chess.square(df, rf) in them_pawns:
                    blocking = True
                    break
            if blocking:
                break
        if not blocking:
            # Bonus scales with how far advanced the pawn is
            advance = (r if us == chess.WHITE else 7 - r)
            score += 10 + advance * 5

    # Mirror for opponent
    for sq in them_pawns:
        f = chess.square_file(sq)

        if them_files.count(f) > 1:
            score += 15

        adjacent = [f - 1, f + 1]
        if not any(af in them_files for af in adjacent if 0 <= af <= 7):
            score += 20

    return score


# ------------------------------------------------------------------ #
# Helper: king safety                                                 #
# ------------------------------------------------------------------ #

def _king_safety(board: chess.Board, us: chess.Color, them: chess.Color) -> float:
    score = 0.0

    our_king_sq = board.king(us)
    their_king_sq = board.king(them)

    if our_king_sq is None or their_king_sq is None:
        return 0.0

    # Pawn shield: count friendly pawns near king
    our_shield = _pawn_shield(board, our_king_sq, us)
    their_shield = _pawn_shield(board, their_king_sq, them)
    score += (our_shield - their_shield) * 10

    # Attackers near king
    our_attackers_near_their_king = _attackers_near(board, their_king_sq, us)
    their_attackers_near_our_king = _attackers_near(board, our_king_sq, them)
    score += (our_attackers_near_their_king - their_attackers_near_our_king) * 8

    return score


def _pawn_shield(board: chess.Board, king_sq: int, color: chess.Color) -> int:
    """Count pawns in the 3 squares directly in front of the king."""
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    direction = 1 if color == chess.WHITE else -1
    shield = 0
    for df in [-1, 0, 1]:
        nf = f + df
        nr = r + direction
        if 0 <= nf <= 7 and 0 <= nr <= 7:
            sq = chess.square(nf, nr)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield += 1
    return shield


def _attackers_near(board: chess.Board, king_sq: int, attacking_color: chess.Color) -> int:
    """Count attacking-color pieces that attack squares adjacent to king_sq."""
    total = 0
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        total += len(board.attackers(attacking_color, sq))
    return total

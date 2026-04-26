#!/usr/bin/env python3
"""
Self-contained Gladiator UCI engine.

Bot:         981432a4-4c96-473b-a3ad-c3f465c66405
Generation:  5456
Best streak: 27 consecutive wins
Engine:      CPU heuristic
"""

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

"""
Base piece-square tables (PSTs) from white's perspective.
Index 0 = a8, index 63 = h1 (visual top-left to bottom-right).
White piece on square sq  -> PST[sq ^ 56]
Black piece on square sq  -> PST[sq]  (then negate for white-relative eval)
"""

import numpy as np

PAWN = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.float32)

KNIGHT = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.float32)

BISHOP = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.float32)

ROOK = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
], dtype=np.float32)

QUEEN = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.float32)

KING_MIDGAME = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
], dtype=np.float32)

KING_ENDGAME = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.float32)

# Keyed by python-chess piece type constants (1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING)
BASE_PST = {
    1: PAWN,
    2: KNIGHT,
    3: BISHOP,
    4: ROOK,
    5: QUEEN,
    6: KING_MIDGAME,
}

BASE_PIECE_VALUES = {
    1: 100.0,   # pawn
    2: 320.0,   # knight
    3: 330.0,   # bishop
    4: 500.0,   # rook
    5: 900.0,   # queen
    6: 20000.0, # king (never captured, but used for eval consistency)
}


"""BotParams: the full parameter set that defines a bot's behaviour."""


import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np



@dataclass
class BotParams:
    bot_id: str
    generation: int
    parent_id: Optional[str]

    # Centipawn values keyed by piece type int (1-6)
    piece_values: Dict[int, float]

    # 64-element arrays keyed by piece type int
    pst: Dict[int, np.ndarray]

    # Scalar eval-feature weights
    mobility_weight: float
    pawn_structure_weight: float
    king_safety_weight: float

    # Search depth (2-4)
    search_depth: int

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "piece_values": self.piece_values,
            "pst": {str(k): v.tolist() for k, v in self.pst.items()},
            "mobility_weight": self.mobility_weight,
            "pawn_structure_weight": self.pawn_structure_weight,
            "king_safety_weight": self.king_safety_weight,
            "search_depth": self.search_depth,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "BotParams":
        return cls(
            bot_id=d["bot_id"],
            generation=d["generation"],
            parent_id=d.get("parent_id"),
            piece_values={int(k): float(v) for k, v in d["piece_values"].items()},
            pst={int(k): np.array(v, dtype=np.float32) for k, v in d["pst"].items()},
            mobility_weight=float(d["mobility_weight"]),
            pawn_structure_weight=float(d["pawn_structure_weight"]),
            king_safety_weight=float(d["king_safety_weight"]),
            search_depth=int(d["search_depth"]),
        )

    @classmethod
    def from_json(cls, s: str) -> "BotParams":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------ #
    # Factory: random                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int, rng: np.random.Generator) -> "BotParams":
        """Completely random bot — all parameters drawn fresh from RNG."""
        piece_values = {
            pt: float(base * rng.uniform(0.7, 1.3))
            for pt, base in BASE_PIECE_VALUES.items()
        }
        piece_values[6] = BASE_PIECE_VALUES[6]  # keep king value fixed

        pst = {
            pt: (base_table + rng.normal(0, 15, size=64)).astype(np.float32)
            for pt, base_table in BASE_PST.items()
        }

        return cls(
            bot_id=str(uuid.uuid4()),
            generation=generation,
            parent_id=None,
            piece_values=piece_values,
            pst=pst,
            mobility_weight=float(rng.uniform(0.0, 0.5)),
            pawn_structure_weight=float(rng.uniform(0.0, 0.5)),
            king_safety_weight=float(rng.uniform(0.0, 0.5)),
            search_depth=2,
        )

    # ------------------------------------------------------------------ #
    # Factory: mutate                                                      #
    # ------------------------------------------------------------------ #

    def mutate(self, generation: int, rng: np.random.Generator) -> "BotParams":
        """Produce a child by perturbing this bot's parameters."""

        def _mutate_scalar(v: float, scale: float = 0.10) -> float:
            return float(v * (1.0 + rng.normal(0, scale)))

        piece_values = {
            pt: _mutate_scalar(val, scale=0.08)
            for pt, val in self.piece_values.items()
        }
        piece_values[6] = BASE_PIECE_VALUES[6]  # keep king fixed

        pst = {
            pt: np.clip(arr + rng.normal(0, 5, size=64), -100, 100).astype(np.float32)
            for pt, arr in self.pst.items()
        }

        new_depth = 2

        return BotParams(
            bot_id=str(uuid.uuid4()),
            generation=generation,
            parent_id=self.bot_id,
            piece_values=piece_values,
            pst=pst,
            mobility_weight=_mutate_scalar(self.mobility_weight, 0.15),
            pawn_structure_weight=_mutate_scalar(self.pawn_structure_weight, 0.15),
            king_safety_weight=_mutate_scalar(self.king_safety_weight, 0.15),
            search_depth=new_depth,
        )


"""
Position evaluation from the perspective of the side to move.
Returns positive values when the side to move is winning.
All scores are in centipawns.
"""


import chess
import numpy as np


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


"""
Negamax search with alpha-beta pruning.

Negamax property: the value of a position to the side to move is the
negation of the value to the opponent, so we never branch on "whose turn".
"""


import chess


_INF = float("inf")
_CONTEMPT = -25  # centipawns: claimable draws are slightly bad for the side to move


def best_move(board: chess.Board, params: BotParams, rng=None, depth: int | None = None) -> chess.Move:
    """Return the best move for the side to move according to this bot."""
    search_depth = depth if depth is not None else params.search_depth
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
        score = -_negamax(board, params, search_depth - 1, -beta, -alpha)
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


"""Bot: wraps BotParams + search into a callable that picks moves."""


import chess
import numpy as np



class Bot:
    def __init__(self, params: BotParams) -> None:
        self.params = params
        self.rng = np.random.default_rng(abs(hash(params.bot_id)) % (2 ** 32))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board, depth: int | None = None) -> chess.Move:
        """Return the best legal move for the current position."""
        return best_move(board, self.params, self.rng, depth=depth)

    @property
    def bot_id(self) -> str:
        return self.params.bot_id

    @property
    def generation(self) -> int:
        return self.params.generation

    def __repr__(self) -> str:
        return f"Bot(id={self.bot_id[:8]}, gen={self.generation}, depth={self.params.search_depth})"

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int, rng: np.random.Generator) -> "Bot":
        return cls(BotParams.random(generation, rng))

    def mutate(self, generation: int, rng: np.random.Generator) -> "Bot":
        return Bot(self.params.mutate(generation, rng))

# ---------------------------------------------------------------------------
# Bot parameters (embedded at export time)
# ---------------------------------------------------------------------------

_PARAMS_JSON = '{"bot_id": "981432a4-4c96-473b-a3ad-c3f465c66405", "generation": 5456, "parent_id": null, "piece_values": {"1": 89.13185666170669, "2": 283.91555690105633, "3": 308.82206440796887, "4": 569.9215271016438, "5": 1099.8486043317268, "6": 20000.0}, "pst": {"1": [-5.31265926361084, 7.920295715332031, -6.829209804534912, -2.918180465698242, -14.553882598876953, 1.7065789699554443, -2.615128993988037, -13.419761657714844, 57.140647888183594, 64.7440414428711, 56.60845947265625, 32.519195556640625, 53.44443893432617, 61.03968811035156, 65.97514343261719, 67.29568481445312, 23.44092559814453, 6.825925827026367, 27.145862579345703, 39.881797790527344, 31.264816284179688, 24.425493240356445, -5.570062160491943, 14.065170288085938, -1.9492347240447998, 2.3935933113098145, 34.83042526245117, 27.921844482421875, 29.194473266601562, 0.3210209906101227, -17.335908889770508, -13.261614799499512, -5.359389305114746, -3.7899117469787598, -27.038068771362305, 9.656394004821777, 19.579322814941406, -12.516196250915527, -6.037397384643555, 1.2625089883804321, 22.539104461669922, -0.4899807870388031, 11.984803199768066, -20.83519172668457, 18.612751007080078, -13.061628341674805, 9.208405494689941, 33.93268585205078, 18.66485595703125, 15.884058952331543, 33.20480728149414, -27.45538902282715, -4.949611663818359, 15.430329322814941, 15.373279571533203, -4.196276664733887, 4.656894207000732, 5.127350807189941, -0.5343753695487976, 4.658936500549316, -20.741527557373047, 13.606813430786133, -3.942559003829956, 3.3767049312591553], "2": [-59.565452575683594, -45.401222229003906, -39.75155258178711, -11.713613510131836, -42.42252731323242, -41.28450012207031, -42.83408737182617, -68.21017456054688, -55.33174133300781, -17.176713943481445, 4.013328552246094, 10.86288070678711, 6.317152500152588, -13.101078987121582, -35.65951156616211, -25.052547454833984, -62.5987434387207, 1.9601852893829346, 19.720500946044922, 31.99293327331543, 34.82791519165039, 25.583370208740234, 27.924320220947266, -56.985904693603516, -34.63407897949219, 9.775225639343262, -18.718120574951172, 8.456153869628906, 37.04532241821289, 4.939501762390137, 19.5712890625, -20.783456802368164, -12.661306381225586, 12.063556671142578, 22.85309600830078, 29.96464729309082, 6.960097312927246, 39.714263916015625, -3.4718146324157715, -41.97975158691406, -23.240493774414062, -15.572861671447754, 40.769691467285156, 50.214820861816406, 32.755924224853516, -3.5405657291412354, 12.009440422058105, -25.519134521484375, -47.416378021240234, -23.889190673828125, 7.15566349029541, -1.8254610300064087, 5.1089653968811035, 1.3013194799423218, -38.00838088989258, -44.106658935546875, -55.338478088378906, -15.00741195678711, -26.93102264404297, -22.76032066345215, -25.36833381652832, -28.964426040649414, -37.580657958984375, -50.50422668457031], "3": [-50.01068115234375, -8.277253150939941, -18.061279296875, -13.2376708984375, -23.02375030517578, -22.39638900756836, -8.746977806091309, -25.702409744262695, -8.081856727600098, 4.566027641296387, 0.9944635033607483, 2.7870712280273438, -7.683195114135742, -2.129239320755005, 10.477789878845215, -11.4892578125, -0.15061603486537933, 2.0683765411376953, 21.20829963684082, -3.2697360515594482, 15.27611255645752, 18.636611938476562, -10.217549324035645, -6.629858016967773, -3.7633328437805176, 4.324605941772461, 19.835721969604492, 16.24192237854004, 7.5915374755859375, 18.4938907623291, 11.622746467590332, 32.813907623291016, -32.172237396240234, 3.4026095867156982, 10.289073944091797, 10.850081443786621, -18.81230926513672, 12.15516185760498, -7.651339530944824, 1.2040085792541504, -25.58272361755371, 35.75144577026367, 9.799491882324219, 3.171680212020874, 14.984014511108398, -11.170652389526367, -5.170260429382324, -1.1925928592681885, -14.493071556091309, 6.253660202026367, -33.375282287597656, -20.384492874145508, -25.478668212890625, 24.071243286132812, 2.1128666400909424, 8.380762100219727, -24.712329864501953, 4.702744960784912, -18.853958129882812, -17.75643539428711, -13.056947708129883, -9.721096992492676, -40.813377380371094, -21.31976318359375], "4": [-15.247892379760742, -11.367462158203125, -5.510043144226074, -7.271399974822998, -2.327136278152466, 14.60660457611084, 15.18209171295166, 24.59125328063965, -6.018106937408447, 7.182677745819092, 4.416586399078369, 2.353749990463257, 18.07615089416504, 20.144926071166992, 7.744516849517822, 6.586262226104736, 12.887489318847656, -2.458754777908325, 15.980609893798828, 0.011568826623260975, 6.9553141593933105, 27.475204467773438, -7.7557692527771, 0.6639887094497681, 13.970005989074707, 2.5875253677368164, 20.21742820739746, 38.456851959228516, 21.828895568847656, 0.06484982371330261, -10.846400260925293, -0.851777970790863, -12.014119148254395, -10.186613082885742, 11.933685302734375, 10.885398864746094, -23.119585037231445, -6.470555305480957, -13.98868465423584, 5.904905319213867, 1.8485116958618164, 15.197810173034668, 0.8846633434295654, 9.012471199035645, 11.635777473449707, 0.5960103273391724, 14.336804389953613, -30.32548713684082, 11.270177841186523, -18.571897506713867, -4.17587423324585, 4.717939853668213, 3.0268137454986572, -19.388748168945312, 1.4417481422424316, -36.664974212646484, -2.826467514038086, -6.376865386962891, 22.674707412719727, 0.9566268920898438, -5.619325160980225, -10.205902099609375, -12.342498779296875, -0.007519194856286049], "5": [-16.756471633911133, 14.010467529296875, -28.938756942749023, 3.641413688659668, -25.682727813720703, -14.31026840209961, 6.592840671539307, -2.017331838607788, 2.6452066898345947, -8.304569244384766, 14.207415580749512, 22.48192024230957, -20.22180938720703, -7.937211513519287, 1.5273796319961548, -6.07310676574707, -5.038297176361084, -2.7095956802368164, -4.411257266998291, 13.086063385009766, -5.4377570152282715, 2.86434268951416, 7.307520866394043, -2.939979076385498, 8.352543830871582, 14.069097518920898, 15.846604347229004, 28.009214401245117, -14.06371021270752, 8.806060791015625, 3.800807476043701, -7.555930137634277, 0.735014796257019, 6.33546781539917, 0.8337947726249695, 35.952674865722656, 35.425323486328125, -0.9980705380439758, 16.381582260131836, 8.554341316223145, -5.3848772048950195, 3.409008264541626, 13.770431518554688, 8.524580001831055, -1.6888298988342285, 19.187198638916016, 3.7349424362182617, 7.68818998336792, -27.942541122436523, -10.357794761657715, -39.825286865234375, -5.072940826416016, -15.972086906433105, 9.411118507385254, 15.716771125793457, -7.6940388679504395, -5.859097003936768, -19.541872024536133, -27.478435516357422, 16.04619789123535, 10.400973320007324, 1.5293997526168823, -14.246987342834473, -4.1888580322265625], "6": [-20.23504638671875, -0.5069003105163574, -78.58895111083984, -67.30958557128906, -36.71784210205078, -28.285213470458984, -34.491878509521484, -31.9973087310791, -39.03805923461914, -15.04037857055664, -22.5021915435791, -44.24782180786133, -43.148067474365234, -14.166144371032715, -54.04427719116211, -46.323368072509766, -31.006908416748047, -54.59436798095703, -26.362205505371094, -38.8130989074707, -38.560516357421875, -42.718353271484375, -48.05188751220703, -23.159780502319336, -60.234920501708984, -62.19034957885742, -54.51829147338867, -70.0963134765625, -49.398006439208984, -56.59832763671875, -29.783689498901367, -31.208839416503906, -44.193843841552734, -66.26455688476562, -2.233510732650757, -35.2832145690918, -33.51736831665039, -37.276771545410156, -44.456626892089844, -0.14136812090873718, -27.514989852905273, -26.239727020263672, -20.077896118164062, -33.16807174682617, -28.22516441345215, -28.128087997436523, -27.521934509277344, 12.78386116027832, 20.571134567260742, 34.54788589477539, -13.242217063903809, -8.431280136108398, -9.606864929199219, -23.717105865478516, 19.865211486816406, -1.4637130498886108, 2.2433879375457764, 29.698932647705078, 14.090384483337402, -12.134345054626465, 1.9414312839508057, 27.485595703125, 51.2995719909668, 33.329673767089844]}, "mobility_weight": 0.31874111684052153, "pawn_structure_weight": 0.28498290332135423, "king_safety_weight": 0.4507777343292991, "search_depth": 2}'
_BOT_NAME = "Gladiator-CPU streak27 gen=5456 id=981432a4"


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
                print(f"id name {_BOT_NAME}", flush=True)
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
        print(f"info depth {depth}", flush=True)
        print(f"bestmove {move.uci()}", flush=True)


if __name__ == "__main__":
    _Engine().run()

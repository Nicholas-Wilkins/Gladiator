#!/usr/bin/env python3
"""
Self-contained Gladiator UCI engine.

Bot:         e6282976-02f7-4ea9-bdf1-571e1141287c
Generation:  7869
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

_PARAMS_JSON = '{"bot_id": "e6282976-02f7-4ea9-bdf1-571e1141287c", "generation": 7869, "parent_id": null, "piece_values": {"1": 70.8692661706534, "2": 239.83216604407136, "3": 386.2074460284242, "4": 434.7262040262635, "5": 1126.1628887917666, "6": 20000.0}, "pst": {"1": [-2.647932291030884, -14.25496768951416, 0.3353583514690399, 13.897449493408203, 1.5233945846557617, 7.974891185760498, 14.526278495788574, 14.340611457824707, 34.278568267822266, 29.00202178955078, 36.68943405151367, 78.5299072265625, 71.30595397949219, 29.665254592895508, 51.96688461303711, 34.149696350097656, -11.554472923278809, -4.992740631103516, 25.076862335205078, 41.6043701171875, 28.384483337402344, 24.828580856323242, 39.528419494628906, 18.169836044311523, 24.44583511352539, -2.3803460597991943, 25.518922805786133, 41.171504974365234, 49.76010513305664, -4.3230743408203125, 9.739311218261719, 9.835095405578613, 10.72635269165039, -3.2334423065185547, -8.751238822937012, 11.715954780578613, -23.6291561126709, 17.12537384033203, 8.16425609588623, 4.045818328857422, 3.4399538040161133, 0.6035519242286682, -2.943899154663086, 4.86735200881958, -0.6445211172103882, -28.082231521606445, 26.672374725341797, -20.521896362304688, 32.25190734863281, 28.473426818847656, -30.337493896484375, 7.289958953857422, -7.006901264190674, 14.388257026672363, 19.528459548950195, 1.0648151636123657, -24.31712532043457, -16.055028915405273, 0.5343148708343506, 2.872958183288574, 11.9844388961792, -10.490300178527832, -16.36765480041504, -18.652833938598633], "2": [-43.9912223815918, -38.2033805847168, -26.98058319091797, -36.083656311035156, -35.75830078125, -44.3574104309082, -74.61492919921875, -19.366600036621094, -44.756858825683594, -6.551082611083984, -21.865015029907227, 23.824779510498047, 8.376387596130371, -22.87224006652832, -6.043647766113281, -63.952396392822266, -11.393861770629883, -13.391704559326172, 11.363192558288574, -2.682365655899048, 16.46670913696289, 26.315940856933594, -26.063140869140625, -25.22309112548828, -25.239803314208984, -24.011600494384766, 4.31325626373291, 26.38057518005371, 19.18491554260254, -15.729901313781738, -12.701730728149414, -13.297245025634766, -16.65347671508789, 1.5042463541030884, 22.923534393310547, 0.4059930741786957, 21.323442459106445, 17.80044174194336, 8.00219440460205, -42.87931442260742, -51.41744613647461, -10.965044975280762, -0.9265409708023071, 6.104434490203857, 21.11152458190918, 22.066781997680664, 2.3700642585754395, -23.846519470214844, -42.823814392089844, -12.955923080444336, -5.884650230407715, 39.426082611083984, 22.30568504333496, -19.04891586303711, -35.01142883300781, -42.359683990478516, -72.9094009399414, -43.781803131103516, -40.69209289550781, -38.139869689941406, -30.444568634033203, -31.691997528076172, -65.36390686035156, -37.835750579833984], "3": [-11.880847930908203, 6.19926118850708, -17.520362854003906, 1.026875376701355, -3.0108964443206787, -11.678910255432129, 0.6564375758171082, -28.4927978515625, 1.4023607969284058, 13.75308609008789, -17.572330474853516, 6.3108720779418945, 0.33195239305496216, 16.603670120239258, 3.5974485874176025, -16.855327606201172, -21.69692039489746, -6.1625847816467285, 8.532079696655273, 11.55092716217041, -5.648967742919922, -29.965673446655273, -11.624173164367676, -7.666019916534424, -12.234745025634766, -16.293869018554688, 16.542795181274414, 8.410841941833496, 7.69359016418457, 2.8113410472869873, -10.99398422241211, -2.0147697925567627, -25.588212966918945, -12.971735000610352, 9.046097755432129, 19.22258186340332, -7.879640579223633, -8.145164489746094, -5.8393168449401855, -30.605836868286133, -3.624279737472534, 1.8650789260864258, 17.45347023010254, 12.402915000915527, 31.89259910583496, 18.516630172729492, 13.933842658996582, -25.94379997253418, -12.76743221282959, 8.580848693847656, 12.337078094482422, -11.657806396484375, -19.968183517456055, -33.45200729370117, 17.06917953491211, -43.762901306152344, -23.200267791748047, -27.879606246948242, -39.65974044799805, 3.1148173809051514, -12.333454132080078, -2.4044582843780518, -0.5700933933258057, -1.6117262840270996], "4": [-12.310543060302734, -10.147177696228027, 12.21395206451416, -13.18078327178955, -6.03653621673584, -18.207035064697266, -14.660662651062012, 13.723336219787598, 45.043731689453125, 5.686179161071777, 10.282633781433105, 31.68204689025879, 20.301851272583008, -0.08439676463603973, 6.502472877502441, -15.039789199829102, -22.949926376342773, 9.624893188476562, -12.226231575012207, 6.376797676086426, 7.358837604522705, 9.782194137573242, 20.70446014404297, 3.195068597793579, -2.7088873386383057, 20.50693702697754, 19.126401901245117, -2.577354669570923, -5.76411247253418, -5.108839511871338, -2.40553879737854, -2.0490915775299072, 17.533802032470703, 4.282032489776611, 17.21921730041504, -6.145471572875977, 7.295349597930908, 3.98218035697937, 8.925662994384766, 7.481128692626953, -5.747844219207764, 25.385190963745117, 31.29610252380371, -4.861300468444824, 10.970267295837402, 7.501695156097412, 2.581942081451416, -0.644066333770752, -3.882398843765259, 5.314395427703857, -7.476305961608887, 9.996191024780273, 20.731298446655273, -10.640141487121582, 2.5286686420440674, -21.70184326171875, -17.264068603515625, -24.316436767578125, -22.982812881469727, 22.57027244567871, -5.718308925628662, -7.730031490325928, -35.822471618652344, -18.70989227294922], "5": [-11.603570938110352, 15.170660972595215, -8.355216979980469, 4.796626567840576, 1.8019949197769165, -4.231408596038818, -34.97346878051758, -50.796295166015625, -3.1468803882598877, 2.0104198455810547, -8.47304630279541, -5.027618885040283, -2.0734992027282715, -1.1069612503051758, 9.941998481750488, -1.3068991899490356, -11.372345924377441, -24.20634651184082, 7.931802749633789, 18.920230865478516, 1.4753518104553223, -12.924616813659668, -13.227575302124023, 15.354689598083496, -16.936857223510742, 1.2581579685211182, 1.1182479858398438, 6.499134540557861, 9.678383827209473, 15.7493896484375, 5.385560035705566, 4.461300373077393, -17.01445770263672, 4.158846378326416, 10.280648231506348, 14.57938003540039, 0.7225349545478821, 7.5474114418029785, -19.56061363220215, -27.331302642822266, -33.417320251464844, 10.460845947265625, -13.959779739379883, 10.116328239440918, -4.710348129272461, 24.22141456604004, 3.517634153366089, -4.762579917907715, 3.9230873584747314, 22.781461715698242, -3.324852466583252, -23.621023178100586, 8.422358512878418, -1.5701237916946411, -12.48762321472168, -1.8949925899505615, -43.33689880371094, -25.60735321044922, 6.465725421905518, -13.734284400939941, 10.442361831665039, -2.8647775650024414, -14.79050064086914, -39.65639114379883], "6": [-35.79750442504883, -13.064213752746582, -23.231489181518555, -52.572200775146484, -25.39641571044922, -38.66705322265625, -17.941804885864258, -17.912988662719727, -26.10057830810547, -44.211036682128906, -58.0229606628418, -51.44625473022461, -46.25713348388672, -12.093055725097656, -56.024879455566406, -21.600954055786133, -21.015871047973633, -53.92709732055664, -4.81171178817749, -67.17112731933594, -70.60203552246094, -26.458885192871094, -31.41094207763672, -52.61719512939453, 8.185599327087402, -25.556049346923828, -51.243675231933594, -50.58927536010742, -49.111080169677734, -27.995311737060547, -46.43271255493164, -23.395977020263672, -40.86237716674805, -32.71665573120117, -2.3537700176239014, -20.303268432617188, -53.9528923034668, -17.891931533813477, -14.97783088684082, -23.829269409179688, -14.283712387084961, -15.277983665466309, 8.330796241760254, -21.51290512084961, -13.775949478149414, -7.229117393493652, -53.049747467041016, -11.933958053588867, 20.51512336730957, 40.72079086303711, 7.420217514038086, 3.876840114593506, 7.851805686950684, 17.503639221191406, 30.30936622619629, 4.010720252990723, 10.44782829284668, 38.87147521972656, 18.246440887451172, -17.013835906982422, 8.234390258789062, 10.549560546875, 28.523542404174805, 1.2599843740463257]}, "mobility_weight": 0.4580373627104294, "pawn_structure_weight": 0.3068371622593503, "king_safety_weight": 0.18886234563757986, "search_depth": 2}'
_BOT_NAME = "Gladiator-CPU streak27 gen=7869 id=e6282976"


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

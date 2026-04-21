"""BotParams: the full parameter set that defines a bot's behaviour."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from gladiator.bot.tables import BASE_PIECE_VALUES, BASE_PST


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

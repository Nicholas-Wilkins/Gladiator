"""
NNBot and NNBotParams: the neural-network equivalent of the CPU engine's Bot/BotParams.

NNBotParams stores the network's state_dict serialised as base64-encoded bytes
in a JSON envelope so the DB schema is identical to the CPU version.

NNBot loads the weights onto the target device at construction time and keeps
the network in eval() mode for fast inference.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from gladiator_nn.bot.network import ChessNet
from gladiator_nn.training.search import best_move


# ---------------------------------------------------------------------------
# NNBotParams
# ---------------------------------------------------------------------------

@dataclass
class NNBotParams:
    bot_id: str
    generation: int
    parent_id: Optional[str]
    state_dict: dict          # raw torch state_dict (tensors on CPU)
    mutation_scale: float = 0.02
    search_depth: int = 1

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        buf = io.BytesIO()
        torch.save(self.state_dict, buf)
        weights_b64 = base64.b64encode(buf.getvalue()).decode()
        return json.dumps({
            "bot_id": self.bot_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_scale": self.mutation_scale,
            "search_depth": self.search_depth,
            "weights_b64": weights_b64,
        })

    @classmethod
    def from_json(cls, s: str) -> "NNBotParams":
        d = json.loads(s)
        buf = io.BytesIO(base64.b64decode(d["weights_b64"]))
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
        return cls(
            bot_id=d["bot_id"],
            generation=d["generation"],
            parent_id=d.get("parent_id"),
            mutation_scale=float(d["mutation_scale"]),
            search_depth=int(d["search_depth"]),
            state_dict=state_dict,
        )

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int, rng=None) -> "NNBotParams":
        """Random params = PyTorch default initialisation (kaiming uniform)."""
        net = ChessNet()
        state_dict = {k: v.clone().cpu() for k, v in net.state_dict().items()}
        return cls(
            bot_id=str(uuid.uuid4()),
            generation=generation,
            parent_id=None,
            state_dict=state_dict,
        )

    def mutate(self, generation: int, rng=None) -> "NNBotParams":
        """Produce a child by adding Gaussian noise to every weight tensor."""
        gen = torch.Generator()
        if rng is not None:
            seed = int(rng.integers(0, 2**31))
            gen.manual_seed(seed)

        new_state = {}
        for k, v in self.state_dict.items():
            noise = torch.randn(v.shape, generator=gen, dtype=v.dtype) * self.mutation_scale
            new_state[k] = (v + noise).cpu()

        return NNBotParams(
            bot_id=str(uuid.uuid4()),
            generation=generation,
            parent_id=self.bot_id,
            state_dict=new_state,
            mutation_scale=self.mutation_scale,
            search_depth=self.search_depth,
        )


# ---------------------------------------------------------------------------
# NNBot
# ---------------------------------------------------------------------------

class NNBot:
    """Wraps NNBotParams + a live ChessNet on the target device."""

    def __init__(self, params: NNBotParams, device: torch.device) -> None:
        self.params = params
        self.device = device
        self._net = ChessNet()
        self._net.load_state_dict(params.state_dict)
        self._net.to(device)
        self._net.eval()
        self._rng = np.random.default_rng(abs(hash(params.bot_id)) % (2 ** 32))

    # ------------------------------------------------------------------ #
    # Public interface (matches CPU Bot API)                               #
    # ------------------------------------------------------------------ #

    def choose_move(self, board) -> object:
        return best_move(board, self._net, self.device, self.params.search_depth, self._rng)

    @property
    def bot_id(self) -> str:
        return self.params.bot_id

    @property
    def generation(self) -> int:
        return self.params.generation

    def __repr__(self) -> str:
        return (
            f"NNBot(id={self.bot_id[:8]}, gen={self.generation}, "
            f"depth={self.params.search_depth}, mut={self.params.mutation_scale:.3f})"
        )

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int, device: torch.device, rng=None) -> "NNBot":
        return cls(NNBotParams.random(generation, rng), device)

    def mutate(self, generation: int, rng=None) -> "NNBot":
        return NNBot(self.params.mutate(generation, rng), self.device)

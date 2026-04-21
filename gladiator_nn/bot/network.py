"""
Neural network architecture and board encoding for Gladiator-NN.

ChessNet is a small MLP that maps a board position to a scalar in [-1, 1]:
  +1.0  ≈  white is winning
  -1.0  ≈  black is winning

Input: 773 floats
  - 768: 12 piece-type planes × 64 squares (one-hot)
         planes 0-5:  white P/N/B/R/Q/K
         planes 6-11: black P/N/B/R/Q/K
  -   5: auxiliary (turn, four castling rights)
"""

from __future__ import annotations

import chess
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

INPUT_DIM = 773   # 12*64 + 5
HIDDEN_DIMS = (256, 128, 64)


class ChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dims = [INPUT_DIM, *HIDDEN_DIMS, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board as a 773-element float32 tensor (CPU)."""
    planes = torch.zeros(12 * 64, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        plane = _PLANE[(piece.piece_type, piece.color)]
        planes[plane * 64 + sq] = 1.0

    aux = torch.tensor([
        1.0 if board.turn == chess.WHITE else -1.0,
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=torch.float32)

    return torch.cat([planes, aux])

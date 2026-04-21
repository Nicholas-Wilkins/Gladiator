"""
Move selection for NN bots using negamax with alpha-beta pruning.

At depth 1, all candidate positions are batched into a single GPU
forward pass, making full use of the GPU for the most common case.
Deeper searches recurse but still batch leaf evaluations.
"""

from __future__ import annotations

import chess
import torch

from gladiator_nn.bot.network import ChessNet, board_to_tensor

_INF = float("inf")
_CONTEMPT = -0.02   # slight penalty for drawable positions (tanh scale)

_PIECE_VALUE_ROUGH = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100,
}


def best_move(
    board: chess.Board,
    net: ChessNet,
    device: torch.device,
    depth: int = 1,
    rng=None,
) -> chess.Move:
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves")
    if len(legal) == 1:
        return legal[0]

    sign = 1.0 if board.turn == chess.WHITE else -1.0

    if depth == 1:
        # Batch all successor positions in one GPU call
        tensors = []
        for move in legal:
            board.push(move)
            tensors.append(board_to_tensor(board))
            board.pop()

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            raw_scores = net(batch).squeeze(1).cpu().tolist()

        best_mv, best_score = legal[0], -_INF
        for move, raw in zip(legal, raw_scores):
            score = sign * raw
            if rng is not None:
                score += rng.uniform(-1e-4, 1e-4)
            if score > best_score:
                best_score, best_mv = score, move
        return best_mv

    # Deeper search via negamax
    best_mv, best_score, alpha = legal[0], -_INF, -_INF
    for move in _order_moves(board, legal):
        board.push(move)
        score = -_negamax(board, net, device, depth - 1, -_INF, -alpha)
        board.pop()
        if rng is not None:
            score += rng.uniform(-1e-4, 1e-4)
        if score > best_score:
            best_score, best_mv = score, move
        alpha = max(alpha, score)
    return best_mv


def _negamax(
    board: chess.Board,
    net: ChessNet,
    device: torch.device,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    if board.is_repetition() or board.is_fifty_moves():
        return _CONTEMPT
    if board.is_game_over():
        return -_INF if board.is_checkmate() else 0.0
    if depth == 0:
        raw = _eval_single(board, net, device)
        # convert from white-perspective to side-to-move perspective
        return raw if board.turn == chess.WHITE else -raw

    best = -_INF
    for move in _order_moves(board, list(board.legal_moves)):
        board.push(move)
        score = -_negamax(board, net, device, depth - 1, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def _eval_single(board: chess.Board, net: ChessNet, device: torch.device) -> float:
    t = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return net(t).item()


def _order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    def _key(move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v = _PIECE_VALUE_ROUGH.get(victim.piece_type, 0) if victim else 0
            a = _PIECE_VALUE_ROUGH.get(attacker.piece_type, 0) if attacker else 0
            score += 10 * v - a + 1000
        if move.promotion:
            score += 800
        return score
    return sorted(moves, key=_key, reverse=True)

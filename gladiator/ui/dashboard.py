"""
Rich terminal dashboard for monitoring Gladiator training in real time.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Optional, Tuple

import chess
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gladiator.training.match import MatchResult
from gladiator.training.trainer import Mode, TrainerState

console = Console()

_LOCK = Lock()

# Shared state updated by the trainer callback
_state: Optional[TrainerState] = None
_last_match: Optional[MatchResult] = None
_current_challenger_id: Optional[str] = None
_current_match_in_progress: Optional[MatchResult] = None

# Board state updated after every half-move
_current_board: Optional[chess.Board] = None
_current_last_san: Optional[str] = None
_current_white_id: Optional[str] = None
_current_black_id: Optional[str] = None

_PIECE_SYMBOLS = {
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.BLACK): "♚",
}


def update_board(board: chess.Board, last_san: str, white_id: str, black_id: str) -> None:
    """Callback called after every half-move with the current board state."""
    global _current_board, _current_last_san, _current_white_id, _current_black_id
    with _LOCK:
        _current_board = board
        _current_last_san = last_san
        _current_white_id = white_id
        _current_black_id = black_id


def update(state: TrainerState, match: MatchResult, challenger) -> None:
    """Callback passed to Trainer; called after each match."""
    global _state, _last_match, _current_challenger_id
    with _LOCK:
        _state = state
        _last_match = match
        _current_challenger_id = challenger.bot_id if challenger else None


def update_in_progress(state: TrainerState, partial: MatchResult, challenger) -> None:
    """Callback called after each game pair (mid-match)."""
    global _state, _current_match_in_progress, _current_challenger_id
    with _LOCK:
        _state = state
        _current_match_in_progress = partial
        _current_challenger_id = challenger.bot_id if challenger else None


def _mode_badge(mode: Mode) -> Text:
    if mode == Mode.RANDOM:
        return Text("● RANDOM", style="bold yellow")
    return Text("● MUTATION", style="bold cyan")


def _render_header(state: Optional[TrainerState]) -> Panel:
    if state is None:
        return Panel("Waiting for first match...", title="Gladiator", border_style="dim")

    elapsed = time.time() - state.start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="right")

    grid.add_row(
        Text(f"Generation: {state.generation}", style="bold"),
        _mode_badge(state.mode),
        Text(f"Uptime: {h:02d}:{m:02d}:{s:02d}", style="dim"),
    )
    grid.add_row(
        Text(f"Total matches: {state.total_matches}"),
        Text(f"Champion changes: {state.total_champion_changes}", style="bold magenta"),
        Text(""),
    )

    return Panel(grid, title="[bold white]GLADIATOR[/bold white]", border_style="blue")


def _render_champion(state: Optional[TrainerState]) -> Panel:
    if state is None:
        return Panel("No champion yet", title="Champion", border_style="dim")

    champ = state.champion
    p = champ.params
    needs = max(0, 100 - state.consecutive_wins) if state.mode == Mode.RANDOM else 0
    streak_text = (
        f"Streak: {state.consecutive_wins}/100 random wins"
        if state.mode == Mode.RANDOM
        else "Streak: MUTATION mode active"
    )

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("ID:", champ.bot_id[:16] + "…")
    t.add_row("Gen:", str(champ.generation))
    t.add_row("Depth:", str(p.search_depth))
    t.add_row(
        "Piece vals:",
        f"P={p.piece_values[1]:.0f} N={p.piece_values[2]:.0f} B={p.piece_values[3]:.0f} "
        f"R={p.piece_values[4]:.0f} Q={p.piece_values[5]:.0f}",
    )
    t.add_row(
        "Weights:",
        f"mob={p.mobility_weight:.3f}  pawn={p.pawn_structure_weight:.3f}  "
        f"king={p.king_safety_weight:.3f}",
    )
    t.add_row("Streak:", streak_text)
    if state.mode == Mode.RANDOM:
        t.add_row("Until mut.:", f"{needs} more random wins needed")

    return Panel(t, title="[bold green]Reigning Champion[/bold green]", border_style="green")


def _render_current_match(state: Optional[TrainerState], match: Optional[MatchResult], challenger_id: Optional[str]) -> Panel:
    if state is None or challenger_id is None:
        return Panel("No match in progress", title="Current Match", border_style="dim")

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("Challenger:", (challenger_id[:16] + "…") if challenger_id else "?")

    if match is not None:
        pairs_done = len(match.pairs)
        t.add_row("Pairs played:", str(pairs_done))
        t.add_row("Champion score:", f"{match.total_score_a:.1f}")
        t.add_row("Challenger score:", f"{match.total_score_b:.1f}")
        if match.total_score_a == match.total_score_b:
            t.add_row("Status:", Text("TIED — playing more", style="yellow"))
        elif match.total_score_a > match.total_score_b:
            t.add_row("Status:", Text("Champion leading", style="green"))
        else:
            t.add_row("Status:", Text("Challenger leading", style="red"))
    else:
        t.add_row("Status:", "Starting…")

    return Panel(t, title="[bold yellow]Current Match[/bold yellow]", border_style="yellow")


def _render_board(
    board: Optional[chess.Board],
    last_san: Optional[str],
    white_id: Optional[str],
    black_id: Optional[str],
) -> Panel:
    if board is None:
        return Panel("Waiting for first move…", title="Live Board", border_style="dim")

    last_move = board.peek() if board.move_stack else None

    lines = []
    for rank in range(7, -1, -1):
        line = Text()
        line.append(f" {rank + 1} ")
        for file in range(8):
            sq = chess.square(file, rank)
            is_light = (file + rank) % 2 == 1
            highlighted = last_move is not None and sq in (last_move.from_square, last_move.to_square)
            if highlighted:
                bg = "on dark_goldenrod" if is_light else "on yellow4"
            else:
                bg = "on grey82" if is_light else "on grey35"
            piece = board.piece_at(sq)
            if piece:
                symbol = _PIECE_SYMBOLS[(piece.piece_type, piece.color)]
                fg = "bright_white" if piece.color == chess.WHITE else "black"
                line.append(f" {symbol} ", style=f"{fg} {bg}")
            else:
                line.append("   ", style=bg)
        lines.append(line)

    file_labels = Text("    a  b  c  d  e  f  g  h", style="dim")
    turn = "White" if board.turn == chess.WHITE else "Black"
    move_num = board.fullmove_number

    info = Text()
    info.append(f" Move {move_num}  ", style="dim")
    info.append(f"Last: {last_san or '—'}  ", style="bold")
    info.append(f"{turn} to move", style="green" if board.turn == chess.WHITE else "blue")

    w_label = (white_id[:14] + "…") if white_id and len(white_id) > 15 else (white_id or "?")
    b_label = (black_id[:14] + "…") if black_id and len(black_id) > 15 else (black_id or "?")
    players = Text()
    players.append(f" ♙ {w_label}", style="bright_white")
    players.append("  vs  ", style="dim")
    players.append(f"♟ {b_label}", style="bold")

    content = Group(players, *lines, file_labels, info)
    return Panel(content, title="[bold magenta]Live Board[/bold magenta]", border_style="magenta")


def _render_history(recent: list[dict]) -> Panel:
    if not recent:
        return Panel("No matches yet", title="Recent Matches", border_style="dim")

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("Gen", style="dim", justify="right")
    table.add_column("Mode", justify="center")
    table.add_column("Result", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Pairs", justify="right")

    for row in recent[:15]:
        result_text = Text("KEPT", style="green") if row["champion_won"] else Text("DETHRONED", style="red bold")
        mode_text = "RNG" if row["mode"] == "RANDOM" else "MUT"
        score = f"{row['score_champion']:.1f}–{row['score_challenger']:.1f}"
        table.add_row(
            str(row["generation"]),
            mode_text,
            result_text,
            score,
            str(row["pairs_played"]),
        )

    return Panel(table, title="[bold]Match History[/bold]", border_style="blue")


class Dashboard:
    """Context manager that runs the Rich Live display in the background."""

    def __init__(self, db) -> None:
        self.db = db
        self._live: Optional[Live] = None

    def __enter__(self) -> "Dashboard":
        self._live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=2,
            screen=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)

    def refresh(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Layout:
        with _LOCK:
            state = _state
            in_prog = _current_match_in_progress
            challenger_id = _current_challenger_id
            board = _current_board
            last_san = _current_last_san
            white_id = _current_white_id
            black_id = _current_black_id

        recent = self.db.recent_matches(15)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="history", size=17),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="board", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="champion"),
            Layout(name="current"),
        )

        layout["header"].update(_render_header(state))
        layout["champion"].update(_render_champion(state))
        layout["current"].update(_render_current_match(state, in_prog, challenger_id))
        layout["board"].update(_render_board(board, last_san, white_id, black_id))
        layout["history"].update(_render_history(recent))

        return layout

    def make_progress_cb(self):
        """Returns (post_match_cb, mid_match_cb) for Trainer wiring."""
        def post_match(state: TrainerState, match: MatchResult, challenger) -> None:
            update(state, match, challenger)
            global _current_match_in_progress
            with _LOCK:
                _current_match_in_progress = match
            self.refresh()

        def mid_match(partial: MatchResult) -> None:
            global _current_match_in_progress
            with _LOCK:
                _current_match_in_progress = partial
            self.refresh()

        return post_match, mid_match

    def make_move_cb(self):
        """Returns a per-half-move callback that updates the live board panel."""
        def on_move(board: chess.Board, san: str, white_id: str, black_id: str) -> None:
            update_board(board, san, white_id, black_id)
            self.refresh()

        return on_move

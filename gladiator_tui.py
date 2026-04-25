"""
Shared Textual TUI for Gladiator parallel training monitors.

Provides GladiatorParallelApp, TUIConfig, and WorkerState — imported by all
four *_parallel.py launchers so display logic lives in one place.
"""
from __future__ import annotations

import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


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


@dataclass
class TUIConfig:
    title: str
    header_color: str
    worker_label: str
    win_threshold: int


@dataclass
class WorkerState:
    alive: bool = False
    initialized: bool = False
    mode: str = "RANDOM"
    consecutive_wins: int = 0
    total_matches: int = 0
    total_champion_changes: int = 0
    bot_id: str = ""
    board: Optional[chess.Board] = None
    last_san: str = ""
    white_id: str = ""
    black_id: str = ""


# ---------------------------------------------------------------------------
# Rich renderers (same logic as the old per-file functions)
# ---------------------------------------------------------------------------

def _streak_bar(wins: int, threshold: int, width: int = 18) -> Text:
    filled = min(wins, threshold)
    n_filled = int(filled / threshold * width)
    n_empty = width - n_filled
    pct = filled / threshold
    color = "green" if pct >= 0.75 else ("yellow" if pct >= 0.4 else "red")
    bar = Text()
    bar.append("[", style="dim")
    bar.append("█" * n_filled, style=color)
    bar.append("░" * n_empty, style="dim")
    bar.append("]", style="dim")
    return bar


def _render_board_inline(
    board: Optional[chess.Board],
    last_san: str,
    white_id: str,
    black_id: str,
) -> Group:
    if board is None:
        return Group(Text("  No games completed yet…", style="dim italic"))

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
    w_label = (white_id[:12] + "…") if white_id and len(white_id) > 13 else (white_id or "?")
    b_label = (black_id[:12] + "…") if black_id and len(black_id) > 13 else (black_id or "?")
    players = Text()
    players.append(f" ♙ {w_label}", style="bright_white")
    players.append("  vs  ", style="dim")
    players.append(f"♟ {b_label}", style="bold")
    turn = "White" if board.turn == chess.WHITE else "Black"
    info = Text()
    info.append(f" Mv {board.fullmove_number}  ", style="dim")
    info.append(f"Last: {last_san or '—'}  ", style="bold")
    info.append(f"{turn} to move", style="green" if board.turn == chess.WHITE else "blue")
    return Group(players, *lines, file_labels, info)


def _worker_panel(cfg: TUIConfig, i: int, pid: int, snap: WorkerState) -> Panel:
    _MUTATION = "MUTATION"
    if not snap.alive:
        status_text = Text("DEAD", style="bold red")
        border = "red"
    elif snap.mode == _MUTATION:
        status_text = Text("WINNER", style="bold bright_green")
        border = "bright_green"
    else:
        status_text = Text("RUNNING", style="bold green")
        border = cfg.header_color

    stats = Table.grid(padding=(0, 1))
    stats.add_column(style="dim", min_width=9)
    stats.add_column(min_width=20)
    stats.add_row("PID:", str(pid))
    stats.add_row("Status:", status_text)

    if snap.initialized:
        mode_text = (
            Text("● MUTATION", style="bold cyan")
            if snap.mode == _MUTATION
            else Text("● RANDOM", style="bold yellow")
        )
        stats.add_row("Mode:", mode_text)
        bar = _streak_bar(snap.consecutive_wins, cfg.win_threshold)
        wins_text = Text()
        wins_text.append_text(bar)
        wins_text.append(f"  {snap.consecutive_wins}/{cfg.win_threshold}", style="dim")
        stats.add_row("Streak:", wins_text)
        stats.add_row("Matches:", str(snap.total_matches))
        stats.add_row("Changes:", str(snap.total_champion_changes))
        short_id = (snap.bot_id[:16] + "…") if len(snap.bot_id) > 16 else snap.bot_id
        stats.add_row("Champion:", short_id)
    else:
        stats.add_row("", Text("initializing…", style="dim italic"))

    board_group = _render_board_inline(snap.board, snap.last_san, snap.white_id, snap.black_id)
    title_style = "bold bright_green" if snap.mode == _MUTATION and snap.alive else "bold"
    content = Group(stats, Rule(style="dim"), board_group)
    return Panel(
        content,
        title=f"[{title_style}]{cfg.worker_label} {i}[/{title_style}]",
        border_style=border,
    )


def _header_panel(
    cfg: TUIConfig,
    n: int,
    start_time: float,
    snapshots: list[WorkerState],
    original_total: int,
    baselines: list[int],
    inactive_extra: int,
    all_time_best_streak: int,
) -> Panel:
    elapsed = int(time.time() - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    active_new = sum(
        max(0, snap.total_matches - base)
        for snap, base in zip(snapshots, baselines)
    )
    grand_total = original_total + inactive_extra + active_new
    live_best = max((snap.consecutive_wins for snap in snapshots if snap.initialized), default=0)
    best_streak = max(all_time_best_streak, live_best)

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="right")
    grid.add_row(
        Text(f"{cfg.title} — Parallel RANDOM Explorer", style="bold white"),
        Text(f"{n} active workers", style="bold yellow"),
        Text(f"Total matches (all-time): {grand_total:,}", style=f"bold {cfg.header_color}"),
        Text(f"Elapsed: {h:02d}:{m:02d}:{s:02d}", style="dim"),
    )
    grid.add_row(
        Text(""),
        Text(""),
        Text(f"Longest win streak so far: {best_streak}", style="bold magenta"),
        Text("Press Q or Ctrl+C to stop", style="dim"),
    )
    return Panel(grid, border_style=cfg.header_color, padding=(0, 1))


def _events_panel(events: deque) -> Panel:
    content = (
        Group(*[Text.from_markup(e) for e in events])
        if events
        else Text("No events yet…", style="dim italic")
    )
    return Panel(content, title="[bold]Events[/bold]", border_style="dim", box=box.SIMPLE_HEAD)


# ---------------------------------------------------------------------------
# Textual app
# ---------------------------------------------------------------------------

class GladiatorParallelApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #header {
        height: auto;
    }
    #scroll-area {
        height: 1fr;
    }
    #workers {
        height: auto;
    }
    #events {
        height: auto;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        cfg: TUIConfig,
        n: int,
        procs: list[subprocess.Popen],
        snapshots: list[WorkerState],
        events: deque,
        poll_fn: Callable,
        poll_interval: float,
        start_time: float,
        original_total: int,
        baselines: list[int],
        inactive_extra: int,
        best_streak: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = n
        self.procs = procs
        self.snapshots = snapshots
        self.events = events
        self.poll_fn = poll_fn
        self.poll_interval = poll_interval
        self.start_time = start_time
        self.original_total = original_total
        self.baselines = baselines
        self.inactive_extra = inactive_extra
        self.all_time_best_streak = best_streak

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with VerticalScroll(id="scroll-area"):
            yield Static("", id="workers")
        yield Static("", id="events")

    def on_mount(self) -> None:
        self._refresh_display()
        self.set_interval(self.poll_interval, self._tick)

    def _tick(self) -> None:
        self.poll_fn()
        for snap in self.snapshots:
            if snap.initialized and snap.consecutive_wins > self.all_time_best_streak:
                self.all_time_best_streak = snap.consecutive_wins
        self._refresh_display()

    def _refresh_display(self) -> None:
        width = self.size.width or 80
        workers_per_row = max(1, width // 38)

        header = _header_panel(
            self.cfg, self.n, self.start_time, self.snapshots,
            self.original_total, self.baselines, self.inactive_extra,
            self.all_time_best_streak,
        )

        row_renderables = []
        for row_start in range(0, self.n, workers_per_row):
            row_end = min(row_start + workers_per_row, self.n)
            panels = [
                _worker_panel(self.cfg, i, self.procs[i].pid, self.snapshots[i])
                for i in range(row_start, row_end)
            ]
            row_renderables.append(Columns(panels, equal=True, expand=True))

        self.query_one("#header", Static).update(header)
        self.query_one("#workers", Static).update(Group(*row_renderables))
        self.query_one("#events", Static).update(_events_panel(self.events))

    def action_quit(self) -> None:
        self.events.appendleft(
            f"[{time.strftime('%H:%M:%S')}]  [yellow]Stopped by user[/yellow]"
        )
        self.exit()

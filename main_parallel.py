"""
Parallel RANDOM-phase explorer for Gladiator.

Copies the current DB to N worker DBs, launches N headless training
workers with independent RNG seeds, and monitors them via a Rich live
dashboard until one reaches MUTATION mode (100 consecutive RANDOM wins).
The winning champion is then promoted back into the main DB and all
workers are stopped.

Usage
-----
    python main_parallel.py                   # 4 workers, source gladiator.db
    python main_parallel.py --workers 5       # 5 workers
    python main_parallel.py --db other.db     # different source DB
    python main_parallel.py --seed 42         # base RNG seed (worker i gets seed+i)
    python main_parallel.py --poll 10         # poll every 10 s instead of 5

After this script exits successfully, run:
    python main.py
to resume training in MUTATION mode.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


_POLL_DEFAULT = 5.0
_MUTATION = "MUTATION"
_RANDOM_WIN_THRESHOLD = 100
_MAX_EVENTS = 12

_SCHEMA = """
CREATE TABLE IF NOT EXISTS champion (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    bot_id              TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    generation          INTEGER NOT NULL,
    consecutive_wins    INTEGER NOT NULL DEFAULT 0,
    mode                TEXT NOT NULL DEFAULT 'RANDOM',
    total_matches       INTEGER NOT NULL DEFAULT 0,
    total_champion_changes INTEGER NOT NULL DEFAULT 0,
    saved_at            REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS champion_log (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id              TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    generation          INTEGER NOT NULL,
    crowned_at          REAL NOT NULL,
    total_matches_at_crowning INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS matches (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    champion_id         TEXT NOT NULL,
    challenger_id       TEXT NOT NULL,
    champion_won        INTEGER NOT NULL,
    score_champion      REAL NOT NULL,
    score_challenger    REAL NOT NULL,
    pairs_played        INTEGER NOT NULL,
    total_games         INTEGER NOT NULL,
    generation          INTEGER NOT NULL,
    mode                TEXT NOT NULL,
    played_at           REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_rowid         INTEGER NOT NULL REFERENCES matches(rowid),
    pair_number         INTEGER NOT NULL,
    game_in_pair        INTEGER NOT NULL,
    white_id            TEXT NOT NULL,
    black_id            TEXT NOT NULL,
    result              TEXT NOT NULL,
    termination         TEXT NOT NULL,
    halfmoves           INTEGER NOT NULL,
    duration_s          REAL NOT NULL,
    pgn                 TEXT NOT NULL
);
"""


def _init_db(path: Path) -> None:
    """Create a fresh empty DB with the training schema."""
    con = sqlite3.connect(str(path))
    con.executescript(_SCHEMA)
    con.commit()
    con.close()

console = Console()

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel Gladiator RANDOM-phase coordinator")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default 4)")
    p.add_argument("--db", default="gladiator.db", help="Source SQLite DB (default gladiator.db)")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed; worker i gets seed+i")
    p.add_argument("--poll", type=float, default=_POLL_DEFAULT, help="Poll interval in seconds (default 5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _worker_db(i: int) -> Path:
    return Path(f"gladiator_worker_{i}.db")


def _worker_log(i: int) -> Path:
    return Path(f"gladiator_worker_{i}.log")


def _read_champion_row(db_path: Path) -> dict | None:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(str(db_path), timeout=2)
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM champion WHERE id = 1").fetchone()
        con.close()
        return dict(row) if row else None
    except Exception:
        return None


def _query_max_streak(db_path: Path) -> int:
    """
    Compute the longest consecutive-win streak from champion_log.

    Avoids the champion_won column in matches, which was written incorrectly
    by an earlier bug.  Instead derives each champion's streak purely from
    total_matches_at_crowning gaps:

        streak(i) = crowning_matches[i+1] - crowning_matches[i] - 1

    The -1 accounts for the final loss that ended the reign.
    The current champion's streak is read from consecutive_wins directly.
    """
    if not db_path.exists():
        return 0
    try:
        con = sqlite3.connect(str(db_path), timeout=2)
        log = con.execute(
            "SELECT total_matches_at_crowning FROM champion_log ORDER BY rowid"
        ).fetchall()
        champ = con.execute(
            "SELECT consecutive_wins FROM champion WHERE id = 1"
        ).fetchone()
        con.close()

        if not log:
            return 0

        crownings = [row[0] for row in log]
        best = 0
        for i in range(len(crownings) - 1):
            streak = max(0, crownings[i + 1] - crownings[i] - 1)
            if streak > best:
                best = streak

        # Current reigning champion's live streak
        if champ and champ[0] > best:
            best = champ[0]

        return best
    except Exception:
        return 0


def _read_latest_board(
    db_path: Path,
) -> tuple[chess.Board, str, str, str] | None:
    """Return (board, last_san, white_id, black_id) from the last completed game, or None."""
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(str(db_path), timeout=2)
        row = con.execute(
            "SELECT pgn, white_id, black_id FROM games ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        con.close()
        if row is None:
            return None
        game = chess.pgn.read_game(io.StringIO(row[0]))
        if game is None:
            return None
        node = game.end()
        board = node.board()
        last_san = node.san() if node.move else ""
        return board, last_san, row[1], row[2]
    except Exception:
        return None


def _promote_winner(winner_db: Path, main_db: Path) -> None:
    """Upsert the winning champion + new champion_log entries into main_db."""
    wcon = sqlite3.connect(str(winner_db))
    wcon.row_factory = sqlite3.Row
    mcon = sqlite3.connect(str(main_db))
    mcon.row_factory = sqlite3.Row

    champ = dict(wcon.execute("SELECT * FROM champion WHERE id = 1").fetchone())
    mcon.execute(
        """
        INSERT INTO champion
            (id, bot_id, params_json, generation, consecutive_wins, mode,
             total_matches, total_champion_changes, saved_at)
        VALUES (1, :bot_id, :params_json, :generation, :consecutive_wins, :mode,
                :total_matches, :total_champion_changes, :saved_at)
        ON CONFLICT(id) DO UPDATE SET
            bot_id                  = excluded.bot_id,
            params_json             = excluded.params_json,
            generation              = excluded.generation,
            consecutive_wins        = excluded.consecutive_wins,
            mode                    = excluded.mode,
            total_matches           = excluded.total_matches,
            total_champion_changes  = excluded.total_champion_changes,
            saved_at                = excluded.saved_at
        """,
        champ,
    )

    for row in wcon.execute("SELECT * FROM champion_log ORDER BY rowid").fetchall():
        exists = mcon.execute(
            "SELECT 1 FROM champion_log WHERE bot_id = ?", (row["bot_id"],)
        ).fetchone()
        if exists is None:
            mcon.execute(
                """
                INSERT INTO champion_log
                    (bot_id, params_json, generation, crowned_at, total_matches_at_crowning)
                VALUES (?, '{}', ?, ?, ?)
                """,
                (row["bot_id"], row["generation"],
                 row["crowned_at"], row["total_matches_at_crowning"]),
            )

    mcon.commit()
    mcon.close()
    wcon.close()


def _stop_all(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.time() + 30
    for i, proc in enumerate(procs):
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def _best_worker_idx(snapshots: list[WorkerState]) -> int | None:
    """Return the index of the best worker to promote.

    Prefers workers in MUTATION mode; tiebreaks by consecutive_wins.
    """
    best_idx = None
    best_key: tuple[int, int] = (-1, -1)
    for i, snap in enumerate(snapshots):
        if not snap.initialized:
            continue
        key = (1 if snap.mode == _MUTATION else 0, snap.consecutive_wins)
        if key > best_key:
            best_key = key
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Worker state tracking
# ---------------------------------------------------------------------------

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


def _poll_workers(
    procs: list[subprocess.Popen],
    worker_dbs: list[Path],
    snapshots: list[WorkerState],
    events: deque,
) -> int | None:
    """
    Refresh snapshots from DBs and emit events for notable changes.
    Returns the index of the first worker in MUTATION mode, or None.
    """
    now = time.strftime("%H:%M:%S")
    winner_idx: int | None = None

    for i, (proc, wdb) in enumerate(zip(procs, worker_dbs)):
        prev = snapshots[i]
        snap = WorkerState()

        snap.alive = proc.poll() is None
        if not snap.alive and prev.alive:
            events.appendleft(f"[{now}]  [red]Worker {i} process exited[/red]")

        # Carry forward board state between polls (boards only change when games finish)
        snap.board = prev.board
        snap.last_san = prev.last_san
        snap.white_id = prev.white_id
        snap.black_id = prev.black_id

        row = _read_champion_row(wdb)
        if row:
            snap.initialized = True
            snap.mode = row["mode"]
            snap.consecutive_wins = row["consecutive_wins"]
            snap.total_matches = row["total_matches"]
            snap.total_champion_changes = row["total_champion_changes"]
            snap.bot_id = row["bot_id"]

            changes = snap.total_champion_changes - prev.total_champion_changes
            if prev.initialized and changes > 0:
                events.appendleft(
                    f"[{now}]  Worker {i}: [magenta]{changes} new champion(s) crowned[/magenta]"
                )

            if prev.initialized and prev.mode == "RANDOM" and snap.mode == _MUTATION:
                events.appendleft(
                    f"[{now}]  Worker {i}: [bold green]★ MUTATION MODE REACHED![/bold green]"
                )
                winner_idx = i

        board_result = _read_latest_board(wdb)
        if board_result is not None:
            snap.board, snap.last_san, snap.white_id, snap.black_id = board_result

        snapshots[i] = snap

    return winner_idx


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

def _streak_bar(wins: int, width: int = 18) -> Text:
    filled = min(wins, _RANDOM_WIN_THRESHOLD)
    n_filled = int(filled / _RANDOM_WIN_THRESHOLD * width)
    n_empty = width - n_filled

    pct = filled / _RANDOM_WIN_THRESHOLD
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
    """Render a chess board as a plain Rich Group (no wrapping Panel)."""
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


def _worker_panel(i: int, pid: int, snap: WorkerState) -> Panel:
    if not snap.alive:
        status_text = Text("DEAD", style="bold red")
        border = "red"
    elif snap.mode == _MUTATION:
        status_text = Text("WINNER", style="bold bright_green")
        border = "bright_green"
    else:
        status_text = Text("RUNNING", style="bold green")
        border = "blue"

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

        bar = _streak_bar(snap.consecutive_wins)
        wins_text = Text()
        wins_text.append_text(bar)
        wins_text.append(f"  {snap.consecutive_wins}/{_RANDOM_WIN_THRESHOLD}", style="dim")
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
    return Panel(content, title=f"[{title_style}]Worker {i}[/{title_style}]", border_style=border)


def _header_panel(
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

    # Active workers' new matches + pre-existing inactive worker matches
    active_new = sum(
        max(0, snap.total_matches - base)
        for snap, base in zip(snapshots, baselines)
    )
    grand_total = original_total + inactive_extra + active_new

    # Best streak: historical max across all DBs OR current live max
    live_best = max((s.consecutive_wins for s in snapshots if s.initialized), default=0)
    best_streak = max(all_time_best_streak, live_best)

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="right")

    grid.add_row(
        Text("GLADIATOR — Parallel RANDOM Explorer", style="bold white"),
        Text(f"{n} active workers", style="bold yellow"),
        Text(f"Total matches (all-time): {grand_total:,}", style="bold cyan"),
        Text(f"Elapsed: {h:02d}:{m:02d}:{s:02d}", style="dim"),
    )
    grid.add_row(
        Text(""),
        Text(""),
        Text(f"Longest win streak so far: {best_streak}", style="bold magenta"),
        Text(""),
    )
    return Panel(grid, border_style="blue", padding=(0, 1))


def _events_panel(events: deque) -> Panel:
    content = (
        Group(*[Text.from_markup(e) for e in events])
        if events
        else Text("No events yet…", style="dim italic")
    )
    return Panel(content, title="[bold]Events[/bold]", border_style="dim", box=box.SIMPLE_HEAD)


def _build_display(
    n: int,
    start_time: float,
    procs: list[subprocess.Popen],
    snapshots: list[WorkerState],
    events: deque,
    original_total: int,
    baselines: list[int],
    inactive_extra: int,
    all_time_best_streak: int,
) -> Group:
    workers_per_row = max(1, console.size.width // 38)
    header = _header_panel(
        n, start_time, snapshots, original_total, baselines,
        inactive_extra, all_time_best_streak,
    )
    row_renderables = []
    for row_start in range(0, n, workers_per_row):
        row_indices = range(row_start, min(row_start + workers_per_row, n))
        panels = [_worker_panel(i, procs[i].pid, snapshots[i]) for i in row_indices]
        row_renderables.append(Columns(panels, equal=True, expand=True))
    ev = _events_panel(events)
    return Group(header, *row_renderables, ev)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    main_db = Path(args.db)

    if not main_db.exists():
        console.print(f"Source DB '{main_db}' not found — creating new database.")
        _init_db(main_db)

    n = args.workers
    python = sys.executable

    # --- Persist original_total across restarts ---
    # original_total = matches in gladiator.db before parallel training ever started.
    # Stored in a sidecar file so restarts don't double-count pre-parallel history.
    meta_path = Path("gladiator_parallel_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        original_total: int = meta["original_total"]
        console.print(f"Resuming parallel session (pre-parallel baseline: [cyan]{original_total:,}[/cyan] matches).")
    else:
        source_row = _read_champion_row(main_db)
        original_total = source_row["total_matches"] if source_row else 0
        meta_path.write_text(json.dumps({"original_total": original_total}))
        console.print(f"New parallel session — pre-parallel baseline: [cyan]{original_total:,}[/cyan] matches.")

    # --- Set up worker DBs (copy only if they don't already exist) ---
    worker_dbs: list[Path] = []
    for i in range(n):
        wdb = _worker_db(i)
        if wdb.exists():
            console.print(f"  Resuming worker {i} from existing [cyan]{wdb}[/cyan]")
        else:
            shutil.copy2(main_db, wdb)
            console.print(f"  Created worker {i}: copied {main_db} → [cyan]{wdb}[/cyan]")
        worker_dbs.append(wdb)

    # baseline per worker = original_total (all workers started from the same source snapshot)
    baselines = [original_total] * n

    # --- Count matches from inactive worker DBs (workers not in this session) ---
    # and compute the historical best streak across ALL worker DBs ever created.
    active_db_paths = set(worker_dbs)
    inactive_extra = 0
    historical_max_streak = _query_max_streak(main_db)
    for wdb in sorted(Path(".").glob("gladiator_worker_*.db")):
        historical_max_streak = max(historical_max_streak, _query_max_streak(wdb))
        if wdb not in active_db_paths:
            row = _read_champion_row(wdb)
            if row:
                inactive_extra += max(0, row["total_matches"] - original_total)
    if inactive_extra:
        console.print(
            f"  Inactive worker DBs contribute [cyan]{inactive_extra:,}[/cyan] additional matches to the total."
        )

    console.print(f"\nLaunching [bold]{n}[/bold] workers…")
    procs: list[subprocess.Popen] = []
    log_files = []
    for i, wdb in enumerate(worker_dbs):
        cmd = [python, "main.py", "--no-ui", "--db", str(wdb), "--seed", str(args.seed + i)]
        lf = _worker_log(i).open("w")
        log_files.append(lf)
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        procs.append(proc)
        console.print(f"  Worker {i}: PID [bold]{proc.pid}[/bold] | log [dim]{_worker_log(i)}[/dim]")

    snapshots: list[WorkerState] = [WorkerState() for _ in range(n)]
    events: deque = deque(maxlen=_MAX_EVENTS)
    start_time = time.time()
    all_time_best_streak = historical_max_streak

    def _display():
        return _build_display(
            n, start_time, procs, snapshots, events,
            original_total, baselines, inactive_extra, all_time_best_streak,
        )

    with Live(_display(), console=console, refresh_per_second=2, screen=True) as live:
        try:
            while True:
                time.sleep(args.poll)
                _poll_workers(procs, worker_dbs, snapshots, events)
                for snap in snapshots:
                    if snap.initialized and snap.consecutive_wins > all_time_best_streak:
                        all_time_best_streak = snap.consecutive_wins
                live.update(_display())

        except KeyboardInterrupt:
            events.appendleft(
                f"[{time.strftime('%H:%M:%S')}]  [yellow]Interrupted by user[/yellow]"
            )
            live.update(_display())

    console.print("\nStopping all workers…")
    _stop_all(procs)
    for lf in log_files:
        lf.close()
    console.print("All workers stopped.")

    winner_idx = _best_worker_idx(snapshots)
    if winner_idx is not None:
        winner_db = worker_dbs[winner_idx]
        console.print(f"\nPromoting champion from [cyan]{winner_db}[/cyan] → [cyan]{main_db}[/cyan]…")
        _promote_winner(winner_db, main_db)
        console.print("\n[bold green]Done![/bold green]  Champion promoted. Re-run to continue training.")
    else:
        console.print("\nNo initialized workers. Worker DBs preserved for inspection.")
        console.print("Re-run [bold]`python main_parallel.py`[/bold] to resume.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_monitor())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Gladiator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINES = {
    "cpu":     {"main": "main.py",          "default_db": "gladiator.db"},
    "nn":      {"main": "main_nn.py",       "default_db": "gladiator_nn.db"},
    "mini":    {"main": "main_mini.py",     "default_db": "gladiator_mini.db"},
    "nn-mini": {"main": "main_nn_mini.py",  "default_db": "gladiator_nn_mini.db"},
}

sessions: dict[str, dict] = {}
sessions_lock = asyncio.Lock()
ws_clients: set[WebSocket] = set()
ws_lock = asyncio.Lock()


_PIECE_SYMBOLS = {
    (chess.PAWN, chess.WHITE): "♟", (chess.KNIGHT, chess.WHITE): "♞",
    (chess.BISHOP, chess.WHITE): "♝", (chess.ROOK, chess.WHITE): "♜",
    (chess.QUEEN, chess.WHITE): "♛", (chess.KING, chess.WHITE): "♚",
    (chess.PAWN, chess.BLACK): "♟", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝", (chess.ROOK, chess.BLACK): "♜",
    (chess.QUEEN, chess.BLACK): "♛", (chess.KING, chess.BLACK): "♚",
}

def _board_to_grid(board: chess.Board | None) -> dict | None:
    if board is None:
        return None
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                cells.append(_PIECE_SYMBOLS[(piece.piece_type, piece.color)])
            else:
                cells.append(None)
        rows.append(cells)
    return {"rows": rows, "turn": "white" if board.turn == chess.WHITE else "black", "fen": board.fen()}

def _read_latest_board(db_path: Path) -> dict | None:
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
        grid = _board_to_grid(board)
        if grid is None:
            return None
        grid["last_san"] = last_san
        grid["white_id"] = row[1]
        grid["black_id"] = row[2]
        return grid
    except Exception:
        return None

def _resolve_db(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else SCRIPT_DIR / p

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


def _read_lineage(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), timeout=2)
        rows = con.execute(
            "SELECT bot_id, generation, crowned_at, total_matches_at_crowning "
            "FROM champion_log ORDER BY rowid"
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _read_matches(db_path: Path, limit: int = 50) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path), timeout=2)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT generation, mode, champion_won, score_champion, score_challenger, "
            "pairs_played, total_games, played_at "
            "FROM matches ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _session_info(sid: str, s: dict) -> dict:
    proc = s["process"]
    alive = proc.poll() is None
    status = "running" if alive else "stopped"
    dbp = _resolve_db(s["db_path"])
    data = _read_champion_row(dbp) or {}
    board = _read_latest_board(dbp)
    return {
        "id": sid,
        "engine": s["engine"],
        "db_path": str(s["db_path"]),
        "pid": proc.pid,
        "status": status,
        "created_at": s["created_at"],
        "seed": s.get("seed"),
        "max_games": s.get("max_games"),
        "worker_id": s.get("worker_id"),
        "champion": data,
        "board": board,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


@app.get("/api/engines")
async def list_engines():
    result = []
    for key, cfg in ENGINES.items():
        dbp = _resolve_db(cfg["default_db"])
        champion = _read_champion_row(dbp)
        result.append({
            "id": key,
            "main": cfg["main"],
            "default_db": cfg["default_db"],
            "db_exists": dbp.exists(),
            "champion": champion,
        })
    return {"engines": result}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@app.post("/api/sessions/create")
async def create_session(body: dict):
    engine = body.get("engine", "cpu")
    num_workers = body.get("num_workers", 1)
    seed = body.get("seed")
    max_games = body.get("max_games")

    if engine not in ENGINES:
        return {"error": f"Unknown engine: {engine}"}

    cfg = ENGINES[engine]
    python = sys.executable
    created = []

    main_db_path = _resolve_db(cfg["default_db"])

    # Atomically assign worker indices that aren't in use by active sessions
    async with sessions_lock:
        in_use = {s["worker_id"] for s in sessions.values()
                  if s["engine"] == engine and s.get("worker_id") is not None}
        for i in range(num_workers):
            idx = 0
            while idx in in_use:
                idx += 1
            in_use.add(idx)

            worker_db_name = f"gladiator_{engine}_worker_{idx}.db"
            worker_db_path = _resolve_db(worker_db_name)
            if not worker_db_path.exists() and main_db_path.exists():
                shutil.copy2(str(main_db_path), str(worker_db_path))
            db_path = str(worker_db_path)

            worker_seed = (seed or 0) + i if seed is not None else None

            cmd = [python, str(SCRIPT_DIR / cfg["main"]), "--no-ui", "--db", db_path]
            if worker_seed is not None:
                cmd.extend(["--seed", str(worker_seed)])
            if max_games is not None:
                cmd.extend(["--max-games", str(max_games)])

            proc = subprocess.Popen(cmd)
            sid = uuid.uuid4().hex[:8]

            sessions[sid] = {
                "engine": engine,
                "db_path": db_path,
                "process": proc,
                "created_at": time.time(),
                "seed": worker_seed,
                "max_games": max_games,
                "worker_id": idx,
            }

            created.append({"session_id": sid, "pid": proc.pid, "db": worker_db_name, "worker": idx})

    await _broadcast_refresh()

    if num_workers == 1:
        return created[0]
    return {"sessions": created, "num_workers": num_workers}


@app.post("/api/sessions/{sid}/stop")
async def stop_session(sid: str):
    async with sessions_lock:
        s = sessions.get(sid)
    if not s:
        return {"error": "Session not found"}
    proc = s["process"]
    if proc.poll() is not None:
        return {"status": "already_stopped"}
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass
    await _broadcast_refresh()
    return {"status": "stopping"}


@app.get("/api/sessions")
async def list_sessions():
    async with sessions_lock:
        result = [_session_info(sid, s) for sid, s in sessions.items()]
    return {"sessions": result}


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    async with sessions_lock:
        s = sessions.get(sid)
    if not s:
        return {"error": "Session not found"}
    return _session_info(sid, s)


# ---------------------------------------------------------------------------
# Databases
# ---------------------------------------------------------------------------


@app.get("/api/dbs")
async def list_dbs():
    files = sorted(SCRIPT_DIR.glob("*.db"))
    result = []
    for f in files:
        champion = _read_champion_row(f)
        lineage = _read_lineage(f)
        result.append({
            "name": f.name,
            "size_kb": f.stat().st_size // 1024,
            "has_champion": champion is not None,
            "mode": champion.get("mode") if champion else None,
            "total_matches": champion.get("total_matches", 0) if champion else 0,
            "num_crownings": len(lineage),
        })
    return {"dbs": result}


@app.get("/api/dbs/{name}/champion")
async def get_champion(name: str):
    path = _resolve_db(name)
    if not path.exists():
        return {"error": "DB not found"}
    champion = _read_champion_row(path)
    if not champion:
        return {"error": "No champion found"}
    return {"champion": champion}


@app.get("/api/dbs/{name}/lineage")
async def get_lineage(name: str):
    path = _resolve_db(name)
    if not path.exists():
        return {"error": "DB not found"}
    return {"lineage": _read_lineage(path)}


@app.get("/api/dbs/{name}/stats")
async def get_stats(name: str, limit: int = 50):
    path = _resolve_db(name)
    if not path.exists():
        return {"error": "DB not found"}
    champion = _read_champion_row(path)
    matches = _read_matches(path, limit)
    return {"champion": champion, "matches": matches}


# ---------------------------------------------------------------------------
# Bot operations
# ---------------------------------------------------------------------------


@app.post("/api/bots/export")
async def export_bot(body: dict):
    engine = body.get("engine", "cpu")
    db_name = body.get("db", ENGINES[engine]["default_db"])
    db_path = str(_resolve_db(db_name))
    bot_id = body.get("bot_id")
    output = body.get("output")

    python = sys.executable
    cmd = [python, str(SCRIPT_DIR / "export_bot.py"), "--engine", engine, "--db", db_path]
    if bot_id:
        cmd.extend(["--bot-id", bot_id])
    if output:
        cmd.extend(["--output", output])
    else:
        default_out_dir = SCRIPT_DIR / "exported_bots"
        default_out_dir.mkdir(exist_ok=True)
        cmd.extend(["--output", str(default_out_dir)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    return {"output": result.stdout.strip(), "stderr": result.stderr.strip()}


@app.post("/api/bots/promote")
async def promote_bot(body: dict):
    engine = body.get("engine", "cpu")
    db_name = body.get("db", ENGINES[engine]["default_db"])
    db_path = str(_resolve_db(db_name))
    bot_id = body.get("bot_id")

    python = sys.executable
    cmd = [python, str(SCRIPT_DIR / "promote_bot.py"), "--engine", engine, "--db", db_path]
    if bot_id:
        cmd.extend(["--bot-id", bot_id])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    return {"output": result.stdout.strip(), "stderr": result.stderr.strip()}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


async def _broadcast_refresh():
    async with sessions_lock:
        data = {
            "type": "refresh",
            "sessions": [_session_info(sid, s) for sid, s in sessions.items()],
        }
    dead: set[WebSocket] = set()
    async with ws_lock:
        for ws in ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        ws_clients.difference_update(dead)


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Worker promotion
# ---------------------------------------------------------------------------


@app.post("/api/workers/promote-best")
async def promote_best_worker(body: dict):
    """Find the best worker DB and promote its champion to the main DB.

    Only promotes if a worker has reached MUTATION mode. Preserves existing
    main DB match totals by summing across all workers.
    """
    engine = body.get("engine", "cpu")
    if engine not in ENGINES:
        return {"error": f"Unknown engine: {engine}"}

    cfg = ENGINES[engine]
    main_db_path = _resolve_db(cfg["default_db"])

    worker_dbs = sorted(SCRIPT_DIR.glob(f"gladiator_{engine}_worker_*.db"))
    if not worker_dbs:
        return {"error": "No worker databases found"}

    best = None
    best_path = None

    for wdb in worker_dbs:
        row = _read_champion_row(wdb)
        if not row:
            continue
        if row.get("mode") == "MUTATION":
            best = row
            best_path = wdb
            break  # first MUTATION worker wins

    if best is None or best_path is None:
        return {"error": "No worker has reached MUTATION mode yet"}

    try:
        wcon = sqlite3.connect(str(best_path), timeout=2)
        wcon.row_factory = sqlite3.Row
        champ = dict(wcon.execute("SELECT * FROM champion WHERE id = 1").fetchone())
        wlogs = [dict(r) for r in wcon.execute("SELECT * FROM champion_log ORDER BY rowid").fetchall()]
        wmatches = [dict(r) for r in wcon.execute("SELECT * FROM matches ORDER BY rowid").fetchall()]
        wcon.close()

        mcon = sqlite3.connect(str(main_db_path), timeout=2)
        mcon.row_factory = sqlite3.Row

        # Preserve existing main DB match count
        existing = mcon.execute("SELECT total_matches FROM champion WHERE id = 1").fetchone()
        existing_total = existing["total_matches"] if existing else 0

        mcon.execute("DELETE FROM champion WHERE id = 1")
        mcon.execute(
            "INSERT INTO champion (id, bot_id, params_json, generation, consecutive_wins, mode, total_matches, total_champion_changes, saved_at) "
            "VALUES (1, :bot_id, :params_json, :generation, :consecutive_wins, :mode, :total_matches, :total_champion_changes, :saved_at)",
            champ,
        )
        for log in wlogs:
            mcon.execute(
                "INSERT OR IGNORE INTO champion_log (bot_id, generation, crowned_at, total_matches_at_crowning) "
                "VALUES (:bot_id, :generation, :crowned_at, :total_matches_at_crowning)",
                log,
            )
        for m in wmatches:
            mcon.execute(
                "INSERT OR IGNORE INTO matches (generation, mode, champion_won, score_champion, score_challenger, pairs_played, total_games, played_at) "
                "VALUES (:generation, :mode, :champion_won, :score_champion, :score_challenger, :pairs_played, :total_games, :played_at)",
                m,
            )
        # Sum matches across ALL worker DBs for an accurate total
        total_worker_matches = existing_total
        for wdb in worker_dbs:
            row = _read_champion_row(wdb)
            if row:
                total_worker_matches += max(0, row["total_matches"] - existing_total)
        mcon.execute("UPDATE champion SET total_matches = ? WHERE id = 1", (total_worker_matches,))
        mcon.commit()
        mcon.close()

        return {
            "status": "ok",
            "engine": engine,
            "source": best_path.name,
            "champion": {
                "bot_id": best["bot_id"],
                "generation": best["generation"],
                "mode": best["mode"],
                "total_matches": total_worker_matches,
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------


async def _monitor():
    while True:
        try:
            await asyncio.sleep(2)
            async with sessions_lock:
                for sid, s in list(sessions.items()):
                    proc = s["process"]
                    if proc.poll() is not None:
                        if time.time() - s["created_at"] > 5:
                            del sessions[sid]
            async with play_lock:
                for gid, g in list(play_games.items()):
                    proc = g["process"]
                    if proc.returncode is not None:
                        del play_games[gid]
            await _broadcast_refresh()
        except asyncio.CancelledError:
            break


# ---------------------------------------------------------------------------
# Play — play chess against exported bots
# ---------------------------------------------------------------------------

play_games: dict[str, dict] = {}
play_lock = asyncio.Lock()

_DEPTH_MAP = {"low": 2, "medium": 3, "high": 5}


async def _uci_read_line(proc: asyncio.subprocess.Process, timeout: float = 10.0) -> str:
    try:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        return line.decode().strip()
    except (asyncio.TimeoutError, Exception):
        return ""


async def _uci_read_until(proc: asyncio.subprocess.Process, prefix: str, timeout: float = 10.0) -> str:
    while True:
        line = await _uci_read_line(proc, timeout=timeout)
        if not line:
            return ""
        if line.startswith(prefix):
            return line


async def _uci_read_bestmove(proc: asyncio.subprocess.Process, timeout: float = 60.0) -> str | None:
    while True:
        line = await _uci_read_line(proc, timeout=timeout)
        if not line:
            return None
        if line.startswith("bestmove "):
            parts = line.split()
            if len(parts) >= 2 and parts[1] != "(none)":
                return parts[1]


async def _uci_send(proc: asyncio.subprocess.Process, cmd: str):
    proc.stdin.write((cmd + "\n").encode())
    await proc.stdin.drain()


def _play_board_to_grid(board: chess.Board) -> dict:
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            cell = {"piece": None, "sq": sq}
            if piece:
                cell["piece"] = {
                    "symbol": _PIECE_SYMBOLS[(piece.piece_type, piece.color)],
                    "color": "white" if piece.color == chess.WHITE else "black",
                }
            cells.append(cell)
        rows.append(cells)
    legal_ucis = [m.uci() for m in board.legal_moves]
    last_move_sq = None
    if board.move_stack:
        last_move_sq = board.move_stack[-1].to_square
    return {
        "rows": rows,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fen": board.fen(),
        "legal_moves": legal_ucis,
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "last_move_sq": last_move_sq,
    }


@app.get("/api/play/bots")
async def play_list_bots():
    bots_dir = SCRIPT_DIR / "exported_bots"
    if not bots_dir.exists():
        return {"bots": []}
    bots = sorted(f.name for f in bots_dir.iterdir() if f.suffix == ".py")
    return {"bots": bots, "bots_dir": str(bots_dir)}


@app.post("/api/play/start")
async def play_start(body: dict):
    bot_path = body.get("bot_path")
    player_color_str = body.get("player_color", "white")
    difficulty = body.get("difficulty", "medium")

    if not bot_path:
        return {"error": "No bot path provided"}

    bp = Path(bot_path)
    if not bp.exists():
        return {"error": f"Bot not found: {bot_path}"}

    player_color = chess.WHITE if player_color_str == "white" else chess.BLACK
    depth = _DEPTH_MAP.get(difficulty, 6)
    python = sys.executable

    try:
        proc = await asyncio.create_subprocess_exec(
            python, str(bp),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return {"error": f"Failed to launch bot: {e}"}

    # UCI handshake
    await _uci_send(proc, "uci")
    engine_name = ""
    handshake_ok = False
    while True:
        line = await _uci_read_line(proc, timeout=5.0)
        if not line:
            break
        if line.startswith("id name"):
            engine_name = line[7:].strip()
        if line == "uciok":
            await _uci_send(proc, "isready")
            ready = await _uci_read_until(proc, "readyok", timeout=5.0)
            handshake_ok = bool(ready)
            break

    if not handshake_ok:
        ret = proc.returncode
        stderr_txt = ""
        if ret is not None:
            try:
                data = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
                stderr_txt = data.decode().strip()
            except Exception:
                pass
        msg = "Bot did not respond to UCI handshake"
        if stderr_txt:
            msg += f"\nBot error:\n{stderr_txt}"
        elif ret is not None:
            msg += f"\nBot process exited with code {ret}"
        if ret is None:
            proc.terminate()
        return {"error": msg}

    await _uci_send(proc, "ucinewgame")

    game_id = uuid.uuid4().hex[:8]
    board = chess.Board()

    # Bot moves first if player chose black
    if player_color == chess.BLACK:
        await _uci_send(proc, "position startpos")
        await _uci_send(proc, f"go depth {depth}")
        bot_move_uci = await _uci_read_bestmove(proc, timeout=30.0)
        if not bot_move_uci:
            stderr_txt = ""
            if proc.returncode is not None:
                try:
                    data = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
                    stderr_txt = data.decode().strip()
                except Exception:
                    pass
            msg = "Bot did not make a move"
            if stderr_txt:
                msg += f"\n{stderr_txt}"
            proc.terminate()
            return {"error": msg}
        board.push_uci(bot_move_uci)

    async with play_lock:
        play_games[game_id] = {
            "process": proc,
            "board": board,
            "player_color": player_color,
            "depth": depth,
            "bot_path": str(bp),
            "engine_name": engine_name,
            "difficulty": difficulty,
            "created_at": time.time(),
        }

    return {
        "game_id": game_id,
        "board": _play_board_to_grid(board),
        "player_color": player_color_str,
        "engine_name": engine_name,
        "bot_path": str(bp),
    }


@app.post("/api/play/{game_id}/move")
async def play_move(game_id: str, body: dict):
    move_uci = body.get("move")
    if not move_uci:
        return {"error": "No move provided"}

    async with play_lock:
        game = play_games.get(game_id)
    if not game:
        return {"error": "Game not found"}

    proc = game["process"]
    board = game["board"]
    player_color = game["player_color"]
    depth = game["depth"]

    if board.is_game_over():
        return {"error": "Game is already over"}

    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return {"error": f"Invalid UCI move: {move_uci}"}

    if move not in board.legal_moves:
        return {"error": "Illegal move"}

    board.push(move)

    if board.is_game_over():
        game["board"] = board
        return {
            "board": _play_board_to_grid(board),
            "game_over": True,
            "result": board.result(),
            "bot_move": None,
        }

    all_ucis = " ".join(m.uci() for m in board.move_stack)
    await _uci_send(proc, f"position startpos moves {all_ucis}")
    await _uci_send(proc, f"go depth {depth}")

    bot_move_uci = await _uci_read_bestmove(proc, timeout=30.0)
    if not bot_move_uci:
        return {"error": "Bot did not respond with a move"}

    bot_move = chess.Move.from_uci(bot_move_uci)
    board.push(bot_move)

    game_over = board.is_game_over()
    game["board"] = board

    return {
        "board": _play_board_to_grid(board),
        "bot_move": bot_move_uci,
        "game_over": game_over,
        "result": board.result() if game_over else None,
    }


@app.get("/api/play/{game_id}")
async def play_status(game_id: str):
    async with play_lock:
        game = play_games.get(game_id)
    if not game:
        return {"error": "Game not found"}
    board = game["board"]
    return {
        "game_id": game_id,
        "board": _play_board_to_grid(board),
        "player_color": "white" if game["player_color"] == chess.WHITE else "black",
        "difficulty": game.get("difficulty", "medium"),
        "engine_name": game.get("engine_name", ""),
        "bot_path": game.get("bot_path", ""),
    }


@app.post("/api/play/{game_id}/reset")
async def play_reset(game_id: str, body: dict):
    player_color_str = body.get("player_color", "white")
    difficulty = body.get("difficulty", "medium")

    async with play_lock:
        game = play_games.get(game_id)
    if not game:
        return {"error": "Game not found"}

    proc = game["process"]
    player_color = chess.WHITE if player_color_str == "white" else chess.BLACK
    depth = _DEPTH_MAP.get(difficulty, 6)

    await _uci_send(proc, "ucinewgame")

    board = chess.Board()

    if player_color == chess.BLACK:
        await _uci_send(proc, "position startpos")
        await _uci_send(proc, f"go depth {depth}")
        bot_move_uci = await _uci_read_bestmove(proc, timeout=30.0)
        if not bot_move_uci:
            return {"error": "Bot did not make a move"}
        board.push_uci(bot_move_uci)

    game["board"] = board
    game["player_color"] = player_color
    game["depth"] = depth
    game["difficulty"] = difficulty

    return {
        "board": _play_board_to_grid(board),
        "player_color": player_color_str,
    }


@app.post("/api/play/{game_id}/stop")
async def play_stop(game_id: str):
    async with play_lock:
        game = play_games.pop(game_id, None)
    if not game:
        return {"error": "Game not found"}
    proc = game["process"]
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except Exception:
        proc.kill()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# Static frontend (mount AFTER all API routes)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
GUI_DIR = SCRIPT_DIR / "gui" / "src"
if GUI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(GUI_DIR), html=True), name="gui")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8742
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()

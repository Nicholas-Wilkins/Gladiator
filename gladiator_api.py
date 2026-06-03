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
    (chess.PAWN, chess.WHITE): "♙", (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗", (chess.ROOK, chess.WHITE): "♖",
    (chess.QUEEN, chess.WHITE): "♕", (chess.KING, chess.WHITE): "♔",
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

    for i in range(num_workers):
        if num_workers > 1:
            worker_db_name = f"gladiator_{engine}_worker_{i}.db"
            worker_db_path = _resolve_db(worker_db_name)
            if not worker_db_path.exists() and main_db_path.exists():
                shutil.copy2(str(main_db_path), str(worker_db_path))
        else:
            worker_db_name = cfg["default_db"]
        db_path = str(_resolve_db(worker_db_name))

        worker_seed = (seed or 0) + i if seed is not None else None

        cmd = [python, str(SCRIPT_DIR / cfg["main"]), "--no-ui", "--db", db_path]
        if worker_seed is not None:
            cmd.extend(["--seed", str(worker_seed)])
        if max_games is not None:
            cmd.extend(["--max-games", str(max_games)])

        proc = subprocess.Popen(cmd)
        sid = uuid.uuid4().hex[:8]

        async with sessions_lock:
            sessions[sid] = {
                "engine": engine,
                "db_path": db_path,
                "process": proc,
                "created_at": time.time(),
                "seed": worker_seed,
                "max_games": max_games,
                "worker_id": i if num_workers > 1 else None,
            }

        created.append({"session_id": sid, "pid": proc.pid, "db": worker_db_name, "worker": i})

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
            await _broadcast_refresh()
        except asyncio.CancelledError:
            break


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

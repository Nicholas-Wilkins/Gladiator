const API_BASE = "http://127.0.0.1:8742";
let ws = null;
let currentPage = "dashboard";
let setupResolved = false;
let setupReadyReceived = false;
let wsConnected = false;
let _stoppingSessions = new Set();

// ---------------------------------------------------------------------------
// Setup overlay (shown during first-launch backend installation)
// ---------------------------------------------------------------------------

const setupOverlay = document.getElementById("setup-overlay");
const setupMessage = document.getElementById("setup-message");
const setupNote = document.getElementById("setup-note");
const setupSpinner = document.querySelector(".setup-spinner");
const progressFill = document.getElementById("progress-fill");

function hideSetup() {
  if (!setupOverlay) return;
  setupOverlay.classList.add("hidden");
  setupResolved = true;
}

function tryHideSetup() {
  if (setupReadyReceived && wsConnected) {
    hideSetup();
  }
}

function updateSetup(msg, pct) {
  if (!setupMessage || !progressFill) return;
  if (msg) setupMessage.textContent = msg;
  if (pct !== undefined) progressFill.style.width = Math.min(pct, 100) + "%";
}

async function pollSetupStatus() {
  try {
    const status = await window.__TAURI__.core.invoke("get_setup_status");
    if (status.step === "ready") {
      updateSetup("Ready!", 100);
      if (setupSpinner) setupSpinner.style.display = "none";
      setupReadyReceived = true;
      setTimeout(tryHideSetup, 400);
      // Safety net: if WS never connects within 15s, hide the overlay anyway
      setTimeout(() => { if (!setupResolved) hideSetup(); }, 15000);
    } else if (status.step === "error") {
      updateSetup("Error: " + (status.message || "Unknown error"), 0);
      if (setupSpinner) setupSpinner.style.display = "none";
      if (setupNote) setupNote.textContent = "Close and re-open the app to try again.";
    } else {
      if (setupSpinner) setupSpinner.style.display = "";
      updateSetup(status.message, status.progress);
    }
  } catch (_) {
    // poll too early — command not available yet; that's fine
  }
}

setInterval(pollSetupStatus, 2000);
pollSetupStatus();

// Fallback: if WebSocket connects before any setup event, hide overlay
// (covers dev mode and subsequent launches where server starts fast)

const _PIECE_MAP = {
  "♙": "P", "♘": "N", "♗": "B", "♖": "R", "♕": "Q", "♔": "K",
  "♟": "p", "♞": "n", "♝": "b", "♜": "r", "♛": "q", "♚": "k",
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function $(sel, parent) { return (parent || document).querySelector(sel); }
function $$(sel, parent) { return Array.from((parent || document).querySelectorAll(sel)); }

async function api(method, path, body) {
  const opts = { method, headers: { "Accept": "application/json" } };
  if (body) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const res = await fetch(`${API_BASE}${path}`, opts);
  return res.json();
}

function toast(msg, type = "info") {
  const container = $("#toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function renderBoard(board, compact) {
  if (!board || !board.rows) return "";
  const size = compact ? "24px" : "32px";
  const fontSize = compact ? "14px" : "20px";
  let html = `<div class="chess-board" style="grid-template-columns:repeat(8,${size});grid-template-rows:repeat(8,${size});font-size:${fontSize};line-height:${size}">`;
  board.rows.forEach((row, ri) => {
    row.forEach((cell, ci) => {
      const dark = (ri + ci) % 2 === 1;
      if (cell && cell.piece) {
        const sym = cell.piece.symbol;
        const color = cell.piece.color;
        html += `<div class="sq ${dark ? "sd" : "sl"}"><span class="${color === "white" ? "pw" : "pb"}">${escapeHtml(sym)}</span></div>`;
      } else {
        html += `<div class="sq ${dark ? "sd" : "sl"}"></div>`;
      }
    });
  });
  html += "</div>";
  if (!compact && board.last_san) {
    html += `<div style="font-size:11px;color:var(--text-dim);margin-top:4px;text-align:center">${escapeHtml(board.last_san)} &middot; ${board.turn === "white" ? "White" : "Black"} to move</div>`;
  }
  return html;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

$$(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".nav-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".page").forEach(p => p.classList.remove("active"));
    const page = $(`#page-${btn.dataset.page}`);
    if (page) page.classList.add("active");
    currentPage = btn.dataset.page;
    if (currentPage === "dashboard") refreshDashboard();
    if (currentPage === "training") refreshTrainingSessions();
    if (currentPage === "champion") refreshChampion();
    if (currentPage === "play") {
      if (!_playGameId) {
        $("#play-no-bot").style.display = "block";
        $("#play-game-area").style.display = "none";
        $("#play-bot-list").style.display = "none";
        $("#play-bot-info").style.display = "none";
      }
    }
  });
});

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  try {
    ws = new WebSocket(`ws://127.0.0.1:8742/api/ws`);
    ws.onopen = () => {
      $("#conn-status").className = "status-dot online";
      $("#conn-text").textContent = "Connected";
      wsConnected = true;
      tryHideSetup();
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "refresh") {
          refreshDashboard();
          if (currentPage === "training") refreshTrainingSessions();
          updatePromoteButton();
        }
      } catch (_) {}
    };
    ws.onclose = () => {
      $("#conn-status").className = "status-dot offline";
      $("#conn-text").textContent = "Polling";
      setTimeout(connectWS, 5000);
    };
    ws.onerror = () => { ws.close(); };
  } catch (_) {
    $("#conn-status").className = "status-dot offline";
    $("#conn-text").textContent = "Polling";
  }
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

async function refreshDashboard() {
  const engines = await api("GET", "/api/engines");
  const sessionsRes = await api("GET", "/api/sessions");
  renderEngineCards(engines.engines || []);
  renderSessionList(sessionsRes.sessions || []);
}

function renderEngineCards(engines) {
  const grid = $("#engine-cards");
  grid.innerHTML = engines.map(e => {
    const c = e.champion || {};
    const mode = c.mode || "none";
    const modeLabel = mode === "MUTATION" ? "MUTATION" : mode === "RANDOM" ? "RANDOM" : "—";
    const modeCls = mode.toLowerCase();
    return `<div class="engine-card">
      <div class="title">${escapeHtml(e.id)}</div>
      <div class="stat"><span>Database</span><span class="stat-value">${escapeHtml(e.default_db)}</span></div>
      <div class="stat"><span>Mode</span><span class="stat-value"><span class="mode-badge ${modeCls}">${modeLabel}</span></span></div>
      <div class="stat"><span>Wins</span><span class="stat-value">${c.consecutive_wins ?? "—"}</span></div>
      <div class="stat"><span>Matches</span><span class="stat-value">${c.total_matches ?? "—"}</span></div>
      <div class="stat"><span>Generation</span><span class="stat-value">${c.generation ?? "—"}</span></div>
    </div>`;
  }).join("");
}

function renderSessionList(sessions) {
  const el = $("#session-list");
  if (!sessions.length) {
    el.innerHTML = '<p class="empty-state">No training sessions running.</p>';
    return;
  }
  el.innerHTML = sessions.map(s => {
    const data = s.champion || {};
    const isRunning = s.status === "running";
    const isStopping = _stoppingSessions.has(s.id);
    const wid = s.worker_id !== null && s.worker_id !== undefined ? `W${s.worker_id + 1}` : "";
    const actionHtml = isStopping
      ? '<span style="color:var(--yellow);font-size:12px">Stopping\u2026</span>'
      : isRunning
        ? `<button class="btn stop-session" data-sid="${s.id}">Stop</button>`
        : '<span style="color:var(--text-dim);font-size:12px">Stopped</span>';
    return `<div class="session-item">
      <div class="session-board">
        ${renderBoard(s.board, true)}
      </div>
      <div class="info">
        <div class="name">${escapeHtml(s.engine)} ${wid} <span style="color:var(--text-dim);font-weight:400;font-size:12px">PID ${s.pid}</span></div>
        <div class="meta">${isRunning ? "RUNNING" : "STOPPED"} &middot; Gen ${data.generation ?? "?"} &middot; ${data.mode || "?"} &middot; ${data.total_matches ?? 0} matches &middot; ${data.consecutive_wins ?? 0} streak</div>
      </div>
      <div class="actions">${actionHtml}</div>
    </div>`;
  }).join("");

  $$(".stop-session").forEach(btn => {
    btn.addEventListener("click", async () => {
      const sid = btn.dataset.sid;
      if (_stoppingSessions.has(sid)) return;
      _stoppingSessions.add(sid);
      btn.disabled = true;
      btn.textContent = "Stopping\u2026";
      await api("POST", `/api/sessions/${sid}/stop`);
      toast("Session stopping...", "info");
      refreshDashboard();
    });
  });
  // Clear sessions from _stoppingSessions once they are no longer in the list
  const active = new Set(sessions.map(s => s.id));
  for (const sid of _stoppingSessions) {
    if (!active.has(sid)) _stoppingSessions.delete(sid);
  }
}

// ---------------------------------------------------------------------------
// Training page
// ---------------------------------------------------------------------------

async function refreshTrainingSessions() {
  const res = await api("GET", "/api/sessions");
  const el = $("#training-session-list");
  const log = $("#training-log");
  const sessions = res.sessions || [];
  if (!sessions.length) {
    el.innerHTML = '<p class="empty-state">No training sessions running.</p>';
    log.textContent = "";
    return;
  }
  log.textContent = sessions.map(s => {
    const d = s.champion || {};
    return `[${s.engine}${s.worker_id !== null ? "#"+(s.worker_id+1) : ""}] PID ${s.pid} ${s.status} gen=${d.generation||"?"} ${d.mode||"?"} matches=${d.total_matches??"?"} streak=${d.consecutive_wins??"?"}`;
  }).join("\n");
  el.innerHTML = sessions.map(s => {
    const data = s.champion || {};
    const isRunning = s.status === "running";
    const isStopping = _stoppingSessions.has(s.id);
    const wid = s.worker_id !== null && s.worker_id !== undefined ? `Worker ${s.worker_id + 1}` : "";
    const actionHtml = isStopping
      ? '<span style="color:var(--yellow);font-size:12px">Stopping\u2026</span>'
      : isRunning
        ? `<button class="btn stop-session" data-sid="${s.id}">Stop</button>`
        : '<span style="color:var(--text-dim);font-size:12px">Stopped</span>';
    return `<div class="session-item">
      <div class="session-board">${renderBoard(s.board, false)}</div>
      <div class="info">
        <div class="name">${escapeHtml(s.engine)} ${wid} <span style="color:var(--text-dim);font-weight:400;font-size:12px">PID ${s.pid}</span></div>
        <div class="meta">${isRunning ? "RUNNING" : "STOPPED"} &middot; Gen ${data.generation ?? "?"} &middot; ${data.mode || "?"} &middot; ${data.total_matches ?? 0} matches &middot; ${data.consecutive_wins ?? 0} streak</div>
      </div>
      <div class="actions">${actionHtml}</div>
    </div>`;
  }).join("");

  $$("#training-session-list .stop-session").forEach(btn => {
    btn.addEventListener("click", async () => {
      const sid = btn.dataset.sid;
      if (_stoppingSessions.has(sid)) return;
      _stoppingSessions.add(sid);
      btn.disabled = true;
      btn.textContent = "Stopping\u2026";
      await api("POST", `/api/sessions/${sid}/stop`);
      toast("Session stopping...", "info");
      refreshTrainingSessions();
    });
  });
  updatePromoteButton();
}

$("#btn-start").addEventListener("click", async () => {
  const engine = $("#tr-engine").value;
  const numWorkers = parseInt($("#tr-workers").value) || 1;
  const seed = $("#tr-seed").value ? parseInt($("#tr-seed").value) : undefined;
  const maxGames = $("#tr-max-games").value ? parseInt($("#tr-max-games").value) : undefined;

  const body = { engine, num_workers: numWorkers };
  if (seed !== undefined) body.seed = seed;
  if (maxGames !== undefined) body.max_games = maxGames;

  const result = await api("POST", "/api/sessions/create", body);
  if (result.error) {
    toast(`Error: ${result.error}`, "error");
  } else if (result.sessions) {
    toast(`Started ${numWorkers} ${engine} worker(s)`, "success");
    refreshTrainingSessions();
  } else {
    toast(`Started ${engine} training (PID ${result.pid})`, "success");
    refreshTrainingSessions();
  }
});

$("#btn-promote-best").addEventListener("click", async () => {
  const engine = $("#tr-engine").value;
  if (!confirm(`Promote the best ${engine} worker champion to the main database?`)) return;
  const result = await api("POST", "/api/workers/promote-best", { engine });
  if (result.error) {
    toast(`Promotion error: ${result.error}`, "error");
  } else {
    toast(`Promoted ${result.source} (gen ${result.champion.generation})`, "success");
    refreshTrainingSessions();
  }
});

// Show promote button when multi-worker sessions are detected
async function updatePromoteButton() {
  const res = await api("GET", "/api/sessions");
  const sessions = res.sessions || [];
  const multiWorker = sessions.some(s => s.worker_id !== null && s.worker_id !== undefined);
  $("#promote-actions").style.display = multiWorker ? "flex" : "none";
}

// ---------------------------------------------------------------------------
// Champion page — aggregates across all worker DBs for the selected engine
// ---------------------------------------------------------------------------

let _bestExportDb = null;

$("#ch-refresh").addEventListener("click", refreshChampion);
$("#ch-engine").addEventListener("change", refreshChampion);

async function refreshChampion() {
  const engine = $("#ch-engine").value;
  const enginePrefix = `gladiator_${engine === "nn-mini" ? "nn_mini" : engine}`;

  const dbsRes = await api("GET", "/api/dbs");
  const dbs = (dbsRes.dbs || []).filter(d => d.name.startsWith(enginePrefix));

  if (!dbs.length) {
    renderChampionCard(null);
    renderLineage([]);
    renderMatchStats([]);
    _bestExportDb = null;
    return;
  }

  const results = await Promise.all(
    dbs.map(d =>
      Promise.all([
        api("GET", `/api/dbs/${d.name}/champion`),
        api("GET", `/api/dbs/${d.name}/lineage`),
        api("GET", `/api/dbs/${d.name}/stats`),
      ]).then(([c, l, s]) => ({
        db: d.name,
        champion: c.champion,
        lineage: l.lineage || [],
        matches: s.matches || [],
      }))
    )
  );

  let bestChamp = null;
  let bestDb = null;
  let allMatches = [];

  for (const r of results) {
    if (r.champion) {
      const streak = r.champion.consecutive_wins || 0;
      if (!bestChamp || streak > (bestChamp.consecutive_wins || 0)) {
        bestChamp = r.champion;
        bestDb = r.db;
      }
    }
    allMatches = allMatches.concat(r.matches);
  }

  allMatches.sort((a, b) => ((b.generation || 0) - (a.generation || 0)));

  renderChampionCard(bestChamp);
  renderLineage(results.find(r => r.db === bestDb)?.lineage || []);
  renderMatchStats(allMatches);

  _bestExportDb = bestDb;
}

function renderChampionCard(champion) {
  const el = $("#champion-card");
  if (!champion) {
    el.innerHTML = '<p class="empty-state">No champion data found.</p>';
    return;
  }
  const mode = champion.mode || "—";
  const modeCls = mode.toLowerCase();
  el.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px">
      <div><span style="color:var(--text-dim);font-size:12px">Bot ID</span><br>${escapeHtml(champion.bot_id || "—")}</div>
      <div><span style="color:var(--text-dim);font-size:12px">Mode</span><br><span class="mode-badge ${modeCls}">${mode}</span></div>
      <div><span style="color:var(--text-dim);font-size:12px">Generation</span><br>${champion.generation || "—"}</div>
      <div><span style="color:var(--text-dim);font-size:12px">Consecutive Wins</span><br>${champion.consecutive_wins ?? "—"}</div>
      <div><span style="color:var(--text-dim);font-size:12px">Total Matches</span><br>${champion.total_matches ?? "—"}</div>
      <div><span style="color:var(--text-dim);font-size:12px">Champion Changes</span><br>${champion.total_champion_changes ?? "—"}</div>
    </div>`;
}

function renderLineage(lineage) {
  const el = $("#lineage-table");
  if (!lineage.length) {
    el.innerHTML = '<p class="empty-state">No champion lineage recorded.</p>';
    return;
  }
  let html = `<table><thead><tr><th>#</th><th>Bot ID</th><th>Generation</th><th>Crowned At</th><th>Matches</th></tr></thead><tbody>`;
  lineage.forEach((row, i) => {
    const date = row.crowned_at ? new Date(row.crowned_at * 1000).toLocaleString() : "—";
    html += `<tr><td>${i + 1}</td><td>${escapeHtml((row.bot_id || "").slice(0, 16))}…</td><td>${row.generation || "—"}</td><td>${date}</td><td>${row.total_matches_at_crowning ?? "—"}</td></tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function renderMatchStats(matches) {
  const el = $("#stats-table");
  if (!matches.length) {
    el.innerHTML = '<p class="empty-state">No match data yet.</p>';
    return;
  }
  let html = `<table><thead><tr><th>Gen</th><th>Mode</th><th>Outcome</th><th>Score</th><th>Games</th></tr></thead><tbody>`;
  matches.slice(0, 30).forEach(m => {
    const outcome = m.champion_won ? '<span class="win">KEPT</span>' : '<span class="loss">DETHRONED</span>';
    html += `<tr><td>${m.generation}</td><td>${m.mode || "—"}</td><td>${outcome}</td><td>${m.score_champion ?? "—"} / ${m.score_challenger ?? "—"}</td><td>${m.total_games ?? "—"}</td></tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

async function _exportFromBest(engine, output) {
  if (!_bestExportDb) {
    toast("No champion data to export.", "error");
    return null;
  }
  const body = { engine, db: _bestExportDb };
  if (output) body.output = output;
  const result = await api("POST", "/api/bots/export", body);
  return result;
}

// Export (auto-selects best champion across all workers)
$("#ch-export").addEventListener("click", async () => {
  const engine = $("#ch-engine").value;
  const result = await _exportFromBest(engine);
  const out = $("#champion-output");
  if (!result) return;
  if (result.error) {
    out.textContent = `Error: ${result.error}`;
    toast("Export failed", "error");
  } else {
    out.textContent = `Exported: ${result.output}\n${result.stderr || ""}`;
    toast("Bot exported successfully", "success");
  }
});

// Export As
$("#ch-export-as").addEventListener("click", () => {
  const row = $("#ch-export-as-row");
  const pathInput = $("#ch-export-path");
  pathInput.value = "";
  pathInput.placeholder = "Output path (default: auto-generated name in exported_bots/)";
  row.style.display = row.style.display === "none" ? "block" : "none";
  if (row.style.display === "block") pathInput.focus();
});

$("#ch-export-as-confirm").addEventListener("click", async () => {
  const engine = $("#ch-engine").value;
  const path = $("#ch-export-path").value.trim();
  const result = await _exportFromBest(engine, path || undefined);
  const out = $("#champion-output");
  if (!result) return;
  if (result.error) {
    out.textContent = `Error: ${result.error}`;
    toast("Export failed", "error");
  } else {
    out.textContent = `Exported: ${result.output}\n${result.stderr || ""}`;
    toast("Bot exported successfully", "success");
    $("#ch-export-as-row").style.display = "none";
  }
});

// Promote
$("#ch-promote").addEventListener("click", async () => {
  if (!confirm("Promote the current champion to MUTATION mode? This will set mode=MUTATION in the database.")) return;
  const engine = $("#ch-engine").value;
  const db = _bestExportDb;
  if (!db) { toast("No champion data to promote.", "error"); return; }
  const result = await api("POST", "/api/bots/promote", { engine, db });
  const out = $("#champion-output");
  if (result.error) {
    out.textContent = `Error: ${result.error}`;
    toast("Promotion failed", "error");
  } else {
    out.textContent = result.output || "Done";
    toast("Promoted to MUTATION mode", "success");
    refreshChampion();
  }
});

// ---------------------------------------------------------------------------
// Play tab — interactive chess against exported bots
// ---------------------------------------------------------------------------

let _playGameId = null;
let _playBoardState = null;
let _selectedSq = null;
let _playLocked = false;
let _snapshotBeforeMove = null;
let _isDragging = false;
let _dragFromSq = null;

$("#play-choose-btn").addEventListener("click", showPlayBotList);
$("#play-change-bot").addEventListener("click", showPlayBotList);
$("#play-new-bot").addEventListener("click", showPlayBotList);
$("#play-reset").addEventListener("click", resetPlayGame);
$("#play-color").addEventListener("change", () => { if (_playGameId) resetPlayGame(); });
$("#play-difficulty").addEventListener("change", () => { if (_playGameId) resetPlayGame(); });

async function showPlayBotList() {
  _playGameId = null;
  _playBoardState = null;
  _selectedSq = null;
  _playLocked = false;
  $("#play-no-bot").style.display = "none";
  $("#play-game-area").style.display = "none";
  $("#play-bot-info").style.display = "none";
  $("#play-bot-list").style.display = "block";
  $("#play-bot-items").innerHTML = '<p class="empty-state">Loading...</p>';
  $("#play-no-bots-msg").style.display = "none";

  const res = await api("GET", "/api/play/bots");
  const bots = res.bots || [];
  const container = $("#play-bot-items");
  if (!bots.length) {
    container.innerHTML = "";
    $("#play-no-bots-msg").style.display = "block";
    return;
  }
  container.innerHTML = bots.map(b =>
    `<button class="play-bot-item" data-path="${res.bots_dir}/${b}">${escapeHtml(b)}</button>`
  ).join("");
  container.querySelectorAll(".play-bot-item").forEach(btn => {
    btn.addEventListener("click", () => startPlayGame(btn.dataset.path));
  });
}

async function startPlayGame(botPath) {
  const color = $("#play-color").value;
  const difficulty = $("#play-difficulty").value;

  const res = await api("POST", "/api/play/start", {
    bot_path: botPath,
    player_color: color,
    difficulty: difficulty,
  });

  if (res.error) {
    toast(`Error: ${res.error}`, "error");
    return;
  }

  _playGameId = res.game_id;
  _playBoardState = res.board;
  _selectedSq = null;
  _playLocked = false;

  const botName = res.engine_name || botPath.split("/").pop() || botPath;
  $("#play-bot-name").textContent = botName;
  $("#play-bot-info").style.display = "flex";
  $("#play-bot-list").style.display = "none";
  $("#play-game-area").style.display = "flex";
  $("#play-no-bot").style.display = "none";

  renderPlayBoard();
  updatePlayStatus();
}

function renderPlayBoard() {
  const el = $("#play-board");
  if (!_playBoardState || !_playBoardState.rows) {
    el.innerHTML = "";
    return;
  }

  const rows = _playBoardState.rows;
  const playerColor = $("#play-color").value;
  const isFlipped = playerColor === "black";
  const legalMoves = _playBoardState.legal_moves || [];

  let legalDests = [];
  if (_selectedSq !== null) {
    const fromName = chess.sqName(_selectedSq);
    legalDests = legalMoves
      .filter(m => m.startsWith(fromName))
      .map(m => {
        const toName = m.slice(2, 4);
        const f = "abcdefgh".indexOf(toName[0]);
        const r = parseInt(toName[1]) - 1;
        return r * 8 + f;
      });
  }

  const rankOrder = isFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];
  const fileOrder = isFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];

  el.style.gridTemplateColumns = "24px repeat(8, 48px)";
  el.style.gridTemplateRows = "repeat(8, 48px) 24px";

  let html = "";

  for (const ri of rankOrder) {
    const rankNum = 8 - ri;
    html += `<div class="sq-label rank-label">${rankNum}</div>`;

    for (const fi of fileOrder) {
      const cell = rows[ri][fi];
      const sq = cell.sq;
      const piece = cell.piece;
      const dark = (ri + fi) % 2 === 1;
      const isSelected = _selectedSq === sq;
      const isLegal = legalDests.includes(sq);
      const isLastMove = _playBoardState.last_move_sq === sq;

      let cls = "sq";
      if (dark) cls += " sd";
      else cls += " sl";
      if (isSelected) cls += " selected";
      if (isLegal) cls += " legal";
      if (isLastMove) cls += " last-move";

      const pieceHtml = piece ? (() => {
        const pt = {'♟':'pawn','♞':'knight','♝':'bishop','♜':'rook','♛':'queen','♚':'king'}[piece.symbol] || '';
        const extra = pt ? ` piece-${pt}` : '';
        return `<span class="piece-${piece.color}${extra}">${escapeHtml(piece.symbol)}</span>`;
      })() : "";
      html += `<div class="${cls}" data-sq="${sq}">${pieceHtml}</div>`;
    }
  }

  html += `<div class="sq-label corner-label"></div>`;
  for (const fi of fileOrder) {
    const letter = "abcdefgh"[fi];
    html += `<div class="sq-label file-label">${letter}</div>`;
  }

  el.innerHTML = html;

  el.querySelectorAll(".sq").forEach(sqEl => {
    sqEl.addEventListener("click", () => handlePlaySquareClick(parseInt(sqEl.dataset.sq)));
    sqEl.addEventListener("mousedown", (e) => handleSqMouseDown(e, parseInt(sqEl.dataset.sq)));
  });
}

function handleSqMouseDown(e, sq) {
  if (_playLocked || !_playBoardState || _playBoardState.is_game_over) return;
  const piece = _playBoardState.rows.flat().find(c => c.sq === sq)?.piece;
  if (!piece) return;
  const playerColor = $("#play-color").value;
  const isPlayerTurn = (playerColor === "white" && _playBoardState.turn === "white") ||
                       (playerColor === "black" && _playBoardState.turn === "black");
  if (piece.color !== playerColor || !isPlayerTurn) return;
  _selectedSq = null;
  _dragFromSq = sq;
  _isDragging = false;
  document.body.classList.add("dragging-piece");
  const ghost = document.createElement("span");
  ghost.className = `piece-${piece.color} drag-ghost`;
  ghost.textContent = piece.symbol;
  ghost.style.left = e.clientX + "px";
  ghost.style.top = e.clientY + "px";
  document.body.appendChild(ghost);
}

function handleDocMouseMove(e) {
  if (_dragFromSq === null) return;
  _isDragging = true;
  const ghost = document.querySelector(".drag-ghost");
  if (ghost) {
    ghost.style.left = e.clientX + "px";
    ghost.style.top = e.clientY + "px";
  }
}

function handleDocMouseUp(e) {
  const ghost = document.querySelector(".drag-ghost");
  if (ghost) ghost.remove();
  if (_dragFromSq === null) { document.body.classList.remove("dragging-piece"); return; }
  const fromSq = _dragFromSq;
  _dragFromSq = null;
  document.body.classList.remove("dragging-piece");
  if (!_isDragging) return;
  _isDragging = false;
  const el = document.elementFromPoint(e.clientX, e.clientY);
  if (!el) return;
  const sqEl = el.closest(".sq");
  if (!sqEl) return;
  const toSq = parseInt(sqEl.dataset.sq);
  if (isNaN(toSq) || fromSq === toSq) return;
  tryMakeMove(fromSq, toSq);
}

function applyMoveLocally(board, fromSq, toSq, promoUci) {
  const newRows = board.rows.map(row => row.map(cell =>
    ({...cell, piece: cell.piece ? {...cell.piece} : null})
  ));
  let fromCell, toCell;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      if (newRows[r][f].sq === fromSq) fromCell = newRows[r][f];
      if (newRows[r][f].sq === toSq) toCell = newRows[r][f];
    }
  }
  if (!fromCell || !fromCell.piece) return false;

  const movingPiece = fromCell.piece;
  const capturedPiece = toCell.piece;
  const symToType = {'♟':'p','♞':'n','♝':'b','♜':'r','♛':'q','♚':'k'};
  const movingType = symToType[movingPiece.symbol] || '?';

  if (promoUci && promoUci.length === 5 && movingType === 'p') {
    const promoSymbols = {'q':'♛','r':'♜','b':'♝','n':'♞'};
    toCell.piece = { symbol: promoSymbols[promoUci[4]] || '♛', color: movingPiece.color };
  } else {
    toCell.piece = movingPiece;
  }
  fromCell.piece = null;

  if (movingType === 'k' && Math.abs(toSq - fromSq) === 2) {
    const isKingside = toSq > fromSq;
    const rookFrom = isKingside ? fromSq + 3 : fromSq - 4;
    const rookTo = isKingside ? fromSq + 1 : fromSq - 1;
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        if (newRows[r][f].sq === rookFrom) {
          const rook = newRows[r][f].piece;
          for (let r2 = 0; r2 < 8; r2++) {
            for (let f2 = 0; f2 < 8; f2++) {
              if (newRows[r2][f2].sq === rookTo) {
                newRows[r2][f2].piece = rook;
                newRows[r][f].piece = null;
              }
            }
          }
        }
      }
    }
  }

  if (movingType === 'p' && !capturedPiece && Math.abs(toSq - fromSq) % 8 !== 0) {
    const capturedRank = movingPiece.color === 'white' ? Math.floor(toSq / 8) - 1 : Math.floor(toSq / 8) + 1;
    const capturedSq = capturedRank * 8 + (toSq % 8);
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        if (newRows[r][f].sq === capturedSq) newRows[r][f].piece = null;
      }
    }
  }

  board.rows = newRows;
  board.turn = board.turn === 'white' ? 'black' : 'white';
  board.legal_moves = [];
  board.last_move_sq = toSq;
  return true;
}

function tryMakeMove(fromSq, toSq) {
  if (_playLocked || !_playBoardState || _playBoardState.is_game_over) return;

  const playerColor = $("#play-color").value;
  const board = _playBoardState;
  const isPlayerTurn = (playerColor === "white" && board.turn === "white") ||
                       (playerColor === "black" && board.turn === "black");
  if (!isPlayerTurn) {
    toast("Wait for the bot to move", "info");
    return;
  }

  const fromName = chess.sqName(fromSq);
  const toName = chess.sqName(toSq);
  const uci = fromName + toName;
  let moveUci = uci;
  const legalMoves = board.legal_moves || [];

  if (!legalMoves.includes(moveUci)) {
    const promoUci = uci + "q";
    if (legalMoves.includes(promoUci)) {
      moveUci = promoUci;
    } else {
      toast("Illegal move", "error");
      renderPlayBoard();
      return;
    }
  }

  _snapshotBeforeMove = JSON.parse(JSON.stringify(_playBoardState));
  applyMoveLocally(board, fromSq, toSq, moveUci);
  _selectedSq = null;
  _playLocked = true;
  renderPlayBoard();
  updatePlayStatus();
  makePlayMove(moveUci);
}

function handlePlaySquareClick(sq) {
  if (_playLocked || _isDragging || !_playBoardState) return;
  _isDragging = false;
  const board = _playBoardState;
  if (board.is_game_over) return;

  const playerColor = $("#play-color").value;
  const isPlayerTurn = (playerColor === "white" && board.turn === "white") ||
                       (playerColor === "black" && board.turn === "black");
  if (!isPlayerTurn) {
    toast("Wait for the bot to move", "info");
    return;
  }

  let clickedPiece = null;
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      if (board.rows[r][f].sq === sq && board.rows[r][f].piece) {
        clickedPiece = board.rows[r][f].piece;
      }
    }
  }

  if (_selectedSq === null) {
    if (clickedPiece && clickedPiece.color === playerColor) {
      _selectedSq = sq;
      renderPlayBoard();
    }
    return;
  }

  const fromSq = _selectedSq;
  _selectedSq = null;

  if (clickedPiece && clickedPiece.color === playerColor && sq !== fromSq) {
    _selectedSq = sq;
    renderPlayBoard();
    return;
  }

  tryMakeMove(fromSq, sq);
}

async function makePlayMove(moveUci) {
  _playLocked = true;
  updatePlayStatus();
  const res = await api("POST", `/api/play/${_playGameId}/move`, { move: moveUci });

  if (res.error) {
    toast(`Error: ${res.error}`, "error");
    _playLocked = false;
    return;
  }

  _playBoardState = res.board;
  renderPlayBoard();
  updatePlayStatus();

  if (res.game_over) {
    const result = res.result;
    let msg = "Game Over — ";
    if (result === "1-0") msg += "White wins!";
    else if (result === "0-1") msg += "Black wins!";
    else msg += "Draw!";
    toast(msg, "info");
    _playLocked = false;
    return;
  }

  if (res.bot_move) {
    setTimeout(() => {
      _playLocked = false;
      updatePlayStatus();
    }, 300);
  } else {
    _playLocked = false;
    updatePlayStatus();
  }
}

function updatePlayStatus() {
  const el = $("#play-status");
  if (!_playBoardState) {
    el.textContent = "Select a bot to begin";
    return;
  }
  if (_playLocked) {
    el.textContent = "Bot is thinking\u2026";
    return;
  }
  if (_playBoardState.is_game_over) {
    const r = _playBoardState.result;
    if (r === "1-0") el.textContent = "Game Over — White wins!";
    else if (r === "0-1") el.textContent = "Game Over — Black wins!";
    else el.textContent = "Game Over — Draw!";
    return;
  }
  const turn = _playBoardState.turn;
  const playerColor = $("#play-color").value;
  const isPlayerTurn = playerColor === turn;
  if (isPlayerTurn) {
    el.textContent = "Your turn";
  } else {
    el.textContent = "Bot is thinking\u2026";
  }
}

async function resetPlayGame() {
  if (!_playGameId) return;
  const color = $("#play-color").value;
  const difficulty = $("#play-difficulty").value;

  const res = await api("POST", `/api/play/${_playGameId}/reset`, {
    player_color: color,
    difficulty: difficulty,
  });

  if (res.error) {
    toast(`Error: ${res.error}`, "error");
    return;
  }

  _playBoardState = res.board;
  _selectedSq = null;
  _playLocked = false;
  renderPlayBoard();
  updatePlayStatus();
}

// Simple chess utilities (subset of python-chess API for coordinate conversion)
const chess = {
  FILE_NAMES: "abcdefgh",
  sqName(sq) {
    const f = sq % 8;
    const r = Math.floor(sq / 8);
    return this.FILE_NAMES[f] + (r + 1);
  },
};

// Mouse-based drag-and-drop (avoids HTML5 DnD no-drop cursor)
document.addEventListener("mousemove", handleDocMouseMove);
document.addEventListener("mouseup", handleDocMouseUp);

// ---------------------------------------------------------------------------
// Auto-refresh dashboard
// ---------------------------------------------------------------------------

setInterval(() => {
  if (currentPage === "dashboard") refreshDashboard();
  if (currentPage === "training") refreshTrainingSessions();
}, 3000);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

connectWS();
refreshDashboard();

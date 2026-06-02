const API_BASE = "http://127.0.0.1:8742";
let ws = null;
let currentPage = "dashboard";
let setupResolved = false;
let setupReadyReceived = false;
let wsConnected = false;

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
      setupReadyReceived = true;
      setTimeout(tryHideSetup, 400);
    } else if (status.step === "error") {
      updateSetup("Error: " + (status.message || "Unknown error"), 0);
      if (setupSpinner) setupSpinner.style.display = "none";
      if (setupNote) setupNote.textContent = "Close and re-open the app to try again.";
    } else {
      updateSetup(status.message, status.progress);
    }
  } catch (_) {
    // Not running inside Tauri or command not available yet
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
      html += `<div class="sq ${dark ? "sd" : "sl"}">${cell ? escapeHtml(cell) : ""}</div>`;
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
    const wid = s.worker_id !== null && s.worker_id !== undefined ? `W${s.worker_id}` : "";
    return `<div class="session-item">
      <div class="session-board">
        ${renderBoard(s.board, true)}
      </div>
      <div class="info">
        <div class="name">${escapeHtml(s.engine)} ${wid} <span style="color:var(--text-dim);font-weight:400;font-size:12px">PID ${s.pid}</span></div>
        <div class="meta">${isRunning ? "RUNNING" : "STOPPED"} &middot; Gen ${data.generation ?? "?"} &middot; ${data.mode || "?"} &middot; ${data.total_matches ?? 0} matches &middot; ${data.consecutive_wins ?? 0} streak</div>
      </div>
      <div class="actions">
        ${isRunning ? `<button class="btn stop-session" data-sid="${s.id}">Stop</button>` : '<span style="color:var(--text-dim);font-size:12px">Stopped</span>'}
      </div>
    </div>`;
  }).join("");

  $$(".stop-session").forEach(btn => {
    btn.addEventListener("click", async () => {
      const sid = btn.dataset.sid;
      await api("POST", `/api/sessions/${sid}/stop`);
      toast("Session stopping...", "info");
      refreshDashboard();
    });
  });
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
    return `[${s.engine}${s.worker_id !== null ? "#"+s.worker_id : ""}] PID ${s.pid} ${s.status} gen=${d.generation||"?"} ${d.mode||"?"} matches=${d.total_matches??"?"} streak=${d.consecutive_wins??"?"}`;
  }).join("\n");
  el.innerHTML = sessions.map(s => {
    const data = s.champion || {};
    const isRunning = s.status === "running";
    const wid = s.worker_id !== null && s.worker_id !== undefined ? `Worker ${s.worker_id}` : "";
    return `<div class="session-item">
      <div class="session-board">${renderBoard(s.board, false)}</div>
      <div class="info">
        <div class="name">${escapeHtml(s.engine)} ${wid} <span style="color:var(--text-dim);font-weight:400;font-size:12px">PID ${s.pid}</span></div>
        <div class="meta">${isRunning ? "RUNNING" : "STOPPED"} &middot; Gen ${data.generation ?? "?"} &middot; ${data.mode || "?"} &middot; ${data.total_matches ?? 0} matches &middot; ${data.consecutive_wins ?? 0} streak</div>
      </div>
      <div class="actions">
        ${isRunning ? `<button class="btn stop-session" data-sid="${s.id}">Stop</button>` : '<span style="color:var(--text-dim);font-size:12px">Stopped</span>'}
      </div>
    </div>`;
  }).join("");

  $$("#training-session-list .stop-session").forEach(btn => {
    btn.addEventListener("click", async () => {
      await api("POST", `/api/sessions/${btn.dataset.sid}/stop`);
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
// Champion page
// ---------------------------------------------------------------------------

$("#ch-refresh").addEventListener("click", refreshChampion);

async function refreshChampion() {
  const engine = $("#ch-engine").value;
  const cfg = { cpu: "gladiator.db", nn: "gladiator_nn.db", mini: "gladiator_mini.db", "nn-mini": "gladiator_nn_mini.db" };
  const db = cfg[engine];

  const [championRes, lineageRes, statsRes] = await Promise.all([
    api("GET", `/api/dbs/${db}/champion`),
    api("GET", `/api/dbs/${db}/lineage`),
    api("GET", `/api/dbs/${db}/stats`),
  ]);

  renderChampionCard(championRes.champion);
  renderLineage(lineageRes.lineage || []);
  renderMatchStats(statsRes.matches || []);
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

// Export
$("#ch-export").addEventListener("click", async () => {
  const engine = $("#ch-engine").value;
  const cfg = { cpu: "gladiator.db", nn: "gladiator_nn.db", mini: "gladiator_mini.db", "nn-mini": "gladiator_nn_mini.db" };
  const db = cfg[engine];
  const result = await api("POST", "/api/bots/export", { engine, db });
  const out = $("#champion-output");
  if (result.error) {
    out.textContent = `Error: ${result.error}`;
    toast("Export failed", "error");
  } else {
    out.textContent = `Exported: ${result.output}\n${result.stderr || ""}`;
    toast("Bot exported successfully", "success");
  }
});

// Promote
$("#ch-promote").addEventListener("click", async () => {
  if (!confirm("Promote the current champion to MUTATION mode? This will set mode=MUTATION in the database.")) return;
  const engine = $("#ch-engine").value;
  const cfg = { cpu: "gladiator.db", nn: "gladiator_nn.db", mini: "gladiator_mini.db", "nn-mini": "gladiator_nn_mini.db" };
  const db = cfg[engine];
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

#!/usr/bin/env bash
# ── Windows build helper ─────────────────────────────────────────
# Prepares backend resources and builds the MSI installer locally.
# Run from the repo root on a Windows machine with Git Bash.
#
# Usage: bash scripts/build-windows.sh
#   or:   scripts/build-windows.sh (from repo root)
#   or:   scripts/build-windows.sh --skip-build (prepare resources only)
set -euo pipefail

# ── Sanity checks ────────────────────────────────────────────────
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *)
    echo "[ERROR] This script is for Windows only. On Linux/macOS, use 'cargo tauri build' directly."
    exit 1
    ;;
esac

SKIP_BUILD="${1:-}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BACKEND_DIR="gui/src-tauri/backend"

# ── Step 1: Create backend zip ───────────────────────────────────
echo "[1/3] Creating backend source bundle..."
mkdir -p "$BACKEND_DIR"
# Remove previous zip so stale contents don't linger
rm -f "$BACKEND_DIR/gladiator-backend-windows.zip"

# 7z is available on Windows runners (GitHub Actions) but may not
# be in PATH locally. Fall back to PowerShell's Compress-Archive.
if command -v 7z &>/dev/null; then
  7z a -tzip "$BACKEND_DIR/gladiator-backend-windows.zip" \
    gladiator_api.py main.py main_nn.py main_mini.py main_nn_mini.py \
    export_bot.py promote_bot.py install.py \
    requirements.txt \
    gladiator/ gladiator_nn/ gladiator_mini/ gladiator_nn_mini/
else
  echo "  7z not found, trying PowerShell Compress-Archive..."
  powershell -Command "
    \$files = @(
      'gladiator_api.py','main.py','main_nn.py','main_mini.py','main_nn_mini.py',
      'export_bot.py','promote_bot.py','install.py',
      'requirements.txt'
    )
    \$dirs = @('gladiator','gladiator_nn','gladiator_mini','gladiator_nn_mini')
    Get-ChildItem -Path \$files -File | Compress-Archive -DestinationPath '$BACKEND_DIR/gladiator-backend-windows.zip' -CompressionLevel Optimal
    foreach (\$d in \$dirs) {
      if (Test-Path \$d) {
        Compress-Archive -Path \"\$d/*\" -Update -DestinationPath '$BACKEND_DIR/gladiator-backend-windows.zip'
      }
    }
  "
fi

if [ -f "$BACKEND_DIR/gladiator-backend-windows.zip" ]; then
  echo "  OK: gladiator-backend-windows.zip created"
else
  echo "[ERROR] Failed to create backend zip!"
  exit 1
fi

# ── Step 2: Download uv.exe ──────────────────────────────────────
echo "[2/3] Downloading uv.exe..."
UV_URL="https://github.com/astral-sh/uv/releases/download/0.5.0/uv-x86_64-pc-windows-msvc.zip"
UV_ZIP="$(mktemp).zip"
UV_TARGET="$BACKEND_DIR/uv.exe"

# Skip download if uv.exe already exists and is recent
if [ -f "$UV_TARGET" ] && [ "$(stat -c%Y "$UV_TARGET" 2>/dev/null || stat -f%m "$UV_TARGET" 2>/dev/null)" -gt "$(date -d '7 days ago' +%s 2>/dev/null || echo 0)" ]; then
  echo "  uv.exe already exists and is recent, skipping download"
else
  if command -v curl &>/dev/null; then
    curl -sSLo "$UV_ZIP" "$UV_URL"
  elif command -v powershell &>/dev/null; then
    powershell -Command "Invoke-WebRequest -Uri '$UV_URL' -OutFile '$UV_ZIP'"
  else
    echo "[ERROR] Neither curl nor PowerShell found — cannot download uv.exe"
    exit 1
  fi

  if command -v 7z &>/dev/null; then
    7z e "$UV_ZIP" -o"$BACKEND_DIR/" uv.exe -aoa
  else
    powershell -Command "
      Expand-Archive -Path '$UV_ZIP' -DestinationPath '$BACKEND_DIR/' -Force
      Get-ChildItem '$BACKEND_DIR/uv.exe'
    "
  fi
  rm -f "$UV_ZIP"

  if [ -f "$UV_TARGET" ]; then
    echo "  OK: uv.exe downloaded and extracted"
  else
    echo "[ERROR] Failed to extract uv.exe!"
    exit 1
  fi
fi

# ── Step 3: Build ────────────────────────────────────────────────
# The zip and uv.exe in gui/src-tauri/backend/ are embedded into the
# Rust binary via include_bytes!() at compile time — no tauri.conf.json
# resource config needed.

if [ "$SKIP_BUILD" = "--skip-build" ]; then
  echo "[3/3] Skipping build (--skip-build). Run manually: cd gui && cargo tauri build"
else
  echo "[3/3] Building MSI installer..."
  cd gui
  cargo tauri build
fi

echo ""
echo "Done! MSI should be at: gui/src-tauri/target/release/bundle/msi/"

#!/usr/bin/env bash
set -euo pipefail

# ── Release helper for Gladiator ─────────────────────────────────
# Run this AFTER `cargo tauri build`. It:
#   1. Creates the Linux AppImage .tar.gz for the updater
#   2. Signs all platform artifacts with your signing key
#   3. Generates latest.json for the auto-updater
#
# Usage:
#   Local:   scripts/release.sh <version> [release notes file]
#   CI/gh:   TAURI_SIGNING_PRIVATE_KEY="$KEY" [TAURI_KEY_PASSWORD="$PW"] scripts/release.sh <version>
#
# Prerequisites (local):
#   - ~/.tauri/gladiator.key must exist
#   - Build artifacts in gui/target/release/bundle/{appimage,msi,nsis,dmg}
# Prerequisites (CI):
#   - TAURI_SIGNING_PRIVATE_KEY env var set to the full private key content
#   - TAURI_KEY_PASSWORD env var (optional — omit if key has no password)
#   - Artifacts in the current directory or a BUNDLE_DIR override

if [ $# -lt 1 ]; then
  echo "Usage: $0 <version> [release notes file]" >&2
  echo "Example: $0 0.1.23" >&2
  exit 1
fi

VERSION="$1"
NOTES_FILE="${2:-}"
KEY_FILE="$HOME/.tauri/gladiator.key"
BUNDLE_DIR="${BUNDLE_DIR:-gui/target/release/bundle}"
REPO="Nicholas-Wilkins/Gladiator"

# If TAURI_SIGNING_PRIVATE_KEY is set, write it to a temp file for the signer;
# otherwise require the local key file
TMP_KEY_FILE=""
cleanup() {
  if [ -n "$TMP_KEY_FILE" ] && [ -f "$TMP_KEY_FILE" ]; then
    rm -f "$TMP_KEY_FILE"
  fi
}
trap cleanup EXIT

if [ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
  TMP_KEY_FILE=$(mktemp)
  printf '%s' "$TAURI_SIGNING_PRIVATE_KEY" > "$TMP_KEY_FILE"
elif [ ! -f "$KEY_FILE" ]; then
  echo "ERROR: No TAURI_SIGNING_PRIVATE_KEY env var and no key file at $KEY_FILE" >&2
  echo "Generate one with: cargo tauri signer generate -w $KEY_FILE" >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

# ── Collect signatures ──────────────────────────────────────────
declare -A SIG

sign_file() {
  local label="$1"
  local path="$2"
  if [ ! -f "$path" ]; then
    echo "  [skip] $label — file not found: $path" >&2
    return 1
  fi
  echo "  [sign] $label" >&2
  local password="${TAURI_KEY_PASSWORD:-}"
  local key="${TMP_KEY_FILE:-$KEY_FILE}"
  local out

  # Prefer `tauri` (npm @tauri-apps/cli) over `cargo tauri` (cargo install tauri-cli).
  # Both CLIs read TAURI_SIGNING_PRIVATE_KEY from the environment automatically;
  # only pass -f (key file) when the env var isn't set (local dev).
  if [ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
    if command -v tauri &> /dev/null; then
      out=$(tauri signer sign -p "$password" "$path")
    elif command -v cargo-tauri &> /dev/null; then
      out=$(cargo tauri signer sign -p "$password" "$path")
    else
      echo "  [error] No tauri CLI found (install @tauri-apps/cli via npm or tauri-cli via cargo)" >&2
      return 1
    fi
  else
    if command -v tauri &> /dev/null; then
      out=$(tauri signer sign -f "$key" -p "$password" "$path")
    elif command -v cargo-tauri &> /dev/null; then
      out=$(cargo tauri signer sign -f "$key" -p "$password" "$path")
    else
      echo "  [error] No tauri CLI found (install @tauri-apps/cli via npm or tauri-cli via cargo)" >&2
      return 1
    fi
  fi

  # Extract just the signature line from the multi-line output
  SIG["$label"]="$(echo "$out" | awk '/^Public signature:/{getline; print}')"
  return 0
}

build_platform_json() {
  local first=true
  for label in "linux-x86_64" "windows-x86_64" "darwin-x86_64" "darwin-aarch64"; do
    local sig="${SIG[$label]:-}"
    [ -z "$sig" ] && continue
    $first || echo ","
    first=false
    local url
    case "$label" in
      linux-x86_64)    url="https://github.com/${REPO}/releases/download/v${VERSION}/Gladiator_${VERSION}_amd64.AppImage.tar.gz" ;;
      windows-x86_64)  url="https://github.com/${REPO}/releases/download/v${VERSION}/Gladiator_${VERSION}_x64_portable.zip" ;;
      darwin-x86_64)   url="https://github.com/${REPO}/releases/download/v${VERSION}/Gladiator_${VERSION}_x64.dmg" ;;
      darwin-aarch64)  url="https://github.com/${REPO}/releases/download/v${VERSION}/Gladiator_${VERSION}_aarch64.dmg" ;;
    esac
    printf '    "%s": {\n      "signature": "%s",\n      "url": "%s"\n    }' "$label" "$sig" "$url"
  done
  echo ""
}

# ── Linux AppImage ───────────────────────────────────────────────
APPIMAGE_FILE="$BUNDLE_DIR/appimage/Gladiator_${VERSION}_amd64.AppImage"
APPIMAGE_TGZ="$BUNDLE_DIR/appimage/Gladiator_${VERSION}_amd64.AppImage.tar.gz"
if [ -f "$APPIMAGE_FILE" ]; then
  echo "[1/4] Packaging AppImage tar.gz..." >&2
  tar czf "$APPIMAGE_TGZ" -C "$BUNDLE_DIR/appimage" "Gladiator_${VERSION}_amd64.AppImage"
  sign_file "linux-x86_64" "$APPIMAGE_TGZ" || true
else
  echo "[1/4] Skipping Linux AppImage — not found" >&2
fi

# ── Windows portable zip (for auto-updater) ─────────────────────
NSIS_DIR="$BUNDLE_DIR/nsis"
PORTABLE_ZIP="$NSIS_DIR/Gladiator_${VERSION}_x64_portable.zip"
if [ -f "$PORTABLE_ZIP" ]; then
  echo "[2/4] Signing Windows portable zip..." >&2
  sign_file "windows-x86_64" "$PORTABLE_ZIP" || true
else
  # Create portable zip from the built exe and resources
  RELEASE_DIR="gui/src-tauri/target/release"
  if [ -f "$RELEASE_DIR/Gladiator.exe" ]; then
    echo "[2/4] Creating and signing Windows portable zip..." >&2
    mkdir -p "$NSIS_DIR"
    (cd "$RELEASE_DIR" && 7z a -tzip "$NSIS_DIR/Gladiator_${VERSION}_x64_portable.zip" \
      Gladiator.exe -mx9) || true
    sign_file "windows-x86_64" "$PORTABLE_ZIP" || true
  else
    echo "[2/4] Skipping Windows portable zip — Gladiator.exe not found" >&2
  fi
fi

# ── macOS Intel DMG ──────────────────────────────────────────────
DMG_X64="$BUNDLE_DIR/dmg/Gladiator_${VERSION}_x64.dmg"
if [ -f "$DMG_X64" ]; then
  echo "[3/4] Signing macOS Intel DMG..." >&2
  sign_file "darwin-x86_64" "$DMG_X64" || true
else
  echo "[3/4] Skipping macOS Intel DMG — not found" >&2
fi

# ── macOS Apple Silicon DMG ──────────────────────────────────────
DMG_AARCH64="$BUNDLE_DIR/dmg/Gladiator_${VERSION}_aarch64.dmg"
if [ -f "$DMG_AARCH64" ]; then
  echo "[4/4] Signing macOS Apple Silicon DMG..." >&2
  sign_file "darwin-aarch64" "$DMG_AARCH64" || true
else
  echo "[4/4] Skipping macOS Apple Silicon DMG — not found" >&2
fi

# ── Generate latest.json ────────────────────────────────────────
NOTES_RAW=""
if [ -n "$NOTES_FILE" ] && [ -f "$NOTES_FILE" ]; then
  NOTES_RAW=$(cat "$NOTES_FILE")
fi
NOTES_JSON=$(printf '%s' "$NOTES_RAW" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

PUB_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PLATFORMS=$(build_platform_json)

cat <<JSON
{
  "version": "${VERSION}",
  "notes": ${NOTES_JSON},
  "pub_date": "${PUB_DATE}",
  "platforms": {
${PLATFORMS}
  }
}
JSON

echo "" >&2
echo "Done! Save the latest.json output above and upload it as a release asset." >&2
echo "Also upload any .tar.gz files created in the bundle directories." >&2

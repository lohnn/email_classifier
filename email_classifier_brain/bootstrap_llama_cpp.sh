#!/usr/bin/env bash
# bootstrap_llama_cpp.sh — ONE-TIME llama.cpp install on the Pi.
#
# Run on the Pi as root:   sudo ./bootstrap_llama_cpp.sh
# (from email_classifier_brain/ — the units/ sibling dir is used)
#
# Steps:
#   1) fetch the FRESHEST ubuntu-arm64 llama.cpp nightly from the RELEASES
#      LIST (GitHub API; NEVER a release-notes link — those rot), capped
#      effort, degrade to manual instructions on failure;
#   2) install llama-server to /home/email_bot/llama.cpp/ (DEPLOY.md path);
#   3) install the systemd units (this dir's units/ + email-embedder.service
#      if not already present) + daemon-reload;
#   4) create the model dir. The GGUF itself flows afterwards via the
#      upload/install endpoints: bootstrap once → the deploy operator uploads via
#      /admin/upload-model-artifact + /admin/install-model-artifact → the
#      path-unit auto-reloads the embedder. Zero-touch from here.
#
# Never touches API keys or any other service. Degrades with clear messages.

set -uo pipefail

EMAIL_BOT_HOME="${EMAIL_BOT_HOME:-/home/email_bot}"
LLAMA_BIN_DIR="$EMAIL_BOT_HOME/llama.cpp"
MODEL_DIR="${EMAIL_BOT_HOME}/models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_TAGS_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
EFFORT_SECONDS=120   # fetch effort cap; degrade to manual instructions after

log()  { echo "[$(date -u +%H:%M:%SZ)] $*"; }
fail() { log "FAILED: $*"; }
need_root() { [ "$(id -u)" = "0" ] || { fail "run as root (sudo $0)"; exit 1; }; }

# --- 1. newest ubuntu-arm64 nightly from the RELEASES LIST -----------------
# The rot trap: llama.cpp release NOTES link to the tag the notes describe,
# which may carry no arm64 asset. The releases LIST is searched newest-first
# for the first ENTRY whose assets contain an ubuntuArm64 build.
fetch_latest_arm64_url() {
  local deadline=$(( $(date +%s) + EFFORT_SECONDS ))
  local page=1 final_url=""
  while [ $page -le 3 ]; do               # effort cap: 3 pages max
    local body
    if ! body=$(curl -fsS -m 20 "${API_TAGS_URL}&page=${page}"); then
      log "GitHub releases list unreachable (page $page)"; page=$(( page + 1 )); continue
    fi
    # asset URLs newest release first; arm64 ubuntu builds match this name
    local hits=""
    hits=$(echo "$body" | grep -o '"browser_download_url": *"[^"]*ubuntu-arm64[^"]*\.zip"' \
           | sed 's/.*"\(https[^"]*\)"/\1/')
    for url in $hits; do
      echo "$url"; return 0
    done
    [ "$(date +%s)" -ge "$deadline" ] && { log "effort cap reached"; return 2; }
    page=$(( page + 1 ))
  done
  return 2
}

need_root

log "== 1/4 llama.cpp: fetching the newest ubuntu-arm64 nightly =="
ASSET_URL="$(fetch_latest_arm64_url)"
if [ -z "$ASSET_URL" ]; then
  cat <<'MANUAL'
Could not resolve an ubuntu-arm64 nightly automatically (rate limit / offline
/ renamed assets). Manual fallback:
  1. open https://github.com/ggml-org/llama.cpp/releases — NEWEST FIRST
  2. pick any recent release whose Assets contain '..-bin-ubuntu-arm64.zip'
     (open the releases LIST, not a pinned release-notes URL — those rot)
  3. unzip it, then:  cp <dir>/llama-server /home/email_bot/llama.cpp/
  4. re-run this script — it skips completed steps.
MANUAL
  exit 2
fi
log "asset: $ASSET_URL"
mkdir -p "$LLAMA_BIN_DIR" /tmp/llama-bootstrap
if ! curl -fL -m 300 -o /tmp/llama-bootstrap/arm64.zip "$ASSET_URL"; then
  fail "download failed ($ASSET_URL)"; exit 2
fi
unzip -o -q /tmp/llama-bootstrap/arm64.zip -d /tmp/llama-bootstrap/pkg
BIN_SERVER=$(find /tmp/llama-bootstrap/pkg -name llama-server -type f | head -1)
if [ -z "$BIN_SERVER" ]; then
  fail "llama-server not found inside the asset — inspect /tmp/llama-bootstrap/pkg"
  exit 2
fi
install -m 755 "$BIN_SERVER" "$LLAMA_BIN_DIR/llama-server"
log "installed $LLAMA_BIN_DIR/llama-server"
rm -rf /tmp/llama-bootstrap

log "== 2/4 model dir: $MODEL_DIR =="
# The GGUF does NOT flow through here: it arrives via the brain's
# /admin/upload-model-artifact + /admin/install-model-artifact endpoints
# (sha256-verified, Tailscale-only, staged→atomic install to MODEL_FILE).
mkdir -p "$MODEL_DIR"
chown email_bot:email_bot "$MODEL_DIR" 2>/dev/null || log "(could not chown $MODEL_DIR to email_bot — adjust if the service runs as that user)"

log "== 3/4 systemd units =="
for unit in "units/email-embedder-reload.path" "units/email-embedder-reload.service"; do
  src="$SCRIPT_DIR/$unit"
  install -m 644 "$src" "/etc/systemd/system/$(basename "$unit")" \
    && log "installed $(basename "$unit")" \
    || { fail "unit install failed: $src"; exit 2; }
done
UNIT_EMBEDDER="/etc/systemd/system/email-embedder.service"
if [ -f "$UNIT_EMBEDDER" ]; then
  log "email-embedder.service already present — left untouched"
else
  install -m 644 "$SCRIPT_DIR/email-embedder.service" "$UNIT_EMBEDDER" \
    && log "installed email-embedder.service"
fi
systemctl daemon-reload && log "daemon-reloaded"

systemctl enable --now email-embedder-reload.path \
  && log "path watcher enabled: new GGUF at $MODEL_DIR/merged_text_q8_0.gguf auto-reloads the embedder"

log "== 4/4 done =="
cat <<'NEXT'
NEXT STEPS (zero-touch model updates):
  1. ensure the brain's .env has EMBEDDER_BACKEND=llama_cpp and
     LLAMA_MODEL_FILE pointing at this model dir ($MODEL_DIR/merged_text_q8_0.gguf).
  2. the deploy operator uploads the GGUF via the brain's
     /admin/upload-model-artifact + /admin/install-model-artifact
     (Tailscale-only, sha256-checked, atomic install).
  3. the path unit above restarts the embedder automatically. Done.
Troubleshoot:  journalctl -u email-embedder  |  /admin/retrain-status
NEXT

#!/bin/bash
# =============================================================================
# build_app.sh — Retrain Orchestrator, Mac-side build (WI-105 P1)
#
# Xcode COMMAND LINE TOOLS ONLY — no full Xcode needed.
#   swift build  →  .app bundle assembly  →  codesign with app-sandbox
#
# Run it on the Mac:
#   cd retrain-orchestrator && ./build_app.sh
# Paste ANY error output back to the coordinator verbatim.
#
# Optional env:
#   CODESIGN_IDENTITY="Developer ID Application: …"  (default: ad-hoc "-")
#   RUN_SWIFT_TEST=1   also run the Swift unit tests before building
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

EXE="RetrainOrchestrator"
APP="$EXE.app"
BUNDLE_ID="se.lohnn.retrain-orchestrator"  # operator decision 2026-10-29: se.lohnn.* domain
ENTITLEMENTS="retrain-orchestrator.entitlements"
IDENTITY="${CODESIGN_IDENTITY:--}"

command -v swift >/dev/null || { echo "ERROR: 'swift' not found — install Xcode Command Line Tools: xcode-select --install" >&2; exit 1; }

if [[ "${RUN_SWIFT_TEST:-0}" == "1" ]]; then
    echo "==> swift test (unit tests: trigger policy, ledger, erase)"
    swift test
fi

echo "==> swift build (release)"
swift build -c release

echo "==> assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp ".build/release/$EXE" "$APP/Contents/MacOS/$EXE"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Retrain Orchestrator</string>
    <key>CFBundleDisplayName</key><string>Retrain Orchestrator</string>
    <key>CFBundleExecutable</key><string>$EXE</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHumanReadableCopyright</key><string>Personal use — jarvis retraining loop.</string>
</dict>
</plist>
PLIST

# LSUIElement=true → menu bar only, no Dock icon (design §3).
# Entitlements: app-sandbox + outbound network (P2 pre-flight/mirror POST).
# No read/write grants: the footprint is the sandbox container dir only.

echo "==> codesign (ad-hoc by default; sandbox entitlements)"
/usr/bin/codesign --force --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$IDENTITY" \
    "$APP"

echo "==> verify"
/usr/bin/codesign --verify --strict --verbose=2 "$APP"
spctl --assess --type execute "$APP" 2>/dev/null || \
    echo "note: Gatekeeper refuses ad-hoc local builds — expected; launching locally is unaffected (no quarantine attribute on locally-built apps)"

# T1 root-cause guard (2026-10-28): 'codesign --verify' certifies signature
# validity, NOT that the entitlements blob made it into the signature. An
# ad-hoc app without its app-sandbox entitlement runs UNSANDBOXED and its
# state lands in ~/Library/Application Support instead of the container
# dir — which surfaced as "the Erase target folder is empty" at runtime.
# Assert presence here so that failure mode dies at build time.
echo "==> assert sandbox entitlements are embedded (assert, not imply)"
ENT_DUMP="$(/usr/bin/codesign -d --entitlements - "$APP" 2>&1 || true)"
if ! printf '%s' "$ENT_DUMP" | /usr/bin/grep -aq "app-sandbox"; then
    echo "FATAL: no app-sandbox entitlement found in the signed bundle." >&2
    echo "       The app would run unsandboxed and write OUTSIDE its container dir." >&2
    echo "--- codesign -d --entitlements - gave: ---" >&2
    printf '%s\n' "$ENT_DUMP" >&2
    exit 1
fi
printf '%s\n' "$ENT_DUMP" | sed 's/^/    /'

cat <<DONE

BUILD COMPLETE (compile-side — runtime acceptance still needs your eyes):

  1. drag '$APP' into /Applications
  2. open it: it should appear in the MENU BAR (no Dock icon)
  3. check menu: Run now / Settings… / Open log / Erase all data… / Quit
  4. Settings → change the threshold → Save → quit & relaunch → value must persist
  5. Erase all data… → confirm → ~/Library/Containers/$BUNDLE_ID/Data/Library/
     Application Support/RetrainOrchestrator/ must be empty again
  6. report what you saw (and any console errors from step 1) back

DONE

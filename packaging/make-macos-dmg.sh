#!/usr/bin/env bash
# Wrap a HertaStation.app into a distributable DMG (macOS only).
#
#   ./packaging/make-macos-dmg.sh                    # auto-find the .app
#   ./packaging/make-macos-dmg.sh build/macos-arm64/HertaStation.app
#   OUT=dist/HertaStation-1.1.0-arm64.dmg ./packaging/make-macos-dmg.sh
#
# The app is ad-hoc signed by build.sh; for distribution outside your own Macs
# you still need a Developer ID certificate and notarisation.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="${1:-}"
if [ -z "$APP" ]; then
  for candidate in dist/HertaStation.app build/macos-arm64/HertaStation.app \
                   build/macos-x64/HertaStation.app; do
    [ -d "$candidate" ] && { APP="$candidate"; break; }
  done
fi
[ -n "$APP" ] && [ -d "$APP" ] || {
  echo "no HertaStation.app found; run: python tools/make_dist.py --platform macos-arm64" >&2
  exit 1
}

OUT="${OUT:-dist/$(basename "$APP" .app).dmg}"
mkdir -p "$(dirname "$OUT")"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

hdiutil create -volname "Herta Station" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null

echo "built: $OUT"
echo "note : right-click → Open the first time (ad-hoc signature, not notarised)"

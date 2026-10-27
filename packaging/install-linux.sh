#!/usr/bin/env bash
# Install a built Linux binary into the user's desktop environment.
#
#   ./packaging/install-linux.sh                # per-user (~/.local)
#   sudo ./packaging/install-linux.sh --system  # /usr/local, all users
#
# Expects dist/HertaStation (built by ./build.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="dist/HertaStation"
[ -f "$BIN" ] || { echo "build first: ./build.sh" >&2; exit 1; }

if [ "${1:-}" = "--system" ]; then
  PREFIX="/usr/local"
  APPS="/usr/share/applications"
  ICONS="/usr/share/icons/hicolor/scalable/apps"
else
  PREFIX="$HOME/.local"
  APPS="$HOME/.local/share/applications"
  ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"
fi

install -d "$PREFIX/bin" "$APPS" "$ICONS"
install -m 755 "$BIN" "$PREFIX/bin/HertaStation"
install -m 644 packaging/herta-station.svg "$ICONS/herta-station.svg"
install -m 644 packaging/herta-station.desktop "$APPS/herta-station.desktop"

if [ "${1:-}" != "--system" ] && [ "$PREFIX" = "$HOME/.local" ]; then
  # Point the desktop entry at the per-user binary.
  sed -i "s#^Exec=.*#Exec=$PREFIX/bin/HertaStation#" "$APPS/herta-station.desktop"
fi

command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "$APPS" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
  gtk-update-icon-cache -f -t "$(dirname "$ICONS")" >/dev/null 2>&1 || true

cat <<EOF
installed:
  $PREFIX/bin/HertaStation
  $APPS/herta-station.desktop
  $ICONS/herta-station.svg

run:  HertaStation            (or launch "Herta Space Station" from the menu)
UI  : http://127.0.0.1:8899   API: http://0.0.0.0:11433/v1
EOF

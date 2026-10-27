#!/usr/bin/env bash
# Build Herta Station on Linux or macOS.
#
#   ./build.sh              # bundle the native window when a backend is available
#   ./build.sh --no-deps    # reuse the existing virtualenv
#   ./build.sh --qt         # Linux: bundle QtWebEngine so no GTK packages are needed
#   ./build.sh --headless   # Linux: no webview at all, browser fallback only
#
# Output
#   Linux : dist/HertaStation          (single file)
#   macOS : dist/HertaStation.app      (bundle, ad-hoc signed)
#
# Env
#   PYTHON=python3.12       interpreter used for the virtualenv
#   HERTA_UNIVERSAL=1       macOS: build universal2 (needs a universal2 Python)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv-unix"
DEPS=1
GUI="auto"

for arg in "$@"; do
  case "$arg" in
    --no-deps) DEPS=0 ;;
    --qt) GUI="qt" ;;
    --headless) GUI="none" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

OS="$(uname -s)"
case "$OS" in
  Linux|Darwin) SEP=":" ;;
  *) echo "this script is for Linux/macOS; use build.ps1 on Windows" >&2; exit 1 ;;
esac

echo "==> python: $("$PY" -V 2>&1)  machine: $(uname -m)"
if [ ! -d "$VENV" ]; then
  echo "==> creating $VENV"
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"

# Qt bundling is a pip-level choice, so it needs the install step even with --no-deps.
if [ "$GUI" = "qt" ] && [ "$DEPS" = "0" ]; then
  if ! "$VENV/bin/python" -c "import PyQt5.QtWebEngineWidgets" >/dev/null 2>&1; then
    echo "==> --qt needs PyQtWebEngine; installing it into $VENV"
    $PIP install "pywebview[qt]>=5.0"
  fi
fi

if [ "$DEPS" = "1" ]; then
  echo "==> installing dependencies"
  $PIP install --upgrade pip >/dev/null
  if [ "$GUI" = "qt" ] || [ "$OS" = "Linux" ]; then
    $PIP install -r requirements-linux.txt
  else
    $PIP install -r requirements.txt
  fi
fi

PYI="$VENV/bin/pyinstaller"
[ -x "$PYI" ] || { echo "pyinstaller missing in $VENV (drop --no-deps)" >&2; exit 1; }

if [ "$GUI" = "none" ]; then
  # Ship without a webview backend: the app falls back to the system browser.
  ARGS=(--noconfirm --clean --name HertaStation
        --add-data "ui.html$SEP." --paths .
        --distpath dist --workpath build/work --specpath build)
else
  ARGS=(--noconfirm --clean --windowed --name HertaStation
        --add-data "ui.html$SEP." --paths .
        --distpath dist --workpath build/work --specpath build
        --collect-all webview)
fi

if [ "$OS" = "Darwin" ]; then
  # A universal2 build needs a universal2 interpreter; opt in explicitly.
  if [ "${HERTA_UNIVERSAL:-0}" = "1" ]; then
    ARGS+=(--target-arch universal2)
  fi
  ARGS+=(--osx-bundle-identifier com.herta.station)
  [ -f assets/herta.icns ] && ARGS+=(--icon assets/herta.icns)
else
  [ -f assets/herta.png ] && ARGS+=(--icon assets/herta.png)
fi

echo "==> building (gui=$GUI)"
"$PYI" "${ARGS[@]}" herta_app.py

if [ "$OS" = "Darwin" ]; then
  codesign --force --deep --sign - dist/HertaStation.app >/dev/null 2>&1 || true
  echo "built: dist/HertaStation.app"
  echo "run  : open dist/HertaStation.app     (or: dist/HertaStation.app/Contents/MacOS/HertaStation)"
else
  chmod +x dist/HertaStation
  echo "built: dist/HertaStation"
  echo "run  : ./dist/HertaStation"
fi

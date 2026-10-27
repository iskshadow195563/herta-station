#!/usr/bin/env bash
# Run Herta Station from source on Linux/macOS (no packaging step).
#
#   ./start.sh                 # native window, falls back to the browser
#   ./start.sh --browser       # force the browser
#   ./start.sh --check         # print the detected GPU / paths
#
# Uses .venv-unix when it exists, otherwise the system python.
set -euo pipefail
cd "$(dirname "$0")"
PY="python3"
[ -x ".venv-unix/bin/python" ] && PY=".venv-unix/bin/python"
exec "$PY" herta_app.py "$@"

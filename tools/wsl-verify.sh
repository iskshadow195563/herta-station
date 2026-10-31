#!/bin/bash
# Full Linux verification of the .deb: start the server, check the preset tab,
# download the smallest curated model through the bundled herta-rs, verify it.
set -u
export HERTA_MODELS_DIR="$HOME/mtest"
export HERTA_UI_PORT=8903
mkdir -p "$HERTA_MODELS_DIR"
rm -f "$HERTA_MODELS_DIR"/*/*.gguf "$HERTA_MODELS_DIR"/*/*.part* 2>/dev/null

herta-station --serve >/tmp/hs.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do
  curl -s --max-time 2 "http://127.0.0.1:8903/api/status" >/dev/null 2>&1 && break
  sleep 1
done

echo "--- presets ---"
curl -s --max-time 5 http://127.0.0.1:8903/api/presets | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("ok", d["ok"], "rust", d["rust"] or "(python fallback)", "items", len(d["items"]))
'

echo "--- download ---"
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"id":"qwen3-1_7b-q4"}' http://127.0.0.1:8903/api/preset/download
echo
for i in $(seq 1 40); do
  S=$(curl -s http://127.0.0.1:8903/api/download/status)
  echo "$S" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ", d["percent"], "active" if d["active"] else "idle", repr(d["error"])[:50])'
  echo "$S" | grep -q '"active": false' && break
  sleep 3
done

echo "--- files ---"
ls -la "$HERTA_MODELS_DIR"/*/ 2>/dev/null | tail -3
echo "--- library ---"
curl -s http://127.0.0.1:8903/api/models | python3 -c '
import json,sys
d=json.load(sys.stdin)
for i in d["items"]:
    print("  ", i["name"], i["gb"], "ACTIVE" if i["active"] else "")
'
echo "--- cleanup ---"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo done

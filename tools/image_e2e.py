"""Live end-to-end exercise of the image engine over the console HTTP API.

Usage:  python tools/image_e2e.py [--base http://127.0.0.1:8899] [--skip-start]

Steps: status -> recommend -> models -> start (spawns ComfyUI, opens browser)
       -> status -> selftest (one real 512x512 image) -> stop (VRAM freed).
Writes a transcript to tools/image_e2e.log and exits non-zero on the first failure.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


def call(base, path, payload=None, timeout=600, post=False):
    """GET unless post=True or a payload is given (start/stop take no body)."""
    url = base.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif post:
        data = b"{}"
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if (post or payload is not None) else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def section(title):
    log("")
    log("=" * 62)
    log(title)
    log("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8899")
    ap.add_argument("--skip-start", action="store_true")
    args = ap.parse_args()
    base = args.base
    t0 = time.time()

    section("1. GET /api/image/status (engine down)")
    st = call(base, "/api/image/status")
    for key in ("phase", "url", "port", "device", "device_name", "engine", "root",
                "engine_installed", "gguf_nodes", "queue", "vram", "error"):
        log(f"  {key:16} = {st.get(key)!r}")
    if st.get("error"):
        log("[fail] status reported an error")
        return 1
    if not st.get("engine_installed"):
        log("[fail] engine not detected - set image.comfyDir or HERTA_COMFY_DIR")
        return 1

    section("2. GET /api/image/recommend (GPU-aware suggestions)")
    rec = call(base, "/api/image/recommend")
    log(f"  gpu = {rec.get('gpu')} ({rec.get('gpu_total_gb')} GB)  vram_limit={rec.get('vram_limit_gb')} GB")
    for pick in rec.get("picks") or []:
        log(f"  [{pick['tier']}] {pick['family']:12} {pick['params']:>6}  "
            f"min {pick['min_vram_gb']} GB  langs={pick['lang_labels']}")
        log(f"        {pick['reason']}")

    section("3. GET /api/image/models (local weights)")
    lib = call(base, "/api/image/models")
    log(f"  dir={lib.get('dir')}  total={lib.get('total_gb')} GB  items={len(lib.get('items') or [])}")
    for it in (lib.get("items") or [])[:8]:
        log(f"  - {it['name']:44} {it['size_gb']:>5} GB  {it['role']:10} "
            f"{it['family']:12} {it.get('lang_labels')}")

    if not args.skip_start:
        section("4. POST /api/image/start (spawn ComfyUI + open browser)")
        t = time.time()
        res = call(base, "/api/image/start", post=True)
        log(f"  took {time.time() - t:.1f}s -> ok={res.get('ok')} url={res.get('url')} "
            f"device={res.get('device')} opened={res.get('opened')} reused={res.get('reused')}")
        if not res.get("ok"):
            log(f"[fail] start: {res.get('message')}")
            return 1

    section("5. GET /api/image/status (engine up)")
    st = call(base, "/api/image/status")
    log(f"  phase={st.get('phase')} device={st.get('device')} name={st.get('device_name')} "
        f"ver={st.get('engine_version')} torch={st.get('torch')} pid={st.get('pid')} "
        f"port={st.get('port')} uptime={st.get('uptime')}s")
    log(f"  vram={st.get('vram')}  queue={st.get('queue')}  gguf_nodes={st.get('gguf_nodes')}")
    if st.get("phase") != "running":
        log("[fail] engine is not running after start")
        return 1
    if str(st.get("device")).lower() != "cuda":
        log("[warn] device is not cuda - it will be very slow")

    section("6. POST /api/image/selftest (real 512x512 sample)")
    t = time.time()
    res = call(base, "/api/image/selftest", payload={}, timeout=900)
    log(f"  took {time.time() - t:.1f}s -> ok={res.get('ok')} msg={res.get('message')}")
    if res.get("file"):
        log(f"  file={res.get('file')}  bytes={res.get('bytes')}  gen={res.get('seconds')}s")
    if not res.get("ok"):
        log(f"[fail] selftest: {res.get('message')}")
        return 1

    section("7. POST /api/image/stop (release VRAM)")
    t = time.time()
    res = call(base, "/api/image/stop", post=True)
    log(f"  took {time.time() - t:.1f}s -> ok={res.get('ok')} freed={res.get('freed_gb')} GB")
    if not res.get("ok"):
        log(f"[fail] stop: {res.get('message')}")
        return 1
    st = call(base, "/api/image/status")
    log(f"  after stop: phase={st.get('phase')} vram={st.get('vram')}")
    if st.get("phase") != "idle":
        log("[fail] engine still reachable after stop")
        return 1

    section(f"ALL GREEN in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    code = main()
    try:
        with open("tools/image_e2e.log", "w", encoding="utf-8") as fh:
            fh.write("\n".join(LOG) + "\n")
    except OSError:
        pass
    sys.exit(code)

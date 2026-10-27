# -*- coding: utf-8 -*-
"""Herta Space Station - llama.cpp switchboard.

Small stdlib-only control plane for a local llama-server instance:
  GET  /              -> the Herta-styled control UI
  GET  /api/status    -> process / VRAM / CUDA telemetry
  POST /api/start     -> launch llama-server with the Qwen3.8-27B Q3_K_XL weights
  POST /api/stop      -> kill the server tree and release VRAM
  POST /api/purge     -> also unload every Ollama model holding VRAM
  POST /api/selftest  -> one tiny chat completion against the running server
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
import models
import settings as appsettings
import anthropic_proxy
import hostos
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ui.html lives next to the script, or inside the PyInstaller bundle
RES_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
_UI_BYTES = None


def ui_html():
    """Read ui.html, including from a zipapp/pyz where it is inside the archive."""
    global _UI_BYTES
    if _UI_BYTES is not None:
        return _UI_BYTES
    for path in (os.path.join(RES_DIR, "ui.html"),):
        try:
            with open(path, "rb") as fh:
                _UI_BYTES = fh.read()
                return _UI_BYTES
        except OSError:
            pass
    archive = sys.argv[0] if zipfile.is_zipfile(sys.argv[0]) else None
    if archive:
        with zipfile.ZipFile(archive) as zf:
            _UI_BYTES = zf.read("ui.html")
            return _UI_BYTES
    raise FileNotFoundError("ui.html missing")
# logs / pid must live beside the executable, never inside the temp bundle
DATA_DIR = appsettings.DATA_DIR
EXE = hostos.default_llama_exe()
MODEL = hostos.default_model_path()
MMPROJ = hostos.default_mmproj_path()
ALIAS = os.environ.get("HERTA_ALIAS", "")
HOST = "127.0.0.1"                               # control plane / probes (never exposed)

CONF = appsettings.load()                        # settings.json wins over env defaults
BIND = CONF["server"]["bind"]                    # llama-server listen address
API_KEY = CONF["server"]["apiKey"]               # optional bearer token for LAN clients
PORT = CONF["server"]["port"]
CTX = CONF["server"]["ctx"]
PARALLEL = CONF["server"]["parallel"]
FA = os.environ.get("HERTA_FA", "on")
CTK = CONF["server"]["ctk"]
CTV = CONF["server"]["ctv"]
# the vision projector costs ~0.9 GB; off by default so 64k context fits in 16 GB
MMPROJ_ON = os.environ.get("HERTA_MMPROJ", "0").lower() in ("1", "on", "true", "yes")
# cap on thinking tokens per reply; -1 disables the cap
REASONING_BUDGET = int(os.environ.get("HERTA_REASONING_BUDGET", "4096"))
NGL = CONF["server"]["ngl"]
UI_PORT = int(os.environ.get("HERTA_UI_PORT", "8899"))   # override for tests

LOG_PATH = os.path.join(DATA_DIR, "llama-server.log")
PID_PATH = os.path.join(DATA_DIR, "llama-server.pid")

_lock = threading.Lock()
_proc = None
_started_at = None
_last_error = ""
_last_error_code = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _kill_tree(pid):
    """Kill a process and its children (taskkill on Windows, SIGTERM/KILL on POSIX)."""
    return hostos.kill_tree(pid)


def _run(args, timeout=15):
    """Run a console tool without ever blocking on a lingering grandchild."""
    return hostos.run(args, timeout=timeout)


_vram_cache = {"at": 0.0, "value": None}
_lan_cache = {"at": 0.0, "value": ""}


def _vram_cached(ttl=2.0):
    """nvidia-smi costs ~150 ms; the dashboard polls every 1.5 s, so memoise it."""
    now = time.time()
    if _vram_cache["value"] is not None and now - _vram_cache["at"] < ttl:
        return _vram_cache["value"]
    value = _vram()
    _vram_cache.update(at=now, value=value)
    return value


def _lan_ip_cached(ttl=60.0):
    now = time.time()
    if _lan_cache["value"] and now - _lan_cache["at"] < ttl:
        return _lan_cache["value"]
    value = _lan_ip()
    _lan_cache.update(at=now, value=value)
    return value


def _vram():
    """Used/total VRAM in MiB. nvidia-smi, AMD sysfs, or Apple unified memory."""
    return hostos.vram_usage(runner=_run)


def _lan_ip():
    """Best-effort primary LAN address of this machine."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return HOST
    finally:
        sock.close()


def _auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _health():
    try:
        req = urllib.request.Request(f"http://{HOST}:{PORT}/health", headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
    except Exception:
        return False


def _port_pid(port=None):
    """Find the PID listening on a local TCP port (netstat / ss / lsof / /proc)."""
    port = PORT if port is None else port
    return hostos.listener_pid(port, runner=_run)


def _tail(n=60):
    """Bound each dashboard read to 64 KiB even when a model writes a huge log."""
    if n <= 0:
        return []
    try:
        with open(LOG_PATH, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            start = max(0, fh.tell() - 65536)
            fh.seek(start)
            raw = fh.read(65536)
        if start and b"\n" in raw:
            raw = raw.split(b"\n", 1)[1]
        return raw.decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _tps(lines):
    """Last tokens/second figure reported by llama.cpp."""
    for line in reversed(lines):
        m = re.search(r"([\d.]+)\s+tokens per second", line)
        if m:
            return float(m.group(1))
    return None


def _pid():
    """PID of the running llama-server.

    The child handle wins. Otherwise trust the PID file only when it still owns
    the listening port *and* really is llama-server; if the file is missing, adopt
    the listener after the same executable check (a console restart must not make
    the app blind to a model it started earlier).
    """
    if _proc is not None and _proc.poll() is None:
        return _proc.pid
    listening = _port_pid()
    if listening is None:
        return None
    try:
        with open(PID_PATH, "r", encoding="utf-8") as fh:
            saved = int(fh.read().strip())
    except Exception:
        saved = None
    if saved is not None and saved == listening and _matches_exe(saved):
        return saved
    return listening if _matches_exe(listening) else None


def _process_image(pid):
    """Full image path of a live process, or None (Win32 / /proc / ps)."""
    return hostos.process_image(pid)


def _matches_exe(pid, expected=None):
    """True when the process image really is llama-server (or ``expected``)."""
    image = _process_image(pid)
    if not image:
        return False
    want = expected or EXE
    return (os.path.normcase(os.path.realpath(image))
            == os.path.normcase(os.path.realpath(want)))


def _alive():
    """Cheap liveness check: the child handle first, then the listening port."""
    if _proc is not None and _proc.poll() is None:
        return True
    return _pid() is not None


def _failure(code, message):
    global _last_error, _last_error_code
    _last_error, _last_error_code = message, code
    return {"ok": False, "message": message, "error_code": code}


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
def current_model():
    """Active GGUF: library selection, env default, else the first local file."""
    path = models.active_model_path() or MODEL
    if path and os.path.isfile(path):
        return path
    try:                                   # nothing selected yet: pick a local weight
        for item in models.scan_local(0):
            if not item.get("mmproj"):
                return item["path"]
    except Exception:
        pass
    return path


def current_alias():
    if ALIAS:
        return ALIAS
    name = os.path.splitext(os.path.basename(current_model()))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "local-model"


def start_model():
    global _proc, _started_at, _last_error, _last_error_code
    with _lock:
        alive = _alive()
        if alive and _health():
            return {"ok": True, "message": "模型已在運行中"}
        if alive:
            _started_at = _started_at or time.time()
            return {"ok": True, "message": "載入程序仍在啟動，請稍候"}

        _apply_server_settings(CONF["server"])
        if _port_pid() is not None:
            return _failure("port_in_use", f"連接埠 {PORT} 已被其他程序使用")
        if not os.path.isfile(EXE):
            return _failure("missing_executable",
                            f"找不到 llama-server：{EXE}（可用 HERTA_LLAMA_EXE 指定）")
        model_path = current_model()
        if not os.path.isfile(model_path):
            return _failure("missing_model", f"找不到模型權重: {model_path}")

        args = [
            EXE, "-m", model_path,
            "-a", current_alias(),
            "--host", BIND, "--port", str(PORT),
            "-c", str(CTX), "-ngl", str(NGL),
            "-np", str(PARALLEL),
            "-fa", FA,
            "--jinja",
        ]
        if API_KEY:
            args += ["--api-key", API_KEY]
        # hard cap on thinking tokens: without it a long "xhigh" reasoning pass can eat the
        # whole max_tokens budget and return a reasoning-only reply (no text, no tool call),
        # which is what breaks DSH sessions. -1 = unlimited.
        if REASONING_BUDGET >= 0:
            args += ["--reasoning-budget", str(REASONING_BUDGET)]
        if MMPROJ_ON:
            args += ["--mmproj", MMPROJ]
        if CTK:
            args += ["-ctk", CTK]
        if CTV:
            args += ["-ctv", CTV]
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            display_args = list(args)
            if "--api-key" in display_args:
                display_args[display_args.index("--api-key") + 1] = "***"
            with open(LOG_PATH, "w", encoding="utf-8", errors="replace") as log:
                log.write("$ " + hostos.cmdline(display_args) + "\n")
                log.flush()
                _proc = subprocess.Popen(
                    args, cwd=os.path.dirname(EXE) or None, stdout=log,
                    stderr=subprocess.STDOUT, **hostos.spawn_kwargs(),
                )
            _started_at = time.time()
            with open(PID_PATH, "w", encoding="utf-8") as fh:
                fh.write(str(_proc.pid))
            _last_error = ""
            _last_error_code = ""
            start_anthropic_proxy(restart=True)
            return {"ok": True, "message": f"已啟動 llama-server (PID {_proc.pid})"}
        except Exception as exc:
            return _failure("launch_failed", f"啟動失敗: {exc}")


def stop_model():
    global _proc, _started_at
    with _lock:
        killed = []
        pid = _pid()
        if pid:
            try:
                if not _kill_tree(pid):
                    return _failure("stop_failed", "無法停止模型程序")
                if _proc is not None:
                    _proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError) as exc:
                return _failure("stop_failed", str(exc))
            killed.append(pid)
        try:
            os.remove(PID_PATH)
        except OSError:
            pass
        _proc = None
        _started_at = None
        _apply_server_settings(CONF["server"])
        start_anthropic_proxy(restart=True)
        return {"ok": True, "message": f"已停止 {len(killed)} 個程序，顯存已釋放" if killed else "目前沒有運行中的實例"}


def _llama_pids():
    """Every live llama-server PID (tasklist / pgrep)."""
    return hostos.pids_by_name("llama-server", runner=_run)


def _sweep_llama():
    """Kill llama-server processes that came from our own build (orphans included)."""
    own_dir = os.path.normcase(os.path.dirname(os.path.realpath(EXE)))
    killed = []
    for pid in _llama_pids():
        image = _process_image(pid) or ""
        if image and os.path.normcase(os.path.dirname(os.path.realpath(image))) == own_dir \
                and _kill_tree(pid):
            killed.append(pid)
    return killed


def purge_all():
    """Free VRAM: stop our model, unload Ollama, sweep orphans, report the delta."""
    gpu = _vram_cached(0) or {}
    before = gpu.get("used", 0) / 1024.0
    stopped = []
    # Ollama is optional: skip it entirely when the CLI is not installed, and keep
    # a tight budget so a busy/hung daemon can never stall the request.
    if shutil.which("ollama"):
        names = []
        out = _run(["ollama", "ps"], timeout=8)
        for line in out.splitlines()[1:]:
            token = line.split()
            if token and token[0] not in ("NAME",):
                names.append(token[0])
        for name in names[:4]:
            _run(["ollama", "stop", name], timeout=12)
            stopped.append(name)
    res = stop_model()
    swept = _sweep_llama()
    time.sleep(1.2)
    after = (_vram_cached(0) or {}).get("used", 0) / 1024.0
    freed = max(0.0, before - after)
    parts = [f"已釋放 {freed:.1f} GB 顯存（{before:.1f} → {after:.1f} GB）"]
    if stopped:
        parts.append("卸載 Ollama: " + ", ".join(stopped))
    if swept:
        parts.append(f"清理殘留 llama-server ×{len(swept)}")
    if res.get("ok") and res.get("message") and "未在執行" not in res["message"]:
        parts.append(res["message"])
    return {"ok": True, "message": " · ".join(parts), "freed_gb": round(freed, 2),
            "vram": {"used": round(after * 1024), "total": gpu.get("total", 0)}}


def selftest():
    body = json.dumps({
        "model": current_alias(),
        "messages": [{"role": "user", "content": "回覆兩個字：在線"}],
        "max_tokens": 32,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        reply = (data["choices"][0]["message"].get("content") or "").strip()
        return {"ok": True, "message": f"{time.time() - t0:.1f}s · {reply[:60]}"}
    except Exception as exc:
        return {"ok": False, "message": f"連線測試失敗: {exc}"}


# --------------------------------------------------------------------------- #
# automatic context / KV planning
# --------------------------------------------------------------------------- #
def plan_for_model(path):
    """Largest context this machine can give ``path``, plus a KV type to match."""
    return models.context_plan(path, models.gpu_total_gb(), models.read_gguf_ctx(path))


def apply_plan(plan):
    """Persist a context/KV plan; live globals refresh on the next start."""
    global CONF
    CONF = appsettings.save({"server": {"ctx": plan["ctx"], "ctk": plan["ctk"], "ctv": plan["ctv"]}})
    if not (_alive() or _health()):
        _apply_server_settings(CONF["server"])
    return {
        "ctx": CONF["server"]["ctx"], "ctk": CONF["server"]["ctk"], "ctv": CONF["server"]["ctv"],
        "ctx_train": plan.get("ctx_train", 0), "weights_gb": plan.get("weights_gb", 0),
        "vram_limit_gb": plan.get("vram_limit_gb", 0),
    }


KV_ORDER = ["f16", "q8_0", "q5_1", "q4_0"]
_fitting = {"active": False, "applied": 0}


def _wait_ready(timeout=150):
    """Block until the freshly spawned server answers /health (or the timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health():
            return True
        if _proc is not None and _proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


def _probe_tps(timeout=90):
    """Generate a few tokens and report tok/s — paging shows up here, not in VRAM."""
    body = json.dumps({
        "model": current_alias(),
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        timings = data.get("timings") or {}
        return float(timings.get("predicted_per_second") or 0.0)
    except Exception:
        return 0.0


def fit_vram(max_tries=4, wait=150):
    """Bring the loaded model back under the paging cliff.

    Trade KV cache quality first, then context length. Memory alone cannot decide
    this: 15.4 GB is healthy for one model and crawling for another, so a short
    generation probe is the real signal and VRAM only guards gross overshoot.
    """
    if not _wait_ready(wait):
        return 0
    time.sleep(1.5)                     # let the driver settle before reading VRAM
    applied = 0
    for _ in range(max_tries):
        gpu = _vram_cached(0) or {}
        used = gpu.get("used", 0) / 1024.0
        total = gpu.get("total", 0) / 1024.0
        if not total:
            return applied
        limit = models.vram_limit(total)
        tps = _probe_tps()
        paging = tps > 0 and tps < 12.0
        if used <= limit + 0.35 and not paging:
            return applied

        lower = [kv for kv in KV_ORDER[KV_ORDER.index(CTK) + 1:] if kv in models.KV_BYTES] \
            if CTK in KV_ORDER else []
        if lower:
            plan = {"ctx": CTX, "ctk": lower[0], "ctv": lower[0]}
        else:
            new_ctx = max(2048, int(CTX * 0.6) // 2048 * 2048)
            if new_ctx >= CTX:
                return applied
            plan = {"ctx": new_ctx, "ctk": CTK, "ctv": CTV}
        apply_plan(plan)
        stop_model()
        time.sleep(0.8)
        start_model()
        if not _wait_ready(wait):
            return applied
        time.sleep(1.5)
        applied = CONF["server"]["ctx"]
    return applied


def fit_async():
    """Run the VRAM fit in the background so the API answer stays instant."""
    if _fitting["active"]:
        return False

    def run():
        _fitting["active"] = True
        try:
            _fitting["applied"] = fit_vram()
        except Exception:
            _fitting["applied"] = 0
        finally:
            _fitting["active"] = False

    threading.Thread(target=run, daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# anthropic shim
# --------------------------------------------------------------------------- #
_anthropic = {"server": None, "port": 0, "error": ""}


def start_anthropic_proxy(restart=False):
    """Run the Anthropic /v1/messages shim in front of llama-server."""
    if _anthropic["server"] is not None and not restart:
        return
    if restart and _anthropic["server"] is not None:
        try:
            _anthropic["server"].shutdown()
            _anthropic["server"].server_close()
        except Exception:
            pass
        _anthropic["server"] = None
    if not CONF["server"].get("anthropicEnabled", True):
        _anthropic["port"] = 0
        return
    port = CONF["server"].get("anthropicPort", 11435)
    try:
        proxy = anthropic_proxy.AnthropicProxy(HOST, PORT, current_alias(), API_KEY)
        _anthropic["server"] = anthropic_proxy.serve(proxy, BIND, port)
        _anthropic["port"] = port
        _anthropic["error"] = ""
    except Exception as exc:
        _anthropic["server"] = None
        _anthropic["port"] = 0
        _anthropic["error"] = str(exc)


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def apply_settings(patch):
    """Persist UI/server preferences and make them the live values."""
    global CONF
    with _lock:
        running = _alive() or _health()
        before = CONF
        updated = appsettings.save(patch)
        CONF = updated
        # Keep probes, stop and the shim pointed at the *running* model until reload.
        if not running:
            _apply_server_settings(CONF["server"])
        if (before["server"] != CONF["server"]):
            start_anthropic_proxy(restart=True)
        return {"ok": True, "settings": CONF, "restart_required": running and _restart_required()}


def _apply_server_settings(srv):
    global CONF, CTX, CTK, CTV, PARALLEL, BIND, PORT, API_KEY, NGL
    CTX = srv["ctx"]
    CTK = srv["ctk"]
    CTV = srv["ctv"]
    PARALLEL = srv["parallel"]
    BIND = srv["bind"]
    PORT = srv["port"]
    API_KEY = srv["apiKey"]
    NGL = srv["ngl"]


def _restart_required():
    return any(CONF["server"][key] != value for key, value in (
        ("ctx", CTX), ("ctk", CTK), ("ctv", CTV), ("parallel", PARALLEL),
        ("bind", BIND), ("port", PORT), ("apiKey", API_KEY), ("ngl", NGL)))


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def status():
    global _last_error, _last_error_code
    lines = _tail(60)
    pid = _pid()
    alive = pid is not None
    running = alive and _health()
    if _proc is not None and _proc.poll() is not None and _started_at is not None and not _last_error:
        detail = "\n".join(lines)
        if re.search(r"out of memory|cudaMalloc.*failed|failed to allocate|unable to allocate", detail, re.I):
            _last_error_code = "out_of_memory"
        elif re.search(r"address already in use|failed to bind|bind.*failed", detail, re.I):
            _last_error_code = "port_in_use"
        else:
            _last_error_code = "model_exited"
        _last_error = f"llama-server exited ({_proc.returncode})"
    if running:
        phase = "running"
    elif alive:
        phase = "loading"
    else:
        phase = "idle"
    uptime = None
    if phase != "idle":
        base = _started_at
        if base is None:
            try:
                base = os.path.getctime(LOG_PATH)
            except OSError:
                base = None
        if base:
            uptime = int(time.time() - base)
    gpu = _vram_cached()
    lan = _lan_ip_cached()
    return {
        "phase": phase,
        "port": PORT,
        "host": HOST,
        "bind": BIND,
        "lan": lan,
        "api_key": bool(API_KEY),
        "alias": current_alias(),
        "model": os.path.basename(current_model()),
        "model_path": current_model(),
        "quant": models.quant_of(os.path.basename(current_model())),
        "models_dir": models.MODELS_DIR,
        "ctx": CTX,
        "ngl": NGL,
        "ctk": CTK,
        "ctv": CTV,
        "parallel": PARALLEL,
        "pid": pid,
        "uptime": uptime,
        "tps": _tps(lines),
        "vram": gpu,
        "log": lines[-40:],
        "error": _last_error,
        "error_code": _last_error_code,
        "restart_required": alive and _restart_required(),
        "ctx_train": models.read_gguf_ctx(current_model() or ""),
        "auto_plan": CONF["server"].get("autoPlan", True),
        "fitting": _fitting["active"],
        "endpoint": f"http://{HOST}:{PORT}/v1",
        "endpoint_lan": f"http://{lan}:{PORT}/v1",
        "anthropic_url": (f"http://{lan}:{_anthropic['port']}" if _anthropic["port"] else ""),
        "anthropic_port": _anthropic["port"],
        "anthropic_enabled": CONF["server"].get("anthropicEnabled", True),
        "anthropic_error": _anthropic["error"],
        "platform": ("windows" if hostos.IS_WIN else
                     ("macos" if hostos.IS_MAC else ("linux" if hostos.IS_LINUX else sys.platform))),
        "unified": bool(models.hardware().get("unified")),
        "llama_exe": EXE,
    }


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "HertaConsole/1.0"

    def log_message(self, *_args):
        pass

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query)
        if path in ("/", "/index.html"):
            try:
                self._send(200, ui_html(), "text/html; charset=utf-8")
            except (OSError, KeyError, zipfile.BadZipFile):
                self._send(500, b"ui.html missing", "text/plain; charset=utf-8")
        elif path == "/api/status":
            self._send(200, status())
        elif path == "/api/models":
            self._send(200, {
                "dir": models.MODELS_DIR,
                "gpu_total_gb": models.gpu_total_gb(),
                "ctx": CTX,
                "items": models.scan_local(CTX),
            })
        elif path == "/api/hf/search":
            q = (params.get("q") or [""])[0]
            try:
                self._send(200, {"ok": True, "items": models.hf_search(
                    q, params.get("limit", ["30"])[0], params.get("sort", ["downloads"])[0])})
            except Exception as exc:
                self._send(200, {"ok": False, "message": str(exc), "items": []})
        elif path == "/api/hf/files":
            repo = (params.get("repo") or [""])[0]
            try:
                self._send(200, {"ok": True, "repo": repo, "items": models.hf_files(repo, CTX)})
            except Exception as exc:
                self._send(200, {"ok": False, "message": str(exc), "items": []})
        elif path == "/api/download/status":
            self._send(200, models.download_state())
        elif path == "/api/presets":
            try:
                self._send(200, {"ok": True, "items": models.preset_state(CTX),
                                 "rust": models.RUST_BIN or "",
                                 "models_dir": models.MODELS_DIR})
            except Exception as exc:
                self._send(200, {"ok": False, "message": str(exc), "items": []})
        elif path == "/api/recommend":
            self._send(200, models.recommend(CTX))
        elif path == "/api/settings":
            self._send(200, {"ok": True, "settings": appsettings.load(),
                             "defaults": appsettings.DEFAULTS})
        else:
            self._send(404, {"ok": False, "message": "not found"})

    def do_POST(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query)
        def _select():
            target = (params.get("path") or [""])[0]
            if not target:
                return {"ok": False, "message": "缺少 path"}
            try:
                models.set_active(target)
            except OSError as exc:
                return {"ok": False, "message": f"找不到檔案: {exc}"}
            except ValueError as exc:
                return {"ok": False, "message": str(exc)}
            plan = None
            if CONF["server"].get("autoPlan", True):
                plan = apply_plan(plan_for_model(target))
            was_running = _health() or _alive()
            if was_running:
                stop_model()
                time.sleep(1.0)
                res = start_model()
                if plan and res["ok"]:
                    fit_async()          # measure after load, shrink in the background
                note = ""
                if plan:
                    note = (f" · 自動配置 ctx={plan['ctx']}"
                            f"（{models.fmt_ctx(plan['ctx'])}）KV={plan['ctk']}")
                return {"ok": res["ok"], "message": f"已切換模型並重啟：{res['message']}{note}",
                        "plan": plan}
            return {"ok": True, "message": f"已選擇 {os.path.basename(target)}", "plan": plan}

        def _dl_start():
            repo = (params.get("repo") or [""])[0]
            name = (params.get("file") or [""])[0]
            if not repo or not name:
                return {"ok": False, "message": "缺少 repo / file"}
            return models.start_download(repo, name)

        def _preset_download():
            preset_id = (params.get("id") or [""])[0]
            if not preset_id:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 0:
                    try:
                        preset_id = json.loads(self.rfile.read(n).decode("utf-8-sig")).get("id", "")
                    except Exception:
                        preset_id = ""
            if not preset_id:
                return {"ok": False, "message": "缺少 preset id", "error_code": "invalid_download"}
            return models.start_preset_download(preset_id)

        def _settings_save():
            n = int(self.headers.get("Content-Length") or 0)
            patch = {}
            if n > 0:
                try:
                    patch = json.loads(self.rfile.read(n).decode("utf-8-sig"))
                except Exception as exc:
                    return {"ok": False, "message": f"設定格式錯誤: {exc}"}
            try:
                return apply_settings(patch)
            except ValueError as exc:
                return {"ok": False, "message": str(exc), "error_code": "settings_invalid"}
            except OSError as exc:
                return {"ok": False, "message": str(exc), "error_code": "settings_write_failed"}

        def _start():
            res = start_model()
            if res.get("ok") and CONF["server"].get("autoPlan", True):
                fit_async()          # correct the VRAM footprint once weights are in
            return res

        routes = {
            "/api/start": _start,
            "/api/stop": stop_model,
            "/api/purge": purge_all,
            "/api/selftest": selftest,
            "/api/select": _select,
            "/api/settings": _settings_save,
            "/api/hf/download": _dl_start,
            "/api/preset/download": _preset_download,
            "/api/download/cancel": models.cancel_download,
            "/api/download/reset": models.reset_download,
        }
        fn = routes.get(path)
        if not fn:
            self._send(404, {"ok": False, "message": "not found"})
            return
        try:
            self._send(200, fn())
        except Exception as exc:
            self._send(500, {"ok": False, "message": str(exc)})


def main():
    srv = ThreadingHTTPServer((HOST, UI_PORT), Handler)
    print(f"Herta console -> http://{HOST}:{UI_PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

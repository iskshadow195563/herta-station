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
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import models
import settings as appsettings
import anthropic_proxy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ui.html lives next to the script, or inside the PyInstaller bundle
RES_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
# logs / pid must live beside the executable, never inside the temp bundle
DATA_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else RES_DIR
EXE = os.environ.get("HERTA_LLAMA_EXE", r"D:\llama.cpp\b11756\llama-server.exe")
MODEL = os.environ.get("HERTA_MODEL", r"D:\ollama-gguf\qwen3.8-27b\Qwen3.8-27B-UD-Q3_K_XL.gguf")
MMPROJ = os.environ.get("HERTA_MMPROJ_PATH", r"D:\ollama-gguf\qwen3.8-27b\mmproj-F16.gguf")
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
UI_PORT = 8899

LOG_PATH = os.path.join(DATA_DIR, "llama-server.log")
PID_PATH = os.path.join(DATA_DIR, "llama-server.pid")

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_lock = threading.Lock()
_proc = None
_started_at = None
_last_error = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(args, timeout=15):
    try:
        out = subprocess.run(
            args, capture_output=True, timeout=timeout, creationflags=NO_WINDOW
        )
        return out.stdout.decode("utf-8", "replace") + out.stderr.decode("utf-8", "replace")
    except Exception as exc:  # pragma: no cover
        return f"{exc}"


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
    txt = _run([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,name",
        "--format=csv,noheader,nounits",
    ], timeout=8).strip().splitlines()
    if not txt:
        return {"used": None, "total": None, "gpu": "n/a"}
    parts = [p.strip() for p in txt[0].split(",")]
    try:
        return {"used": int(parts[0]), "total": int(parts[1]), "gpu": parts[2]}
    except (ValueError, IndexError):
        return {"used": None, "total": None, "gpu": "n/a"}


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


def _port_pid(port=PORT):
    """Find the PID listening on a local TCP port (Windows netstat)."""
    out = _run(["netstat", "-ano", "-p", "TCP"], timeout=10)
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line.upper():
            tail = line.split()
            if tail and tail[-1].isdigit():
                return int(tail[-1])
    return None


def _tail(n=60):
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [ln.rstrip("\r\n") for ln in lines[-n:]]
    except FileNotFoundError:
        return []


def _tps(lines):
    """Last tokens/second figure reported by llama.cpp."""
    for line in reversed(lines):
        m = re.search(r"([\d.]+)\s+tokens per second", line)
        if m:
            return float(m.group(1))
    return None


def _pid():
    if _proc is not None and _proc.poll() is None:
        return _proc.pid
    try:
        with open(PID_PATH, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return _port_pid()


def _alive():
    """Cheap liveness check: the child handle first, then the listening port."""
    if _proc is not None and _proc.poll() is None:
        return True
    return _port_pid() is not None


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
def current_model():
    """Active GGUF: whatever the library selection points at, else the env default."""
    path = models.active_model_path() or MODEL
    return path


def current_alias():
    if ALIAS:
        return ALIAS
    name = os.path.splitext(os.path.basename(current_model()))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "local-model"


def start_model():
    global _proc, _started_at, _last_error
    with _lock:
        if _health():
            return {"ok": True, "message": "模型已在運行中"}
        if _alive():
            _started_at = _started_at or time.time()
            return {"ok": True, "message": "載入程序仍在啟動，請稍候"}

        if not os.path.exists(EXE):
            return {"ok": False, "message": f"找不到 llama-server.exe: {EXE}"}
        model_path = current_model()
        if not os.path.exists(model_path):
            return {"ok": False, "message": f"找不到模型權重: {model_path}"}

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
            log = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
            log.write("$ " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n")
            log.flush()
            _proc = subprocess.Popen(
                args,
                cwd=os.path.dirname(EXE),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW,
            )
            _started_at = time.time()
            with open(PID_PATH, "w", encoding="utf-8") as fh:
                fh.write(str(_proc.pid))
            _last_error = ""
            return {"ok": True, "message": f"已啟動 llama-server (PID {_proc.pid})"}
        except Exception as exc:
            _last_error = str(exc)
            return {"ok": False, "message": f"啟動失敗: {exc}"}


def stop_model():
    global _proc, _started_at
    with _lock:
        killed = []
        if _proc is not None and _proc.poll() is None:
            _run(["taskkill", "/PID", str(_proc.pid), "/T", "/F"])
            killed.append(_proc.pid)
        pid = _port_pid()
        if pid and pid not in killed:
            _run(["taskkill", "/PID", str(pid), "/T", "/F"])
            killed.append(pid)
        try:
            os.remove(PID_PATH)
        except OSError:
            pass
        _proc = None
        _started_at = None
        time.sleep(0.6)
        return {"ok": True, "message": f"已停止 {len(killed)} 個程序，顯存已釋放" if killed else "目前沒有運行中的實例"}


def purge_all():
    names = []
    out = _run(["ollama", "ps"], timeout=20)
    for line in out.splitlines()[1:]:
        token = line.split()
        if token and token[0] not in ("NAME",):
            names.append(token[0])
    stopped = []
    for name in names:
        _run(["ollama", "stop", name], timeout=40)
        stopped.append(name)
    res = stop_model()
    return {"ok": True, "message": f"已卸載 Ollama: {', '.join(stopped) if stopped else '無'}；{res['message']}"}


def selftest():
    body = json.dumps({
        "model": ALIAS,
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
    global CONF, CTX, CTK, CTV, PARALLEL, BIND, PORT, API_KEY, NGL
    before = CONF["server"]
    CONF = appsettings.save(patch or {})
    srv = CONF["server"]
    CTX = srv["ctx"]
    CTK = srv["ctk"]
    CTV = srv["ctv"]
    PARALLEL = srv["parallel"]
    BIND = srv["bind"]
    PORT = srv["port"]
    API_KEY = srv["apiKey"]
    NGL = srv["ngl"]
    if (before.get("anthropicEnabled"), before.get("anthropicPort"), before.get("apiKey")) != \
       (srv["anthropicEnabled"], srv["anthropicPort"], srv["apiKey"]):
        start_anthropic_proxy(restart=True)
    running = _health() or _alive()
    return {"ok": True, "settings": CONF, "restart_required": running}


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def status():
    lines = _tail(60)
    running = _health()
    alive = _alive()
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
        "pid": _pid() if (running or alive) else None,
        "uptime": uptime,
        "tps": _tps(lines),
        "vram": gpu,
        "log": lines[-40:],
        "error": _last_error,
        "endpoint": f"http://{HOST}:{PORT}/v1",
        "endpoint_lan": f"http://{lan}:{PORT}/v1",
        "anthropic_url": (f"http://{lan}:{_anthropic['port']}" if _anthropic["port"] else ""),
        "anthropic_port": _anthropic["port"],
        "anthropic_enabled": CONF["server"].get("anthropicEnabled", True),
        "anthropic_error": _anthropic["error"],
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
                with open(os.path.join(RES_DIR, "ui.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
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
            was_running = _health() or _alive()
            if was_running:
                stop_model()
                time.sleep(1.0)
                res = start_model()
                return {"ok": res["ok"], "message": f"已切換模型並重啟：{res['message']}"}
            return {"ok": True, "message": f"已選擇 {os.path.basename(target)}"}

        def _dl_start():
            repo = (params.get("repo") or [""])[0]
            name = (params.get("file") or [""])[0]
            if not repo or not name:
                return {"ok": False, "message": "缺少 repo / file"}
            return models.start_download(repo, name)

        def _settings_save():
            n = int(self.headers.get("Content-Length") or 0)
            patch = {}
            if n > 0:
                try:
                    patch = json.loads(self.rfile.read(n).decode("utf-8-sig"))
                except Exception as exc:
                    return {"ok": False, "message": f"設定格式錯誤: {exc}"}
            return apply_settings(patch)

        routes = {
            "/api/start": start_model,
            "/api/stop": stop_model,
            "/api/purge": purge_all,
            "/api/selftest": selftest,
            "/api/select": _select,
            "/api/settings": _settings_save,
            "/api/hf/download": _dl_start,
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

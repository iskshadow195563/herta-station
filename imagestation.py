# -*- coding: utf-8 -*-
"""Herta Space Station - ComfyUI image engine.

The text side of the console drives llama-server; this module drives ComfyUI the
same way, so both halves of the UI share one control vocabulary:

  start()     spawn ``python main.py`` in a ComfyUI checkout, wait for /system_stats
  stop()      kill the process tree, then prove the VRAM came back
  purge()     stop + sweep stray engine processes, report the bytes reclaimed
  open_ui()   hand the canvas URL to the user's default (or configured) browser
  selftest()  one real generation through the HTTP API, or a connectivity probe
  scan_models()  what the checkout actually has on disk
  recommend()    which families fit this GPU, and in which prompt languages
  hf_search()/hf_files()/start_download()  the Hugging Face side of the library

Everything is stdlib-only and side-effect free at import time, mirroring
``models.py``: the console imports this module but touches no process until the
user presses start.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import hostos
import models

ENGINE = "ComfyUI"
DEFAULT_PORT = 8188
DEFAULT_HOST = "127.0.0.1"

DATA_DIR = os.path.abspath(os.environ.get("HERTA_DATA_DIR") or hostos.app_dir())
LOG_PATH = os.path.join(DATA_DIR, "comfyui.log")
PID_PATH = os.path.join(DATA_DIR, "comfyui.pid")

HF_BASE = os.environ.get("HERTA_HF_ENDPOINT") or "https://huggingface.co"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

_lock = threading.RLock()
_proc = None
_started_at = None
_last_error = ""
_last_error_code = ""
_owned = None            # True when this console spawned the running engine
_roots = {"root": "", "python": ""}

# ComfyUI model folders, in the order a launcher should search them.
MODEL_ROLES = [
    ("diffusion_models", "unet"),
    ("checkpoints", "checkpoint"),
    ("text_encoders", "te"),
    ("vae", "vae"),
    ("loras", "lora"),
]

ROLE_DIR = {"unet": "diffusion_models", "checkpoint": "checkpoints",
            "te": "text_encoders", "vae": "vae", "lora": "loras"}

# Prompt language capability per model family. ``tags`` means Danbooru-style
# booru tags rather than prose; the order is the recommended one.
LANG_LABEL = {
    "en": "English 自然語言",
    "zh": "中文",
    "ja": "日本語",
    "tags": "Danbooru 標籤（英文）",
}

FAMILIES = [
    # key, pattern, params, min VRAM GB, prompt languages, note
    ("qwen-image", r"qwen[-_]?image", "20B", 8, ["en", "zh"],
     "中英雙語提示詞、畫面文字最準；Turbo 8 步蒸餾版可上 16 GB 卡"),
    ("z-image", r"z[-_]?image|lumina2|zimage", "6B", 6, ["en"],
     "8 步蒸餾、Apache-2.0；只吃英文長句，寫 tag 或中文會失效"),
    ("flux", r"flux", "12B", 12, ["en"],
     "英文自然語言最強；dev 需授權，schnell 為 Apache-2.0 4 步版"),
    ("chroma", r"chroma", "8.9B", 14, ["en"],
     "FLUX 血統的無過濾重繪，30 步，8 GB 卡上很慢"),
    ("hunyuan", r"hunyuan", "13B", 12, ["en", "zh"],
     "騰訊混元，中文提示詞表現好，生態與 LoRA 較少"),
    ("wan", r"\bwan[-_ ]?2|wan2\.", "14B", 12, ["en", "zh"],
     "影片模型，文生圖可用但節點鏈較長"),
    ("kolors", r"kolors", "3B", 8, ["en", "zh"],
     "快手可圖，中文理解好，SDXL 相容 LoRA"),
    ("sdxl-anime", r"illustrious|noobai|pony|animagine|anything[-_]?v", "6.5B", 6, ["tags", "en"],
     "動漫專用，Danbooru 標籤才是正解，用自然語言會變糊"),
    ("sdxl", r"sdxl|sd[_-]?xl|stable-diffusion-xl|dreamshaper|juggernaut|realvis", "6.5B", 6, ["en", "tags"],
     "生態最完整、LoRA 最多，1024² 原生"),
    ("sd15", r"stable-diffusion-v1|sd1[._-]5|sd15|v1[-_]5", "1B", 4, ["en", "tags"],
     "老架構，512² 原生，僅適合驗證流程"),
]

# English copies of the prose above, so the console can run fully in English
# without translating strings on the client. Keys mirror FAMILIES / KNOWN.
NOTE_EN = {
    "qwen-image": "Bilingual prompts and the most accurate in-image text; the 8-step Turbo build fits 16 GB.",
    "z-image": "8-step distill, Apache-2.0; English prose only, tags or Chinese fall apart.",
    "flux": "Best English-prose adherence; dev needs a licence, schnell is Apache-2.0 at 4 steps.",
    "chroma": "Uncensored FLUX-lineage redraw, 30 steps, slow on 8 GB cards.",
    "hunyuan": "Tencent Hunyuan; strong Chinese prompts, thinner LoRA ecosystem.",
    "wan": "Video model; usable for stills but the graph is long.",
    "kolors": "Kuaishou Kolors; good Chinese understanding, SDXL-compatible LoRA.",
    "sdxl-anime": "Anime specialists: Danbooru tags are the way, prose turns to mush.",
    "sdxl": "Richest ecosystem and LoRA supply, native 1024².",
    "sd15": "Legacy architecture, native 512², only good for smoke tests.",
    "other": "Unknown family; trust the model card for prompt languages.",
}

KNOWN_EN = {
    "Qwen/Qwen-Image": "20B base; on a 16 GB card look for a Q4/Q6 GGUF quant.",
    "black-forest-labs/FLUX.1-schnell": "4-step distill, English.",
    "stabilityai/stable-diffusion-xl-base-1.0": "English tags or prose, both fine.",
    "lodestones/Chroma": "8.9B uncensored redraw.",
}

REASON = {
    "首選": ("在 16 GB 等級的卡上畫質與速度最平衡，且提示詞語言支援符合需求",
             "Best balance of quality and speed on a 16 GB card, with prompt languages that fit."),
    "首選_over": ("顯存不足，僅能以最低解析度嘗試，速度與穩定度都不保證",
                  "Not enough VRAM: only the lowest resolution will do, and speed is not guaranteed."),
    "速度優先": ("8 步以內出圖，適合反覆試稿", "Renders in 8 steps or fewer, good for iterating."),
    "動漫特化": ("動漫題材請用 Danbooru 標籤，不要寫自然語言長句",
                 "For anime, use Danbooru tags rather than long prose."),
    "畫質上限": ("接近顯存上限時請降解析度或改用 Q4/Q6 GGUF 量化",
                 "Near the VRAM ceiling, drop the resolution or move to a Q4/Q6 GGUF."),
}


def is_en(lang):
    return str(lang or "").lower().startswith("en")


def localize(text_zh, text_en, lang):
    return text_en if is_en(lang) else text_zh


KNOWN = [
    # repo, role of the useful file, display name, note
    ("Qwen/Qwen-Image", "unet", "Qwen-Image（官方，需自行轉 GGUF）",
     "20B 基座；16 GB 卡建議找 GGUF Q4/Q6 量化"),
    ("black-forest-labs/FLUX.1-schnell", "unet", "FLUX.1-schnell", "4 步蒸餾，英文"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "checkpoint", "SDXL Base 1.0", "英文 tag / 自然語言皆可"),
    ("lodestones/Chroma", "unet", "Chroma1-HD", "8.9B 無過濾重繪"),
]


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _candidates():
    here = hostos.app_dir()
    home = os.path.expanduser("~")
    out = []
    for drive in ("D:", "E:", "C:"):
        out.append(os.path.join(drive + os.sep, "AI", ENGINE))
        out.append(os.path.join(drive + os.sep, ENGINE))
    out += [
        os.path.join(home, "AI", ENGINE),
        os.path.join(home, ENGINE),
        os.path.join(home, "Documents", ENGINE),
        os.path.join(here, ENGINE),
        os.path.join(here, "..", ENGINE),
        "/opt/ComfyUI",
        os.path.join(home, "comfy", ENGINE),
    ]
    seen, uniq = set(), []
    for item in out:
        try:
            path = os.path.abspath(os.path.expanduser(str(item)))
        except (OSError, ValueError):
            continue
        if path.lower() in seen:
            continue
        seen.add(path.lower())
        uniq.append(path)
    return uniq


def is_checkout(path):
    """A directory counts as a checkout when main.py and the comfy package exist."""
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "main.py")):
        return False
    return os.path.isdir(os.path.join(path, "comfy")) or os.path.isfile(
        os.path.join(path, "nodes.py"))


def find_root(prefer=""):
    for item in [prefer] + _candidates():
        if item and is_checkout(os.path.abspath(os.path.expanduser(item))):
            return os.path.abspath(os.path.expanduser(item))
    return ""


def find_python(root, prefer=""):
    """The interpreter that owns torch for this checkout, best guess first."""
    root = root or ""
    parent = os.path.dirname(root.rstrip("\\/")) if root else ""
    names = (os.path.join("Scripts", "python.exe"), "python.exe") if hostos.IS_WIN else (
        os.path.join("bin", "python3"), os.path.join("bin", "python"), "python3")
    bases = [prefer] if prefer else []
    for base in (parent and [parent] or []) + [root] + ([parent] if parent else []):
        for name in names:
            bases.append(os.path.join(base, "venv", name))
            bases.append(os.path.join(base, ".venv", name))
    for item in bases:
        if item and os.path.isfile(item):
            return os.path.abspath(item)
    return ""


def inspect(conf=None):
    """Resolved root/python/models_dir plus what is missing."""
    image = image_conf(conf)
    root = _roots["root"] or find_root(image["comfyDir"])
    python = _roots["python"] or find_python(root, image["pythonExe"])
    return {
        "root": root,
        "python": python,
        "models_dir": os.path.join(root, "models") if root else "",
        "has_root": bool(root),
        "has_python": bool(python),
        "custom_nodes_gguf": bool(root) and os.path.isdir(
            os.path.join(root, "custom_nodes", "ComfyUI-GGUF")),
    }


def image_conf(conf=None):
    """The validated ``image`` settings group, with safe fallbacks."""
    data = {}
    if isinstance(conf, dict):
        data = conf.get("image") if isinstance(conf.get("image"), dict) else conf
    if not isinstance(data, dict):
        data = {}
    import settings as appsettings          # local import keeps import order simple
    defaults = appsettings.DEFAULTS["image"]
    merged = dict(defaults)
    merged.update({k: v for k, v in data.items() if k in defaults})
    return merged


# --------------------------------------------------------------------------- #
# process / HTTP probing
# --------------------------------------------------------------------------- #
def _run(args, timeout=15):
    return hostos.run(args, timeout)


def _http(url, timeout=4, payload=None):
    data = None
    headers = {"User-Agent": "herta-station/1.1"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # Keep the engine's own error text: ComfyUI explains bad graphs in the body.
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {(detail or str(exc.reason))[:800]}") from None
    return json.loads(raw.decode("utf-8")) if raw else {}


def stats(port=None):
    """``/system_stats`` or None when nothing answers on the port."""
    port = int(port or (image_conf().get("port") or DEFAULT_PORT))
    try:
        return _http(f"http://{DEFAULT_HOST}:{port}/system_stats", timeout=3)
    except Exception:
        return None


def queue(port=None):
    port = int(port or (image_conf().get("port") or DEFAULT_PORT))
    try:
        data = _http(f"http://{DEFAULT_HOST}:{port}/queue", timeout=3)
    except Exception:
        return {"running": 0, "pending": 0}
    return {"running": len(data.get("queue_running") or []),
            "pending": len(data.get("queue_pending") or [])}


def _listener(port):
    try:
        return hostos.listener_pid(port, runner=_run)
    except Exception:
        return None


def _pid():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc.pid
    try:
        with open(PID_PATH, "r", encoding="utf-8") as fh:
            pid = int((fh.read() or "").strip() or 0)
    except (OSError, ValueError):
        pid = 0
    if pid and hostos.pid_alive(pid) and _is_engine(pid):
        return pid
    return None


_CMD_CACHE = {}
CMD_TTL = 6.0


def _cmdline(pid):
    """Full command line of ``pid`` (``hostos.cmdline`` builds argv, it does not read /proc)."""
    pid = int(pid or 0)
    if pid <= 0:
        return ""
    hit = _CMD_CACHE.get(pid)
    if hit and time.time() - hit[0] < CMD_TTL:
        return hit[1]
    text = ""
    try:
        if hostos.IS_WIN:
            text = (_run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % pid],
                         timeout=20) or "").strip()
            if not text or "Get-CimInstance" in text or "CommandLine" in text:
                text = (_run(["wmic", "process", "where", "processid=%d" % pid,
                              "get", "commandline", "/format:list"], timeout=20) or "").strip()
        elif hostos.IS_LINUX:
            try:
                with open("/proc/%d/cmdline" % pid, "rb") as fh:
                    text = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            except OSError:
                text = ""
        else:
            text = (_run(["ps", "-o", "command=", "-p", str(pid)], timeout=15) or "").strip()
    except Exception:
        text = ""
    _CMD_CACHE[pid] = (time.time(), text)
    return text


def _is_engine(pid):
    """True when ``pid`` looks like ``python main.py`` inside some ComfyUI tree."""
    line = _cmdline(pid)
    if not line:
        return False
    return bool(re.search(r"main\.py", line, re.I)) and bool(re.search(r"python", line, re.I))


def _alive():
    return _pid() is not None


def _log_tail(count=80):
    try:
        size = os.path.getsize(LOG_PATH)
        with open(LOG_PATH, "rb") as fh:
            fh.seek(max(0, size - 65536))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [ln for ln in text.splitlines()][-count:]


def _failure(code, message):
    global _last_error, _last_error_code
    _last_error, _last_error_code = message, code
    return {"ok": False, "message": message, "error_code": code}


def _clear_error():
    """Drop a stale failure so the console stops showing it after a recovery."""
    global _last_error, _last_error_code
    _last_error, _last_error_code = "", ""


def _vram_mb():
    try:
        info = hostos.vram_usage(runner=_run) or {}
        return int(info.get("used") or 0)
    except Exception:
        return 0


def _wait_ready(port, timeout):
    deadline = time.time() + max(20, int(timeout or 180))
    while time.time() < deadline:
        if stats(port):
            return True
        if _proc is not None and _proc.poll() is not None:
            return False
        time.sleep(1.0)
    return False


# ComfyUI dropped --normalvram years ago: "normal" simply means "let dynamic VRAM decide",
# so it must not send any flag at all. Sending one aborts startup with "unrecognized arguments".
VRAM_ARGS = {"high": ["--highvram"], "normal": [], "low": ["--lowvram"], "verylow": ["--novram"]}


def _total_vram_gb():
    try:
        return float((hostos.gpu_info() or {}).get("vram_gb") or 0.0)
    except Exception:
        return 0.0


def mode_for(mode, total_gb=0.0):
    """Resolve the stored vramMode into a mode ComfyUI actually understands."""
    mode = str(mode or "auto").strip().lower()
    if mode in VRAM_ARGS:
        return mode
    if total_gb and total_gb < 8:
        return "low"
    return "high" if total_gb >= 12 else "normal"


def vram_flags(mode):
    return list(VRAM_ARGS.get(str(mode or "normal").strip().lower(), []))


def _device_summary(info):
    if not info:
        return {"device": "", "device_name": "", "engine_version": "", "torch": "",
                "python": "", "vram_total": 0, "vram_free": 0}
    devices = info.get("devices") or []
    first = devices[0] if devices else {}
    system = info.get("system") or {}
    return {
        "device": first.get("type") or first.get("name") or "",
        "device_name": first.get("name") or "",
        "engine_version": str(system.get("comfyui_version") or ""),
        "torch": str(system.get("pytorch_version") or ""),
        "python": str(system.get("python_version") or "").split(" ")[0],
        "vram_total": int(first.get("vram_total") or 0),
        "vram_free": int(first.get("vram_free") or 0),
    }


# --------------------------------------------------------------------------- #
# start / stop / purge / open
# --------------------------------------------------------------------------- #
def start(conf=None):
    global _proc, _started_at, _last_error, _last_error_code, _owned, _roots
    image = image_conf(conf)
    port = int(image["port"])
    if stats(port):
        _owned = False
        _clear_error()
        return {"ok": True, "message": f"生圖引擎已在執行（{DEFAULT_HOST}:{port}）",
                "url": f"http://{DEFAULT_HOST}:{port}", "reused": True}
    info = inspect(conf)
    if not info["has_root"]:
        return _failure("missing_engine",
                        "找不到 ComfyUI 安裝；請在設定裡指定 comfyDir 或設定 HERTA_COMFY_DIR")
    if not info["has_python"]:
        return _failure("missing_python",
                        "找不到 ComfyUI 的 Python 環境（venv）；請在設定裡指定 pythonExe")
    if _listener(port):
        return _failure("port_in_use", f"連接埠 {port} 已被其他程式佔用")

    args = [info["python"], "main.py", "--listen", image["listen"], "--port", str(port),
            "--disable-auto-launch"]
    args += vram_flags(mode_for(image["vramMode"], _total_vram_gb()))
    if image["extraArgs"]:
        args += [part for part in str(image["extraArgs"]).split() if part]

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        log = open(LOG_PATH, "ab", buffering=0)
        log.write((f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {hostos.cmdline(args)}\n").encode("utf-8"))
        _proc = subprocess.Popen(args, cwd=info["root"], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, **hostos.spawn_kwargs())
    except OSError as exc:
        return _failure("launch_failed", f"生圖引擎啟動失敗: {exc}")
    _started_at = time.time()
    _owned = True
    _roots.update(info)
    _last_error = _last_error_code = ""
    try:
        with open(PID_PATH, "w", encoding="utf-8") as fh:
            fh.write(str(_proc.pid))
    except OSError:
        pass

    ok = _wait_ready(port, image["loadTimeout"])
    if not ok:
        code = "engine_exited"
        detail = "\n".join(_log_tail(40))
        if re.search(r"out of memory|OutOfMemory|CUDA error", detail, re.I):
            code = "out_of_memory"
        elif re.search(r"address already in use", detail, re.I):
            code = "port_in_use"
        elif re.search(r"ModuleNotFoundError|No module named", detail, re.I):
            code = "python_env_broken"
        stop(conf)
        return _failure(code, f"生圖引擎沒有就緒（{code}）；請看日誌")

    summary = _device_summary(stats(port))
    url = f"http://{DEFAULT_HOST}:{port}"
    _clear_error()
    opened = open_ui(conf) if image["autoOpen"] else {"ok": True, "opened": False, "url": url}
    return {"ok": True, "url": url, "device": summary["device"],
            "engine_version": summary["engine_version"], "opened": opened.get("opened", False),
            "message": f"生圖引擎已啟動：{url}（{summary['device'] or 'cpu'}）"}


def stop(conf=None, quiet=False):
    global _proc, _started_at, _last_error_code, _owned
    image = image_conf(conf)
    port = int(image["port"])
    before = _vram_mb()
    pids = []
    if _proc is not None and _proc.poll() is None:
        pids.append(_proc.pid)
    if not pids:
        pid = _pid()
        if pid:
            pids.append(pid)
    if not pids:
        pid = _listener(port)
        if pid and _is_engine(pid):
            pids.append(pid)
        elif pid:
            return _failure("port_in_use",
                            f"連接埠 {port} 被其他程式（PID {pid}）佔用，不是本控制台啟動的引擎")
    for pid in pids:
        hostos.kill_tree(pid)
    deadline = time.time() + 25
    while time.time() < deadline and (stats(port) or _listener(port)):
        time.sleep(0.6)
    _proc = None
    _started_at = None
    _owned = None
    try:
        os.remove(PID_PATH)
    except OSError:
        pass
    time.sleep(0.8)
    freed = max(0, before - _vram_mb())
    if pids and stats(port):
        return _failure("stop_failed", "生圖引擎仍在回應，停止失敗")
    if quiet:
        return {"ok": True, "freed_gb": round(freed / 1024.0, 2)}
    _clear_error()
    return {"ok": True, "freed_gb": round(freed / 1024.0, 2),
            "message": f"已停止生圖引擎，釋放 {freed / 1024.0:.1f} GB 顯存"}


def _engine_pids(root):
    """Live ``python main.py`` processes that belong to a ComfyUI tree.

    The command line usually holds a *relative* ``main.py`` (the cwd is the
    checkout), so a candidate also counts when its path mentions comfyui —
    otherwise ``purge`` would either miss everything or kill unrelated scripts.
    """
    root = (root or "").rstrip("\\/").lower()
    parent = os.path.dirname(root).lower() if root else ""
    pids = []
    for pid in hostos.pids_by_name("python", runner=_run) or []:
        if pid == os.getpid():
            continue
        line = _cmdline(pid)
        if not re.search(r"main\.py", line, re.I):
            continue
        low = line.lower()
        if (root and root in low) or (parent and parent in low) or "comfyui" in low:
            pids.append(pid)
    return pids


def purge(conf=None):
    """Stop the engine and sweep any stray ComfyUI process still holding VRAM."""
    image = image_conf(conf)
    root = (inspect(conf)["root"] or "").lower()
    before = _vram_mb()
    result = stop(conf, quiet=True)
    swept = 0
    for pid in _engine_pids(root):
        if hostos.kill_tree(pid):
            swept += 1
    if swept:
        time.sleep(1.2)
    freed = max(0, before - _vram_mb())
    note = f"，另外清掉 {swept} 個殘留程序" if swept else ""
    if not result.get("ok"):
        return {**result, "message": result.get("message", "") + note}
    return {"ok": True, "freed_gb": round(freed / 1024.0, 2), "swept": swept,
            "message": f"已清空生圖顯存，釋放 {freed / 1024.0:.1f} GB{note}"}


def open_ui(conf=None):
    """Open the canvas in the configured browser, else the system default."""
    image = image_conf(conf)
    port = int(image["port"])
    if not stats(port):
        return {"ok": False, "opened": False,
                "message": "生圖引擎還沒啟動，先按一鍵啟動", "error_code": "engine_not_running"}
    url = f"http://{DEFAULT_HOST}:{port}"
    target = str(image.get("browser") or "").strip()
    if target and target.lower() not in ("default", "auto"):
        exe = target if os.path.isfile(target) else (shutil.which(target) or "")
        if exe:
            try:
                subprocess.Popen([exe, url], **hostos.spawn_kwargs())
                return {"ok": True, "opened": True, "url": url, "browser": os.path.basename(exe)}
            except OSError:
                pass
    opened = hostos.open_url(url)
    return {"ok": True, "opened": bool(opened), "url": url,
            "browser": "default" if opened else ""}


def status(conf=None):
    """Everything the image view polls; never raises, never blocks for long."""
    image = image_conf(conf)
    port = int(image["port"])
    info = stats(port)
    pid = _pid() or _listener(port)
    phase = "running" if info else ("loading" if pid else "idle")
    summary = _device_summary(info)
    uptime = None
    if phase != "idle":
        base = _started_at
        if base is None:
            try:
                base = os.path.getctime(LOG_PATH)
            except OSError:
                base = None
        if base:
            uptime = int(max(0, time.time() - base))
    inspected = inspect(conf)
    return {
        "phase": phase,
        "engine": ENGINE,
        "port": port,
        "host": DEFAULT_HOST,
        "url": f"http://{DEFAULT_HOST}:{port}",
        "pid": pid,
        "owned": bool(_owned),
        "uptime": uptime,
        "device": summary["device"],
        "device_name": summary["device_name"],
        "engine_version": summary["engine_version"],
        "torch": summary["torch"],
        "python_ver": summary["python"],
        "root": inspected["root"],
        "python": inspected["python"],
        "models_dir": inspected["models_dir"],
        "engine_installed": inspected["has_root"] and inspected["has_python"],
        "gguf_nodes": inspected["custom_nodes_gguf"],
        "queue": queue(port) if info else {"running": 0, "pending": 0},
        "vram": _vram_payload(summary),
        "auto_open": bool(image["autoOpen"]),
        "browser": str(image["browser"] or "default"),
        "vram_mode": image["vramMode"],
        "listen": image["listen"],
        "log": _log_tail(40),
        "error": _last_error,
        "error_code": _last_error_code,
        "platform": ("windows" if hostos.IS_WIN else
                     ("macos" if hostos.IS_MAC else ("linux" if hostos.IS_LINUX else ""))),
    }


def _vram_payload(summary):
    used = _vram_mb()
    total = 0
    try:
        total = int((hostos.vram_usage(runner=_run) or {}).get("total") or 0)
    except Exception:
        total = 0
    if not total:
        total = int((summary.get("vram_total") or 0) / (1024 * 1024))
    return {"used": used, "total": total}


# --------------------------------------------------------------------------- #
# models on disk
# --------------------------------------------------------------------------- #
def _weights(root, roles=("diffusion_models", "checkpoints")):
    out = []
    for folder, role in MODEL_ROLES:
        if role not in roles:
            continue
        base = os.path.join(root, "models", folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not os.path.isfile(path) or os.path.getsize(path) < 1024 * 1024:
                continue
            if not name.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".bin")):
                continue
            out.append({"name": name, "role": role, "path": path,
                        "size_gb": round(os.path.getsize(path) / (1024 ** 3), 2),
                        "family": family_of(name)})
    return out


def family_of(name):
    low = (name or "").lower()
    for key, pattern, _params, _vram, _langs, _note in FAMILIES:
        if re.search(pattern, low):
            return key
    return "other"


def family_meta(key):
    for item in FAMILIES:
        if item[0] == key:
            return {"key": item[0], "params": item[2], "min_vram_gb": item[3],
                    "langs": list(item[4]), "lang_labels": [LANG_LABEL[x] for x in item[4]],
                    "note": item[5], "note_en": NOTE_EN.get(item[0], item[5])}
    return {"key": "other", "params": "", "min_vram_gb": 8, "langs": ["en"],
            "lang_labels": [LANG_LABEL["en"]],
            "note": "未知家族，提示詞語言請以模型卡為準",
            "note_en": NOTE_EN["other"]}


def scan_models(conf=None):
    info = inspect(conf)
    root = info["root"]
    if not root:
        return {"ok": False, "dir": "", "items": [], "total_gb": 0, "root": "",
                "message": "找不到 ComfyUI 安裝"}
    items = []
    for folder, role in MODEL_ROLES:
        base = os.path.join(root, "models", folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not os.path.isfile(path) or name.startswith("."):
                continue
            size = os.path.getsize(path)
            if size < 1024 * 1024:
                continue
            key = family_of(name)
            meta = family_meta(key)
            items.append({"name": name, "role": role, "folder": folder, "path": path,
                          "size_gb": round(size / (1024 ** 3), 2),
                          "family": key, "langs": meta["langs"],
                          "lang_labels": meta["lang_labels"],
                          "params": meta["params"], "min_vram_gb": meta["min_vram_gb"]})
    total = round(sum(i["size_gb"] for i in items), 2)
    return {"ok": True, "dir": os.path.join(root, "models"), "items": items,
            "total_gb": total, "root": root}


# --------------------------------------------------------------------------- #
# GPU recommendation
# --------------------------------------------------------------------------- #
def recommend(conf=None, lang="zh-Hant"):
    total = float(models.gpu_total_gb() or 0)
    hardware = models.hardware() if hasattr(models, "hardware") else {}
    local = scan_models(conf)
    installed = {}
    for item in local.get("items", []):
        installed.setdefault(item["family"], []).append(item)

    def fits(key):
        return family_meta(key)["min_vram_gb"] <= max(0.0, total - 1.5)

    order = ["qwen-image", "flux", "chroma", "sdxl", "sdxl-anime", "hunyuan", "kolors",
             "z-image", "wan", "sd15"]
    best = next((k for k in order if fits(k)), "")
    if not best:
        # Nothing clears the budget: name the lightest family so the UI still has
        # something concrete to show, and carry the warning up to the caller.
        best = min(order, key=lambda k: family_meta(k)["min_vram_gb"])
    fast = next((k for k in ("qwen-image", "z-image", "sdxl") if fits(k)), best)
    anime = "sdxl-anime" if fits("sdxl-anime") else best
    tight = total >= 12
    warning = ""
    if not fits(best):
        warning = (f"這張卡（{total:.1f} GB）低於任何建議模型的顯存門檻；"
                   f"最輕的 {family_meta(best)['params']} 模型也需要 "
                   f"{family_meta(best)['min_vram_gb']} GB，只能降解析度硬跑或改用雲端。")
        if is_en(lang):
            warning = (f"This card ({total:.1f} GB) is under every recommendation's VRAM floor; "
                       f"even the lightest {family_meta(best)['params']} model needs "
                       f"{family_meta(best)['min_vram_gb']} GB, so drop the resolution or go cloud.")

    picks = []
    for tier, key, reason_key in (
            ("首選", best, "首選" if fits(best) else "首選_over"),
            ("速度優先", fast, "速度優先"),
            ("動漫特化", anime, "動漫特化"),
            ("畫質上限", "qwen-image" if tight else best, "畫質上限"),
    ):
        meta = family_meta(key)
        have = installed.get(key) or []
        picks.append({
            "tier": tier, "family": key, "params": meta["params"],
            "min_vram_gb": meta["min_vram_gb"], "fits": fits(key),
            "langs": meta["langs"],
            "lang_labels": meta["lang_labels"],
            "note": localize(meta["note"], meta["note_en"], lang),
            "reason": localize(*REASON[reason_key], lang),
            "installed": [i["name"] for i in have],
            "search": {"qwen-image": "Qwen-Image GGUF", "z-image": "Z-Image GGUF",
                       "flux": "FLUX.1 schnell", "chroma": "Chroma1-HD",
                       "sdxl": "SDXL base 1.0", "sdxl-anime": "Illustrious XL",
                       "hunyuan": "HunyuanDiT", "kolors": "Kolors",
                       "wan": "Wan2.1", "sd15": "stable-diffusion v1-5"}.get(key, key),
        })
    return {
        "gpu": (hardware.get("gpu") or "unknown"),
        "gpu_total_gb": total,
        "unified": bool(hardware.get("unified")),
        "picks": picks,
        "warning": warning,
        "local": local.get("items", []),
        "local_gb": local.get("total_gb", 0),
        "installed_families": sorted(installed.keys()),
        "known": [{"repo": repo, "role": role, "name": name,
                   "note": localize(note, KNOWN_EN.get(repo, note), lang)}
                  for repo, role, name, note in KNOWN],
        "hint": localize("先按一鍵啟動引擎，再從模型庫下載缺少的權重；"
                         "下載會直接放進 ComfyUI 對應的 models 子目錄。",
                         "Start the engine first, then pull the missing weights from the "
                         "image model library; downloads land in the matching ComfyUI "
                         "models folder.", lang),
        "lang": "en" if is_en(lang) else "zh-Hant",
    }


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #
def _hf_json(url, timeout=25):
    headers = {"User-Agent": "herta-station/1.1"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _langs_of(row):
    """Prompt languages a repo advertises, unioned with the family default."""
    langs = []
    card = row.get("cardData") if isinstance(row.get("cardData"), dict) else {}
    declared = card.get("language")
    if isinstance(declared, str):
        declared = [declared]
    for item in declared or []:
        code = str(item).lower()
        if code.startswith("zh"):
            code = "zh"
        elif code.startswith("ja"):
            code = "ja"
        elif code.startswith("en"):
            code = "en"
        if code in ("en", "zh", "ja") and code not in langs:
            langs.append(code)
    key = family_of(row.get("id") or "")
    for code in family_meta(key)["langs"]:
        if code == "tags":
            if "tags" not in langs:
                langs.append("tags")
        elif code not in langs:
            langs.append(code)
    return langs or ["en"]


def hf_search(query="", sort="downloads", limit=30, pipeline="text-to-image"):
    limit = max(1, min(60, int(limit or 30)))
    params = {"limit": str(limit), "full": "true", "sort": sort or "downloads",
              "direction": "-1"}
    if pipeline:
        params["filter"] = pipeline
    if query:
        params["search"] = query
    rows = _hf_json(f"{HF_BASE}/api/models?" + urllib.parse.urlencode(params))
    items = []
    for row in rows or []:
        repo = row.get("id") or ""
        key = family_of(repo)
        meta = family_meta(key)
        langs = _langs_of(row)
        items.append({
            "repo": repo, "downloads": row.get("downloads") or 0, "likes": row.get("likes") or 0,
            "pipeline": row.get("pipeline_tag") or "", "family": key,
            "params": meta["params"], "min_vram_gb": meta["min_vram_gb"],
            "langs": langs, "lang_labels": [LANG_LABEL[x] for x in langs],
            "gated": bool(row.get("gated")), "updated": (row.get("lastModified") or "")[:10],
            "build": "gguf" if "gguf" in (row.get("tags") or []) else "diffusers",
        })
    return items


def _valid_repo(repo):
    """``org/name`` only - no traversal, no extra segments."""
    text = str(repo or "").strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", text):
        return ""
    owner, name = text.split("/", 1)
    if owner in (".", "..") or name in (".", ".."):
        return ""
    return text


def hf_files(repo):
    """Weight files inside a repo, with the ComfyUI folder each belongs to."""
    repo = _valid_repo(repo)
    if not repo:
        raise ValueError("invalid Hugging Face repository")
    rows = _hf_json(f"{HF_BASE}/api/models/{repo}/tree/main?recursive=true")
    items = []
    for row in rows or []:
        if row.get("type") != "file":
            continue
        path = row.get("path") or ""
        low = path.lower()
        if not low.endswith((".safetensors", ".gguf", ".ckpt", ".pt")):
            continue
        size = row.get("size") or (row.get("lfs") or {}).get("size") or 0
        items.append({"path": path, "size_gb": round(int(size) / (1024 ** 3), 2),
                      "role": guess_role(path), "family": family_of(path)})
    items.sort(key=lambda i: (-i["size_gb"], i["path"]))
    return items


def guess_role(path):
    low = (path or "").lower()
    if re.search(r"(^|/)(vae|.*_vae|ae)\b|vae[._-]", low):
        return "vae"
    if re.search(r"clip|t5|text_encoder|qwen3vl|llm", low):
        return "te"
    if re.search(r"lora", low):
        return "lora"
    if re.search(r"checkpoint|sd_xl_base|xl_base", low):
        return "checkpoint"
    return "unet"


# --------------------------------------------------------------------------- #
# downloader (image weights land straight in the ComfyUI model folders)
# --------------------------------------------------------------------------- #
_state = {"active": False, "repo": "", "file": "", "path": "", "role": "", "done": 0,
          "total": 0, "speed": 0.0, "started": 0.0, "error": "", "error_code": "",
          "finished": False, "target": ""}
_cancel = threading.Event()


def download_state():
    with _lock:
        data = dict(_state)
    pct = (data["done"] / data["total"] * 100.0) if data["total"] else 0.0
    data["percent"] = round(min(100.0, pct), 1)
    return data


def start_download(conf, repo, filename, role=""):
    global _state
    repo = str(repo or "").strip()
    filename = str(filename or "").strip()
    if not _valid_repo(repo):
        return {"ok": False, "message": "Hugging Face 倉庫名稱無效", "error_code": "invalid_download"}
    if not filename or "\\" in filename or any(p in ("", ".", "..") for p in filename.split("/")):
        return {"ok": False, "message": "檔名無效", "error_code": "invalid_download"}
    if not filename.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt")):
        return {"ok": False, "message": "只接受 safetensors / gguf / ckpt 權重",
                "error_code": "invalid_download"}
    info = inspect(conf)
    if not info["has_root"]:
        return {"ok": False, "message": "找不到 ComfyUI 安裝，無法放置權重", "error_code": "missing_engine"}
    role = role if role in ROLE_DIR else guess_role(filename)
    folder = os.path.join(info["root"], "models", ROLE_DIR[role])
    target = os.path.join(folder, os.path.basename(filename))
    with _lock:
        if _state["active"]:
            return {"ok": False, "message": "已有下載進行中", "error_code": "download_busy"}
        _state = {"active": True, "repo": repo, "file": filename, "path": "",
                  "role": role, "done": 0, "total": 0, "speed": 0.0, "started": time.time(),
                  "error": "", "error_code": "", "finished": False, "target": target}
        _cancel.clear()
    threading.Thread(target=_worker, args=(repo, filename, folder, target), daemon=True).start()
    return {"ok": True, "role": role, "target": target,
            "message": f"開始下載 {os.path.basename(filename)} → models/{ROLE_DIR[role]}"}


def cancel_download():
    with _lock:
        if _state["active"]:
            _cancel.set()
    return {"ok": True, "message": "已請求取消下載"}


def reset_download():
    global _state
    with _lock:
        if _state["active"]:
            return {"ok": False, "message": "下載進行中，無法重設"}
        _state = {"active": False, "repo": "", "file": "", "path": "", "role": "", "done": 0,
                  "total": 0, "speed": 0.0, "started": 0.0, "error": "", "error_code": "",
                  "finished": False, "target": ""}
    return {"ok": True, "message": "已清除下載狀態"}


def _worker(repo, filename, folder, target):
    part = target + ".part"
    url = f"{HF_BASE}/{repo}/resolve/main/{urllib.parse.quote(filename)}"
    headers = {"User-Agent": "herta-station/1.1"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        os.makedirs(folder, exist_ok=True)
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with _lock:
                _state["total"] = total
            done, mark, mark_t = 0, 0, time.time()
            with open(part, "wb") as fh:
                while True:
                    if _cancel.is_set():
                        raise RuntimeError("download_cancelled")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    speed = (done - mark) / (now - mark_t) if now > mark_t else 0.0
                    if now - mark_t >= 0.5:
                        mark, mark_t = done, now
                    with _lock:
                        _state.update({"done": done, "speed": speed})
        size = os.path.getsize(part)
        if _state.get("total") and size < _state["total"]:
            raise RuntimeError("download_incomplete")
        os.replace(part, target)
        with _lock:
            _state.update({"active": False, "finished": True, "path": target,
                           "done": size, "total": size or _state.get("total") or size,
                           "speed": 0.0})
    except Exception as exc:
        try:
            os.remove(part)
        except OSError:
            pass
        text = str(exc)
        code = "download_failed"
        if "download_cancelled" in text:
            code = "download_cancelled"
        elif "download_incomplete" in text:
            code = "download_incomplete"
        elif isinstance(getattr(exc, "code", None), int) and exc.code in (401, 403):
            code = "auth_required"
            text = "此倉庫需要 Hugging Face 授權（設定 HF_TOKEN 後重試）"
        elif isinstance(getattr(exc, "code", None), int) and exc.code == 404:
            code = "not_found"
            text = "倉庫或檔案不存在"
        with _lock:
            _state.update({"active": False, "finished": True, "error": text,
                           "error_code": code, "speed": 0.0})


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def _pick(root, folder, needles):
    base = os.path.join(root, "models", folder)
    if not os.path.isdir(base):
        return ""
    names = sorted(os.listdir(base))
    for needle in needles:
        for name in names:
            if re.search(needle, name, re.I) and os.path.isfile(os.path.join(base, name)):
                return name
    return ""


def selftest(conf=None):
    """One real 512x512 4-step generation through /prompt, else a probe report."""
    image = image_conf(conf)
    port = int(image["port"])
    info = stats(port)
    if not info:
        return _failure("engine_not_running", "生圖引擎沒在跑，先按一鍵啟動")
    summary = _device_summary(info)
    root = inspect(conf)["root"]
    try:
        objects = _http(f"http://{DEFAULT_HOST}:{port}/object_info", timeout=25)
    except Exception as exc:
        return _failure("api_error", f"讀取節點清單失敗: {exc}")

    workflow = str(image.get("selftestWorkflow") or "").strip()
    graph = None
    if workflow and os.path.isfile(workflow):
        try:
            with open(workflow, "r", encoding="utf-8") as fh:
                graph = json.load(fh)
        except (OSError, ValueError) as exc:
            return _failure("workflow_invalid", f"自訂工作流讀取失敗: {exc}")
    elif root:
        graph = _auto_graph(objects, root)

    if not graph:
        return {"ok": True, "mode": "probe", "device": summary["device"],
                "engine_version": summary["engine_version"], "nodes": len(objects),
                "message": (f"API 正常：{summary['device'] or 'cpu'} · 引擎 "
                            f"{summary['engine_version'] or '?'} · 節點 {len(objects)} 個；"
                            "模型庫裡沒有可自動組裝的 DiT/TE/VAE 組合，略過實際出圖")}

    started = time.time()
    try:
        queued = _http(f"http://{DEFAULT_HOST}:{port}/prompt", timeout=30,
                       payload={"prompt": graph, "client_id": "herta-selftest"})
    except Exception as exc:
        return _failure("prompt_rejected", f"送出工作流失敗: {exc}")
    prompt_id = (queued or {}).get("prompt_id")
    if not prompt_id:
        return _failure("prompt_rejected", f"引擎沒有回傳 prompt_id: {queued}")
    deadline = time.time() + max(60, int(image["genTimeout"]))
    while time.time() < deadline:
        try:
            history = _http(f"http://{DEFAULT_HOST}:{port}/history/{prompt_id}", timeout=10)
        except Exception:
            history = {}
        entry = (history or {}).get(prompt_id) or {}
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            return _failure("prompt_error", _history_error(status) or "引擎執行失敗，請看日誌")
        if status.get("completed"):
            seconds = round(time.time() - started, 2)
            files = []
            for node in (entry.get("outputs") or {}).values():
                for item in node.get("images") or []:
                    files.append(item.get("filename"))
            return {"ok": True, "mode": "generate", "seconds": seconds, "images": files,
                    "device": summary["device"],
                    "message": f"實測出圖成功：{seconds} 秒 / 512² 4 步（{summary['device']}）"}
        time.sleep(1.0)
    return _failure("prompt_timeout", "工作流送出後逾時沒有完成，請看日誌")


SELFTEST_PROMPT = "a clean studio photograph of a red apple on a white table, soft daylight"


def _history_error(status):
    """Pull the human-readable reason out of a /history error status."""
    for item in status.get("messages") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        kind, payload = item[0], item[1]
        if kind == "execution_error" and isinstance(payload, dict):
            node = payload.get("node_type") or payload.get("node_id") or "?"
            detail = payload.get("exception_message") or ""
            return f"{node}: {detail.strip().splitlines()[0] if detail else 'execution_error'}"
    return ""


def _auto_graph(objects, root):
    """Smallest graph the checkout can actually honour.

    Qwen-Image 2.1 ships one fused encoder (TextEncodeQwenImage21) that returns
    positive, negative *and* the empty latent, so that family needs its own
    7-node shape; every other family uses CLIPTextEncode + EmptySD3LatentImage.
    """
    names = set(objects.keys())
    dit = _pick(root, "diffusion_models",
                [r"turbo", r"qwen", r"z[-_]image", r"flux", r"chroma", r".*"])
    te = _pick(root, "text_encoders", [r"qwen3vl", r"qwen", r"t5", r"clip", r".*"])
    qwen = bool(re.search(r"qwen", dit, re.I))
    # The VAE has to match the family: a Qwen latent is packed to 64 channels and
    # the SD3/Flux VAE (ae.safetensors, 16ch) cannot decode it.
    vae_needles = ([r"qwen.*vae", r"vae.*qwen"] if qwen
                   else [r"ae\.safetensors", r"z[-_]image.*vae", r"vae"])
    vae = _pick(root, "vae", vae_needles) or _pick(root, "vae", [r".*"])
    if not (dit and te and vae):
        return None
    te_type = "qwen_image" if qwen else "lumina2"
    if dit.lower().endswith(".gguf"):
        if "UnetLoaderGGUF" not in names:
            return None
        unet_node = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": dit}}
    else:
        if "UNETLoader" not in names:
            return None
        unet_node = {"class_type": "UNETLoader",
                     "inputs": {"unet_name": dit, "weight_dtype": "default"}}
    # Quantised text encoders go through ComfyUI-GGUF, safetensors through core CLIPLoader.
    if te.lower().endswith(".gguf"):
        if "CLIPLoaderGGUF" not in names:
            return None
        te_node = {"class_type": "CLIPLoaderGGUF",
                   "inputs": {"clip_name": te, "type": te_type}}
    else:
        if "CLIPLoader" not in names:
            return None
        te_node = {"class_type": "CLIPLoader",
                   "inputs": {"clip_name": te, "type": te_type, "device": "default"}}

    graph = {
        "1": unet_node,
        "2": te_node,
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    if qwen and "TextEncodeQwenImage21" in names:
        # The fused encoder's `latent` output is a *packed* latent that the plain
        # VAEDecode cannot read, so only its positive/negative are used and the
        # empty latent comes from EmptySD3LatentImage (verified against ComfyUI 0.43).
        graph["4"] = {"class_type": "TextEncodeQwenImage21",
                      "inputs": {"clip": ["2", 0], "prompt": SELFTEST_PROMPT,
                                 "negative_prompt": "", "resolution": 512, "images": []}}
        graph["5"] = {"class_type": "EmptySD3LatentImage",
                      "inputs": {"width": 512, "height": 512, "batch_size": 1}}
        graph["6"] = {"class_type": "KSampler",
                      "inputs": {"model": ["1", 0], "positive": ["4", 0],
                                 "negative": ["4", 1], "latent_image": ["5", 0],
                                 "seed": 12345, "steps": 4, "cfg": 1.0,
                                 "sampler_name": "euler", "scheduler": "simple",
                                 "denoise": 1.0}}
        source = "6"
    else:
        if "CLIPTextEncode" not in names or "EmptySD3LatentImage" not in names:
            return None
        graph["4"] = {"class_type": "CLIPTextEncode",
                      "inputs": {"clip": ["2", 0], "text": SELFTEST_PROMPT}}
        graph["5"] = {"class_type": "EmptySD3LatentImage",
                      "inputs": {"width": 512, "height": 512, "batch_size": 1}}
        graph["6"] = {"class_type": "KSampler",
                      "inputs": {"model": ["1", 0], "positive": ["4", 0],
                                 "negative": ["4", 0], "latent_image": ["5", 0],
                                 "seed": 12345, "steps": 4, "cfg": 1.0,
                                 "sampler_name": "euler", "scheduler": "simple",
                                 "denoise": 1.0}}
        source = "6"
    graph["7"] = {"class_type": "VAEDecode",
                  "inputs": {"samples": [source, 0], "vae": ["3", 0]}}
    graph["8"] = {"class_type": "SaveImage",
                  "inputs": {"images": ["7", 0], "filename_prefix": "herta-selftest"}}
    return graph

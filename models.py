# -*- coding: utf-8 -*-
"""Model library for Herta Station: local GGUF catalogue, Hugging Face search,
and a background downloader.

Everything here is stdlib-only so it survives PyInstaller's one-file build.
"""

import json
import os
import re
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request

RES_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else RES_DIR

MODELS_DIR = os.environ.get("HERTA_MODELS_DIR", os.path.join(DATA_DIR, "models"))
ACTIVE_FILE = os.path.join(DATA_DIR, "active-model.json")

QUANT_RE = re.compile(
    r"\b(IQ\d+_[A-Z0-9]+|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d|BF16|F16|F32|MXFP4|NVFP4)\b",
    re.IGNORECASE,
)
PARAM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[Bb](?!\w)")

FAMILIES = [
    ("qwen", "Qwen"),
    ("deepseek", "DeepSeek"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("mistral", "Mistral"),
    ("mixtral", "Mistral"),
    ("glm", "GLM"),
    ("phi", "Phi"),
    ("internlm", "InternLM"),
    ("minicpm", "MiniCPM"),
    ("kimi", "Kimi"),
    ("gpt-oss", "GPT-OSS"),
    ("yi-", "Yi"),
]


_ctx_cache = {}


def read_gguf_ctx(path):
    """Native context length straight out of the GGUF metadata (``<arch>.context_length``)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime, st.st_size)
    if key in _ctx_cache:
        return _ctx_cache[key]

    def rd_str(fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n).decode("utf-8", "replace")

    def rd_val(fh, t):
        if t == 0: return struct.unpack("<B", fh.read(1))[0]
        if t == 1: return struct.unpack("<b", fh.read(1))[0]
        if t == 2: return struct.unpack("<H", fh.read(2))[0]
        if t == 3: return struct.unpack("<h", fh.read(2))[0]
        if t == 4: return struct.unpack("<I", fh.read(4))[0]
        if t == 5: return struct.unpack("<i", fh.read(4))[0]
        if t == 6: return struct.unpack("<f", fh.read(4))[0]
        if t == 7: return struct.unpack("<B", fh.read(1))[0]
        if t == 8: return rd_str(fh)
        if t == 9:
            at = struct.unpack("<I", fh.read(4))[0]
            n = struct.unpack("<Q", fh.read(8))[0]
            return [rd_val(fh, at) for _ in range(n)]
        if t == 10: return struct.unpack("<Q", fh.read(8))[0]
        if t == 11: return struct.unpack("<q", fh.read(8))[0]
        if t == 12: return struct.unpack("<d", fh.read(8))[0]
        raise ValueError("unknown gguf type")

    value = None
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                raise ValueError("not gguf")
            struct.unpack("<I", fh.read(4))[0]          # version
            struct.unpack("<Q", fh.read(8))[0]          # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]
            for _ in range(n_kv):
                name = rd_str(fh)
                vtype = struct.unpack("<I", fh.read(4))[0]
                val = rd_val(fh, vtype)
                if isinstance(val, int) and name.endswith(".context_length"):
                    value = int(val)
    except Exception:
        value = None
    _ctx_cache[key] = value
    return value


def fmt_ctx(n):
    if not n:
        return ""
    if n >= 1048576 and n % 1048576 == 0:
        return f"{n // 1048576}M"
    if n >= 1024:
        return f"{round(n / 1024)}k"
    return str(n)
_lock = threading.Lock()
_hw_cache = {"at": 0.0, "value": None}
_dl = {
    "active": False,
    "repo": "", "file": "", "path": "",
    "done": 0, "total": 0, "speed": 0.0,
    "started": 0.0, "error": "", "finished": False,
}
_dl_cancel = threading.Event()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gb(n):
    return round(n / (1024 ** 3), 2)


def family_of(text):
    low = text.lower()
    for needle, label in FAMILIES:
        if needle in low:
            return label
    return "其他"


def quant_of(text):
    m = QUANT_RE.search(text)
    return m.group(1).upper() if m else "—"


def params_of(text):
    best = None
    for m in PARAM_RE.finditer(text):
        val = float(m.group(1))
        if 0.3 <= val <= 1000:
            best = m.group(1) + "B"
    return best or "—"


def gpu_total_gb():
    hw = hardware()
    return hw["vram_gb"] or 16.0


def hardware():
    """Detect the GPU and its VRAM at runtime — never assume one machine.

    Cached for a few seconds: the dashboard polls lookups often and every call
    used to spawn nvidia-smi.
    """
    now = time.time()
    if _hw_cache["value"] is not None and now - _hw_cache["at"] < 10:
        return _hw_cache["value"]
    info = {"gpu": "unknown", "vram_gb": 0.0, "driver": "", "source": "none"}
    try:
        out = os.popen(
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
        ).read().strip().splitlines()
        if out:
            name, mem, drv = [x.strip() for x in out[0].split(",")[:3]]
            info.update({
                "gpu": name,
                "vram_gb": round(int(mem) / 1024, 2),
                "driver": drv,
                "source": "nvidia-smi",
            })
            _hw_cache.update(at=now, value=info)
            return info
    except Exception:
        pass
    try:  # AMD / Intel / no nvidia driver: DXGI adapter memory
        out = os.popen(
            "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | "
            "Sort-Object AdapterRAM -Descending | Select-Object -First 1).Name\""
        ).read().strip()
        mem = os.popen(
            "powershell -NoProfile -Command \"(Get-CimInstance Win32_VideoController | "
            "Measure-Object -Property AdapterRAM -Maximum).Maximum\""
        ).read().strip()
        if out:
            info.update({
                "gpu": out,
                "vram_gb": round(int(mem or 0) / (1024 ** 3), 2),
                "source": "Win32_VideoController",
            })
    except Exception:
        pass
    _hw_cache.update(at=now, value=info)
    return info


# GB of weights per billion parameters, measured on real GGUF files
QUANT_DENSITY = [
    ("Q8_0", 0.99), ("Q6_K", 0.77), ("Q5_K_M", 0.68), ("Q4_K_M", 0.59),
    ("Q3_K_XL", 0.455), ("Q2_K_XL", 0.37),
]
SANE_QUANTS = 4  # indexes 0..3 are the "don't wreck quality" range
SIZE_CLASSES = [70, 32, 27, 24, 14, 13, 12, 9, 8, 7, 4, 3, 1.5]


def recommend(ctx=65536):
    """Work out what actually fits this GPU, at runtime.

    Everything is derived from the detected VRAM: no per-machine constants.
    """
    hw = hardware()
    total = hw["vram_gb"] or 0.0
    overhead = round(1.9 + (ctx / 1000.0) * 0.019, 2)    # buffers + KV for this ctx
    budget = round(total * 0.965, 2) if total else 0.0   # leave a little for the desktop
    weights = round(max(0.0, budget - overhead), 2) if total else 0.0
    ceiling_weights = round(max(0.0, total * 0.99 - overhead), 2) if total else 0.0

    def best_for(cls, limit_idx, room):
        for q, dens in QUANT_DENSITY[:limit_idx]:
            gb = round(cls * dens, 2)
            if gb <= room:
                return q, gb
        return None, None

    picks = []
    main = None
    for cls in SIZE_CLASSES:
        q, gb = best_for(cls, SANE_QUANTS, weights)      # Q8_0 … Q4_K_M
        if q:
            main = {"tier": "建議", "params_b": cls, "quant": q, "gb": gb,
                    "headroom_gb": round(weights - gb, 2),
                    "reason": f"{cls}B @ {q} fits fully on a {hw['vram_gb']} GB card with "
                              f"{round(weights - gb, 2)} GB to spare."}
            break
    if main:
        picks.append(main)

    # quality ceiling: bigger model, lower quant, right up against the limit
    for cls in SIZE_CLASSES:
        if main and cls <= main["params_b"]:
            continue
        q, gb = best_for(cls, 5, ceiling_weights)        # … down to Q3_K_XL
        if q:
            picks.append({"tier": "品質上限", "params_b": cls, "quant": q, "gb": gb,
                          "headroom_gb": round(ceiling_weights - gb, 2),
                          "reason": f"{cls}B only fits at {q}; it sits close to the VRAM "
                                    f"limit and generates noticeably slower."})
            break

    # speed: same size, lighter quant
    if main:
        gb = round(main["params_b"] * dict(QUANT_DENSITY)["Q4_K_M"], 2)
        if gb <= weights:
            picks.append({"tier": "速度", "params_b": main["params_b"], "quant": "Q4_K_M",
                          "gb": gb, "headroom_gb": round(weights - gb, 2),
                          "reason": f"same {main['params_b']}B model at Q4_K_M: "
                                    f"{round((main['gb'] - gb) / main['gb'] * 100)}% smaller weights, "
                                    f"faster tokens."})

    local_best = None
    for item in scan_local(ctx):
        if item["projector"] or item["fit"]["level"] == "over":
            continue
        try:
            params = float(str(item["params"]).replace("B", ""))
        except ValueError:
            params = 0.0
        gap = abs(params - (main["params_b"] if main else params))
        if local_best is None or gap < local_best[0]:
            local_best = (gap, item)
    return {
        "hardware": hw,
        "ctx": ctx,
        "overhead_gb": overhead,
        "budget_gb": budget,
        "weights_budget_gb": weights,
        "picks": picks,
        "local_best": local_best[1] if local_best else None,
        "query_hint": f"qwen3.5 {main['params_b']}b" if main else "",
    }


def estimate(weight_gb, ctx=65536):
    """Rough VRAM need: weights + compute/driver overhead + KV for the given context.

    Calibrated on a 16 GB RTX 5080 running a 27B Q3 model at 64k context
    (12.24 GB weights -> ~15.4 GB total).
    """
    return round(weight_gb + 1.9 + (ctx / 1000.0) * 0.019, 2)


def fit_of(weight_gb, ctx=65536, total=None):
    total = total or gpu_total_gb()
    need = estimate(weight_gb, ctx)
    if need <= total * 0.92:
        return {"level": "ok", "need": need, "label": "可全載入 GPU"}
    if need <= total * 0.99:
        return {"level": "tight", "need": need, "label": "勉強塞得下"}
    return {"level": "over", "need": need, "label": "超出顯存"}


# --------------------------------------------------------------------------- #
# local catalogue
# --------------------------------------------------------------------------- #
def scan_local(ctx=65536):
    os.makedirs(MODELS_DIR, exist_ok=True)
    total = gpu_total_gb()          # one lookup for the whole scan
    items = []
    active = active_model_path()
    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            if not fname.lower().endswith(".gguf"):
                continue
            full = os.path.join(root, fname)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, MODELS_DIR)
            items.append({
                "name": fname,
                "rel": rel,
                "path": full,
                "gb": _gb(size),
                "bytes": size,
                "quant": quant_of(fname),
                "params": params_of(fname),
                "family": family_of(rel),
                "projector": "mmproj" in fname.lower(),
                "ctx_train": read_gguf_ctx(full),
                "fit": fit_of(_gb(size), ctx, total),
                "active": os.path.normcase(full) == os.path.normcase(active or ""),
            })
    items.sort(key=lambda m: (m["projector"], m["name"].lower()))
    return items


def active_model_path():
    try:
        with open(ACTIVE_FILE, "r", encoding="utf-8") as fh:
            path = json.load(fh).get("path") or ""
        if path and os.path.exists(path):
            return path
    except (OSError, ValueError):
        pass
    return os.environ.get("HERTA_MODEL", "")


def is_projector(path):
    return "mmproj" in os.path.basename(path).lower()


def set_active(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if is_projector(path):
        raise ValueError("視覺投影檔（mmproj）不能當成模型載入")
    os.makedirs(os.path.dirname(ACTIVE_FILE) or ".", exist_ok=True)
    with open(ACTIVE_FILE, "w", encoding="utf-8") as fh:
        json.dump({"path": path}, fh, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------- #
# hugging face
# --------------------------------------------------------------------------- #
def _get_json(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "herta-station/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


HF_SORTS = {"downloads": "downloads", "likes": "likes", "updated": "lastModified"}


def hf_search(query, limit=30, sort="downloads"):
    query = (query or "").strip()
    if not query:
        return []
    sort = HF_SORTS.get(sort, "downloads")
    qs = urllib.parse.urlencode({
        "search": query,
        "filter": "gguf",
        "sort": sort,
        "direction": -1,
        "limit": max(1, min(100, int(limit))),
    })
    data = _get_json(f"https://huggingface.co/api/models?{qs}")
    out = []
    for row in data:
        rid = row.get("id", "")
        out.append({
            "id": rid,
            "downloads": row.get("downloads", 0) or 0,
            "likes": row.get("likes", 0) or 0,
            "updated": (row.get("lastModified") or "")[:10],
            "family": family_of(rid),
            "params": params_of(rid.split("/")[-1]),
            "quant": quant_of(rid),
            "task": row.get("pipeline_tag") or "",
        })
    return out


_cache = {}


def hf_files(repo, ctx=65536):
    key = (repo, ctx)
    if key in _cache and time.time() - _cache[key][0] < 300:
        return _cache[key][1]
    qs = urllib.parse.urlencode({"blobs": "true"})
    data = _get_json(f"https://huggingface.co/api/models/{repo}?{qs}")
    files = []
    for sib in data.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.lower().endswith(".gguf"):
            continue
        size = sib.get("size") or 0
        gb = _gb(size) if size else 0
        files.append({
            "name": name,
            "bytes": size,
            "gb": gb,
            "quant": quant_of(name),
            "params": params_of(name),
            "fit": fit_of(gb, ctx) if size else {"level": "unknown", "need": 0, "label": "未知"},
        })
    files.sort(key=lambda f: f["bytes"])
    _cache[key] = (time.time(), files)
    if len(_cache) > 50:                      # keep the memo table bounded
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    return files


# --------------------------------------------------------------------------- #
# downloader
# --------------------------------------------------------------------------- #
def download_state():
    with _lock:
        snap = dict(_dl)
    if snap["active"] and snap["started"]:
        snap["elapsed"] = round(time.time() - snap["started"], 1)
        snap["percent"] = round(snap["done"] / snap["total"] * 100, 1) if snap["total"] else 0.0
    else:
        snap["percent"] = 100.0 if snap["finished"] and not snap["error"] else 0.0
        snap["elapsed"] = 0
    return snap


def _download_worker(repo, filename):
    global _dl
    target = os.path.join(MODELS_DIR, repo.replace("/", "__"), os.path.basename(filename))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}?download=true"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "herta-station/1.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with _lock:
                _dl.update({"total": total, "done": 0})
            last = time.time()
            with open(part, "wb") as fh:
                while True:
                    if _dl_cancel.is_set():
                        raise RuntimeError("已取消")
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    with _lock:
                        _dl["done"] += len(chunk)
                        now = time.time()
                        if now - last >= 0.5:
                            _dl["speed"] = _dl["done"] / max(0.001, now - _dl["started"])
                            last = now
        os.replace(part, target)
        with _lock:
            _dl.update({"active": False, "finished": True, "path": target, "speed": 0.0,
                        "total": os.path.getsize(target), "done": os.path.getsize(target)})
        if not is_projector(target):     # a projector is not loadable as a model
            set_active(target)
    except Exception as exc:
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        with _lock:
            _dl.update({"active": False, "finished": True, "error": str(exc)})


def start_download(repo, filename):
    global _dl
    with _lock:
        if _dl["active"]:
            return {"ok": False, "message": "已有下載進行中"}
        _dl = {"active": True, "repo": repo, "file": filename, "path": "",
               "done": 0, "total": 0, "speed": 0.0, "started": time.time(),
               "error": "", "finished": False}
    _dl_cancel.clear()
    threading.Thread(target=_download_worker, args=(repo, filename), daemon=True).start()
    return {"ok": True, "message": f"開始下載 {filename}"}


def cancel_download():
    _dl_cancel.set()
    return {"ok": True, "message": "已請求取消"}


def reset_download():
    """Clear a finished/errored download so the UI can drop the progress bar."""
    global _dl
    with _lock:
        if _dl["active"]:
            return {"ok": False, "message": "下載進行中，無法重設"}
        _dl = {"active": False, "repo": "", "file": "", "path": "",
               "done": 0, "total": 0, "speed": 0.0, "started": 0.0,
               "error": "", "finished": False}
    return {"ok": True, "message": "已清除下載狀態"}

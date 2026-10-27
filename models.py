# -*- coding: utf-8 -*-
"""Model library for Herta Station: local GGUF catalogue, Hugging Face search,
and a background downloader.

Everything here is stdlib-only so it survives PyInstaller's one-file build.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import settings as appsettings
import hostos

RES_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = appsettings.DATA_DIR

MODELS_DIR = hostos.default_models_dir()
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


_meta_cache = {}


def read_gguf_meta(path):
    """Attention geometry from the GGUF header, used to size the KV cache.

    Returns ``{ctx_train, layers, heads, kv_heads, head_dim}`` (any of them may be
    missing when a file does not carry the key).
    """
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (path, st.st_mtime, st.st_size)
    if key in _meta_cache:
        return _meta_cache[key]

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

    meta = {}
    hybrid = False
    keys = {
        "context_length": "ctx_train",
        "block_count": "layers",
        "attention.head_count": "heads",
        "attention.head_count_kv": "kv_heads",
        "attention.key_length": "head_dim_k",
        "attention.value_length": "head_dim_v",
        "embedding_length": "embed",
    }
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
                for suffix, field in keys.items():
                    if name.endswith("." + suffix) and isinstance(val, int):
                        meta.setdefault(field, int(val))
                if ".ssm." in name or ".linear_attn." in name or name.endswith(".conv_kernel"):
                    hybrid = True
    except Exception:
        meta = {}

    heads = meta.get("heads") or 0
    kv_heads = meta.get("kv_heads") or heads
    embed = meta.get("embed") or 0
    head_dim = meta.get("head_dim_k") or (embed // heads if heads and embed else 0)
    out = {
        "ctx_train": meta.get("ctx_train"),
        "layers": meta.get("layers"),
        "heads": heads or None,
        "kv_heads": kv_heads or None,
        "head_dim": head_dim or None,
        "hybrid": hybrid,
    }
    _meta_cache[key] = out
    return out


def read_gguf_ctx(path):
    """Native context length straight out of the GGUF metadata."""
    return read_gguf_meta(path).get("ctx_train")


def fmt_ctx(n):
    if not n:
        return ""
    if n >= 1048576 and n % 1048576 == 0:
        return f"{n // 1048576}M"
    if n >= 1024:
        return f"{round(n / 1024)}k"
    return str(n)


# Past this much VRAM Windows starts paging and throughput collapses ~10x.
VRAM_CLIFF_GB = 15.3

# Bytes per KV element for the quantised cache types llama.cpp accepts.
KV_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0625,     # 8.5 bits
    "q5_1": 0.75,       # 5.5 bits
    "q5_0": 0.6875,     # 5 bits + fp16 delta
    "q4_1": 0.625,      # 4.5 bits
    "q4_0": 0.5625,     # 4.5 bits, fp16 scale
}


def kv_for_context(ctx):
    """Preferred KV cache type for a context length.

    Bigger contexts need a smaller cache; this is only the opening guess — the
    controller re-measures VRAM after the model loads and shrinks the context if
    the machine disagrees.
    """
    if ctx <= 65536:
        return "f16"
    if ctx <= 131072:
        return "q8_0"
    return "q4_0"


def context_plan(path, total_gb=None, ctx_train=None):
    """Largest context this machine can give one model, plus the matching KV type.

    KV bytes come from the GGUF geometry (layers x kv heads x head dim x element
    size). Hybrid SSM/attention models only keep a cache for their attention
    layers, which the layer count does not show, so those are scaled by a factor
    measured on this box (9B/27B Qwen3.5 = 0.3). The controller still re-measures
    VRAM after load and shrinks the context if the estimate was optimistic.
    """
    meta = read_gguf_meta(path)
    train = int(ctx_train or meta.get("ctx_train") or 32768)
    try:
        weights = _gb(os.path.getsize(path))
    except OSError:
        weights = 0.0
    total = float(total_gb or gpu_total_gb() or 0)
    limit = vram_limit(total)
    # 0.85 keeps us clear of the paging cliff even when the estimate is optimistic;
    # the controller still re-measures after load and shrinks further if needed.
    room = max(0.0, (limit - weights - 1.5) * 0.85)

    layers = meta.get("layers") or 0
    kv_heads = meta.get("kv_heads") or 0
    head_dim = meta.get("head_dim") or 0
    elements = 2 * layers * kv_heads * head_dim          # K + V, per token
    if meta.get("hybrid"):
        elements = int(elements * 0.3)

    def max_ctx(kv):
        if not elements or not room:
            return 2048
        per_token = elements * KV_BYTES[kv]
        return max(2048, min(train, int(room * 1024 ** 3 / per_token) // 2048 * 2048))

    for kv in ("f16", "q8_0", "q4_0"):
        if max_ctx(kv) >= train:
            ctx = train
            break
    else:
        kv, ctx = "q4_0", max_ctx("q4_0")
    return {
        "ctx": ctx,
        "ctk": kv,
        "ctv": kv,
        "ctx_train": train,
        "weights_gb": round(weights, 2),
        "vram_limit_gb": round(limit, 2),
    }


_lock = threading.Lock()
_hw_cache = {"at": 0.0, "value": None}
_dl = {
    "active": False,
    "repo": "", "file": "", "path": "",
    "done": 0, "total": 0, "speed": 0.0,
    "started": 0.0, "error": "", "finished": False,
    "preset": "", "file_index": 0, "file_count": 0, "file_name": "",
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
    """Detect the GPU and its memory at runtime — never assume one machine.

    Cached for a few seconds: the dashboard polls lookups often and every call
    used to spawn nvidia-smi. Delegated to hostos so Windows, Linux (NVIDIA/AMD
    sysfs) and macOS (unified memory) all report something usable.
    """
    now = time.time()
    if _hw_cache["value"] is not None and now - _hw_cache["at"] < 10:
        return _hw_cache["value"]
    info = hostos.gpu_info()
    _hw_cache.update(at=now, value=info)
    return info


# GB of weights per billion parameters, measured on real GGUF files
QUANT_DENSITY = [
    ("Q8_0", 0.99), ("Q6_K", 0.77), ("Q5_K_M", 0.68), ("Q4_K_M", 0.59),
    ("Q3_K_XL", 0.455), ("Q2_K_XL", 0.37),
]
SANE_QUANTS = 4  # indexes 0..3 are the "don't wreck quality" range
SIZE_CLASSES = [70, 32, 27, 24, 14, 13, 12, 9, 8, 7, 4, 3, 1.5]


def vram_limit(total_gb=None):
    """Usable VRAM for one model, in GB.

    On a 16 GB card under Windows, WDDM starts paging as soon as the whole
    budget is carved up, so the limit is pinned just under the measured cliff.
    Bigger cards and the Linux / macOS drivers keep the full 96.5 % budget.
    """
    total = float(total_gb if total_gb is not None else (gpu_total_gb() or 0))
    if not total:
        return VRAM_CLIFF_GB
    limit = total * 0.965
    if hostos.IS_WIN and total < 17.0:
        limit = min(limit, VRAM_CLIFF_GB)
    return limit


def recommend(ctx=65536):
    """Work out what actually fits this GPU, at runtime.

    Everything is derived from the detected VRAM: no per-machine constants.
    """
    hw = hardware()
    total = hw["vram_gb"] or 0.0
    overhead = round(1.9 + (ctx / 1000.0) * 0.019, 2)    # buffers + KV for this ctx
    budget = round(vram_limit(total), 2) if total else 0.0    # leave a little for the desktop
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
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def shard_of(name):
    """(index, count) for a split GGUF; (1, 1) when it is a single file."""
    m = SHARD_RE.search(name)
    if not m:
        return 1, 1
    return int(m.group(1)), int(m.group(2))


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
            index, count = shard_of(fname)
            if index > 1:               # only the first shard is a loadable entry
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
                "shards": count,
                "shard_gb": round(sum(
                    os.path.getsize(os.path.join(root, n))
                    for n in files if SHARD_RE.sub("", n) == SHARD_RE.sub("", fname)), 2) if count > 1 else _gb(size),
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
        if path and os.path.isfile(path):
            return path
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return os.environ.get("HERTA_MODEL", "")


def is_projector(path):
    return "mmproj" in os.path.basename(path).lower()


def set_active(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".gguf"):
        raise ValueError("only GGUF model files can be selected")
    if is_projector(path):
        raise ValueError("視覺投影檔（mmproj）不能當成模型載入")
    os.makedirs(os.path.dirname(ACTIVE_FILE) or ".", exist_ok=True)
    appsettings.atomic_json(ACTIVE_FILE, {"path": path})
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
# curated presets — one click, no typing, no links to copy
# --------------------------------------------------------------------------- #
# ``files`` holds every shard of a split GGUF; ``entry`` is the shard llama.cpp
# must be pointed at (it discovers the rest by naming convention).
PRESETS = [
    {
        "id": "strata-flash-next-q2",
        "name": "Qwen3.8-Flash-Next Q2_0",
        "sub": "Strata 官方推薦 · ISTA-DASLab",
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "entry": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "files": [
            {"path": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", "size": 37623740192},
            {"path": "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", "size": 28800138432},
        ],
        "quant": "Q2_0",
        "params_label": "MoE 旗艦",
        "params_b": 0,
        "tags": ["strata", "flagship"],
        "homepage": "https://github.com/Niko1221/Strata",
        "note": "Strata 引擎的預設旗艦模型。權重 66 GB，16 GB 顯示卡必須靠 SSD 換頁，速度只有個位數 tok/s；建議 128 GB 記憶體以上的機器。",
    },
    {
        "id": "qwen38-27b-q3",
        "name": "Qwen3.8-27B UD-Q3_K_XL",
        "sub": "unsloth · 本機甜點",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "entry": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        "files": [{"path": "Qwen3.8-27B-UD-Q3_K_XL.gguf", "size": 13146393504}],
        "quant": "Q3_K_XL",
        "params_label": "27B",
        "params_b": 27,
        "tags": [],
        "note": "16 GB 顯示卡上品質與速度的平衡點：約 15.3 GB 顯存、47 tok/s、可開 60k 上下文。",
    },
    {
        "id": "qwen35-9b-q4",
        "name": "Qwen3.5-9B UD-Q4_K_XL",
        "sub": "unsloth · 快速全能",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "entry": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "files": [{"path": "Qwen3.5-9B-UD-Q4_K_XL.gguf", "size": 5966095584}],
        "quant": "Q4_K_XL",
        "params_label": "9B",
        "params_b": 9,
        "tags": ["light"],
        "note": "6 GB 權重留下大量顯存放 KV：同樣的卡可以跑滿 256k 上下文，輸出約 80 tok/s。",
    },
    {
        "id": "qwen35-4b-q4",
        "name": "Qwen3.5-4B UD-Q4_K_XL",
        "sub": "unsloth · 輕量首選",
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "entry": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "files": [{"path": "Qwen3.5-4B-UD-Q4_K_XL.gguf", "size": 2912109728}],
        "quant": "Q4_K_XL",
        "params_label": "4B",
        "params_b": 4,
        "tags": ["light"],
        "note": "3 GB 權重，內顯或筆電也能跑；在 5080 上幾乎是即時回應。",
    },
    {
        "id": "qwen36-35b-a3b-q3",
        "name": "Qwen3.6-35B-A3B UD-Q3_K_XL",
        "sub": "unsloth · MoE 高效",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "entry": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
        "files": [{"path": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf", "size": 16845511648}],
        "quant": "Q3_K_XL",
        "params_label": "35B-A3B",
        "params_b": 35,
        "tags": [],
        "note": "MoE 只啟用約 3B 參數，速度接近小模型、知識量接近大模型；需要 24 GB 以上顯存才好跑。",
    },
    {
        "id": "qwen3-1_7b-q4",
        "name": "Qwen3-1.7B UD-Q4_K_XL",
        "sub": "unsloth · 極輕量",
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "entry": "Qwen3-1.7B-UD-Q4_K_XL.gguf",
        "files": [{"path": "Qwen3-1.7B-UD-Q4_K_XL.gguf", "size": 1132952128}],
        "quant": "Q4_K_XL",
        "params_label": "1.7B",
        "params_b": 2,
        "tags": ["light"],
        "note": "1 GB 出頭，當成翻譯機、分類器或 agent 的便宜副手最合適。",
    },
    {
        "id": "gemma3-4b-qat",
        "name": "Gemma 3 4B IT（QAT q4_0）",
        "sub": "Google · 官方量化",
        "repo": "google/gemma-3-4b-it-qat-q4_0-gguf",
        "entry": "gemma-3-4b-it-q4_0.gguf",
        "files": [
            {"path": "gemma-3-4b-it-q4_0.gguf", "size": 3155051328},
            {"path": "mmproj-model-f16-4B.gguf", "size": 851251104},
        ],
        "quant": "Q4_0",
        "params_label": "4B",
        "params_b": 4,
        "tags": ["light", "vision"],
        "note": "Google 官方 QAT 量化，附視覺投影檔，可以直接看圖。",
    },
    {
        "id": "llama32-3b-q4",
        "name": "Llama 3.2 3B Instruct Q4_K_M",
        "sub": "bartowski · 相容性最好",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "entry": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "files": [{"path": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size": 2019377696}],
        "quant": "Q4_K_M",
        "params_label": "3B",
        "params_b": 3,
        "tags": ["light"],
        "note": "2 GB，老卡與 CPU 都能跑，工具鏈相容性最好。",
    },
]


def preset_dir(preset):
    return os.path.join(MODELS_DIR, preset["repo"].replace("/", "__"))


def preset_files(preset):
    """Absolute path + size for every file a preset needs."""
    base = preset_dir(preset)
    out = []
    for item in preset["files"]:
        rel = item["path"]
        out.append({"path": os.path.join(base, os.path.basename(rel)),
                    "rel": rel, "size": int(item["size"])})
    return out


def _free_gb(path):
    try:
        usage = shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path))
        return round(usage.free / 1e9, 1)
    except OSError:
        return None


def preset_fit(weight_gb, ctx=65536, total=None):
    """Verdict for a preset we have not downloaded yet.

    Without the real GGUF header we cannot size the KV cache exactly, so this
    answers the question that matters: do the weights themselves fit, and does
    the full context fit as well?
    """
    total = total or gpu_total_gb()
    limit = vram_limit(total)
    fixed = round(weight_gb + 1.9, 2)          # weights + driver/compute overhead
    full = estimate(weight_gb, ctx)
    if full <= total * 0.92:
        return {"level": "ok", "need": full, "weights_only": fixed, "ctx": ctx,
                "label": "可全載入 GPU"}
    if fixed <= limit:
        return {"level": "tight", "need": full, "weights_only": fixed, "ctx": ctx,
                "label": "可載入，需縮短上下文"}
    return {"level": "over", "need": full, "weights_only": fixed, "ctx": ctx,
            "label": "超出顯存"}


def preset_state(ctx=65536, total=None):
    """Everything the UI needs to render one card: sizes, fit, disk, install state."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    total = total or gpu_total_gb()
    free = _free_gb(MODELS_DIR)
    active = active_model_path()
    rows = []
    for preset in PRESETS:
        files = preset_files(preset)
        have = 0
        done_bytes = 0
        for f in files:
            try:
                size = os.path.getsize(f["path"])
            except OSError:
                size = 0
            if size == f["size"] or (size > 0 and abs(size - f["size"]) < 1024 * 1024):
                have += 1
                done_bytes += size
            elif size > 0:
                done_bytes += size
        entry_path = os.path.join(preset_dir(preset), os.path.basename(preset["entry"]))
        weight_gb = round(sum(f["size"] for f in files if not f["path"].lower().endswith("mmproj.gguf")
                              and "mmproj" not in os.path.basename(f["rel"]).lower()) / 1e9, 2)
        disk_gb = round(sum(f["size"] for f in files) / 1e9, 2)
        rows.append({
            "id": preset["id"],
            "name": preset["name"],
            "sub": preset.get("sub", ""),
            "repo": preset["repo"],
            "quant": preset.get("quant", ""),
            "params": preset.get("params_label", ""),
            "params_b": preset.get("params_b", 0),
            "note": preset.get("note", ""),
            "homepage": preset.get("homepage", ""),
            "tags": preset.get("tags", []),
            "files": len(files),
            "weight_gb": weight_gb,
            "disk_gb": disk_gb,
            "installed": have == len(files),
            "partial": 0 < have < len(files),
            "downloaded_gb": round(done_bytes / 1e9, 2),
            "entry": entry_path,
            "entry_exists": os.path.isfile(entry_path),
            "active": os.path.normcase(entry_path) == os.path.normcase(active or ""),
            "fit": preset_fit(weight_gb, ctx, total),
            "est": estimate(weight_gb, ctx),
            "disk_ok": (free is None) or (free > disk_gb * 1.1),
            "free_gb": free,
        })
    return rows


def find_preset(preset_id):
    for preset in PRESETS:
        if preset["id"] == preset_id:
            return preset
    return None


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


def _rust_bin():
    """The optional Rust helper (multi-connection, resumable fetch)."""
    name = "herta-rs" + (".exe" if hostos.IS_WIN else "")
    candidates = [os.environ.get("HERTA_RS", "")]
    roots = [getattr(sys, "_MEIPASS", ""), hostos.app_dir(),
             os.path.dirname(os.path.abspath(sys.executable)),
             os.path.dirname(os.path.abspath(__file__)),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")]
    for root in roots:
        if not root:
            continue
        candidates += [os.path.join(root, name), os.path.join(root, "bin", name)]
    # a cargo build in the source tree counts as a dev build
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "tools", "herta-rs", "target", "release", name))
    for path in candidates:
        if path and os.path.isfile(path) and (hostos.IS_WIN or os.access(path, os.X_OK)):
            return path
    found = shutil.which("herta-rs")
    return found or None


RUST_BIN = _rust_bin()


def _rust_fetch(url, part, expected, tick):
    """Download with the Rust helper. Returns True when the file is complete."""
    if not RUST_BIN:
        return False
    conns = "6"
    cmd = [RUST_BIN, "download", url, part, "--connections", conns]
    if os.path.exists(part):
        cmd.append("--resume")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, **hostos.spawn_kwargs())
    except OSError:
        return False
    ok = False
    try:
        for line in proc.stdout or []:
            if _dl_cancel.is_set():
                hostos.kill_tree(proc.pid)
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if data.get("phase") == "done":
                ok = True
            elif "done" in data:
                tick(int(data["done"]), int(data.get("total") or expected or 0))
        proc.wait(timeout=30)
    finally:
        try:
            if proc.poll() is None:
                hostos.kill_tree(proc.pid)
        except OSError:
            pass
    return ok and proc.returncode == 0


def _py_fetch(url, part, expected, tick):
    req = urllib.request.Request(url, headers={"User-Agent": "herta-station/1.2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or expected or 0)
        done = 0
        with open(part, "wb") as fh:
            while True:
                if _dl_cancel.is_set():
                    raise RuntimeError("已取消")
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                tick(done, total)


def _download_worker(jobs, preset_id=""):
    """Download every job (one shard each) in order, then mark the model active."""
    error_code = "download_failed"
    tick_state = {"last": 0.0, "base": 0, "seen_total": 0}

    def tick(done, total):
        now = time.time()
        if total:
            tick_state["seen_total"] = int(total)
        with _lock:
            _dl["done"] = tick_state["base"] + done
            if total:
                _dl["total"] = max(_dl["total"], tick_state["base"] + total)
            if now - tick_state["last"] >= 0.4:
                _dl["speed"] = _dl["done"] / max(0.001, now - _dl["started"])
                tick_state["last"] = now

    try:
        for index, job in enumerate(jobs):
            target = job["target"]
            tick_state["seen_total"] = 0
            part = target + ".part"
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with _lock:
                _dl.update({"file_index": index + 1, "file_name": os.path.basename(target)})
            if os.path.isfile(target) and job.get("size") and \
                    abs(os.path.getsize(target) - job["size"]) < 1024 * 1024:
                tick_state["base"] += os.path.getsize(target)
                with _lock:
                    _dl["done"] = tick_state["base"]
                continue
            url = (f"https://huggingface.co/{job['repo']}/resolve/main/"
                   f"{urllib.parse.quote(job['path'])}?download=true")
            if not _rust_fetch(url, part, job.get("size"), tick):
                if _dl_cancel.is_set():
                    error_code = "download_cancelled"
                    raise RuntimeError("已取消")
                tick_state["base"] += _part_size(part)
                with _lock:
                    _dl["done"] = tick_state["base"]
                _py_fetch(url, part, job.get("size"), tick)
            size = os.path.getsize(part)
            expected = job.get("size") or tick_state["seen_total"] or 0
            # exact size when the declaration is authoritative; allow 0.1 %
            # drift (capped at 1 MiB) in case a repo re-uploads a file
            tol = min(1024 * 1024, int(expected * 0.001)) if expected else 0
            if not size or (expected and abs(size - expected) > tol):
                error_code = "download_incomplete"
                raise RuntimeError(f"incomplete download: received {size} of {expected} bytes")
            with open(part, "rb") as fh:
                if fh.read(4) != b"GGUF":
                    error_code = "download_invalid"
                    raise RuntimeError("download is not a GGUF file")
            if _dl_cancel.is_set():
                error_code = "download_cancelled"
                raise RuntimeError("已取消")
            os.replace(part, target)
            tick_state["base"] += size
            with _lock:
                _dl["done"] = tick_state["base"]
        entry = jobs[0]["target"]
        with _lock:
            if not is_projector(entry):      # a projector is not loadable as a model
                error_code = "selection_write_failed"
                set_active(entry)
            _dl.update({"active": False, "finished": True, "path": entry, "speed": 0.0})
    except Exception as exc:
        for job in jobs:
            for leftover in [job["target"] + ".part"] + \
                    [job["target"] + f".part.c{i}" for i in range(16)]:
                try:
                    os.remove(leftover)
                except OSError:
                    pass
        with _lock:
            _dl.update({"active": False, "finished": True, "error": str(exc),
                        "error_code": error_code, "speed": 0.0, "preset": preset_id})


def _part_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def start_download(repo, filename):
    global _dl
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or "") or any(
            component in (".", "..") for component in repo.split("/")):
        return {"ok": False, "message": "invalid Hugging Face repository", "error_code": "invalid_download"}
    if not filename or not filename.lower().endswith(".gguf") or "\\" in filename or any(
            component in ("", ".", "..") for component in filename.split("/")):
        return {"ok": False, "message": "invalid GGUF filename", "error_code": "invalid_download"}
    target = os.path.join(MODELS_DIR, repo.replace("/", "__"), os.path.basename(filename))
    jobs = [{"repo": repo, "path": filename, "target": target, "size": 0}]
    return _begin(jobs, {"repo": repo, "file": filename}, "", f"開始下載 {filename}")


def start_preset_download(preset_id):
    """One click: queue every shard of a curated preset."""
    preset = find_preset(preset_id)
    if not preset:
        return {"ok": False, "message": "unknown preset", "error_code": "invalid_download"}
    jobs = []
    for item in preset_files(preset):
        jobs.append({"repo": preset["repo"], "path": item["rel"],
                     "target": item["path"], "size": item["size"]})
    meta = {"repo": preset["repo"], "file": preset["entry"]}
    return _begin(jobs, meta, preset["id"], f"開始下載 {preset['name']}")


def _begin(jobs, meta, preset_id, message):
    global _dl
    with _lock:
        if _dl["active"]:
            return {"ok": False, "message": "已有下載進行中", "error_code": "download_busy"}
        _dl = {"active": True, "repo": meta["repo"], "file": meta["file"], "path": "",
               "done": 0, "total": sum(j.get("size") or 0 for j in jobs),
               "speed": 0.0, "started": time.time(), "error": "", "finished": False,
               "preset": preset_id, "file_index": 1, "file_count": len(jobs),
               "file_name": os.path.basename(jobs[0]["target"])}
        _dl_cancel.clear()
    threading.Thread(target=_download_worker, args=(jobs, preset_id), daemon=True).start()
    return {"ok": True, "message": message}


def cancel_download():
    with _lock:
        if _dl["active"]:
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
               "error": "", "finished": False,
               "preset": "", "file_index": 0, "file_count": 0, "file_name": ""}
    return {"ok": True, "message": "已清除下載狀態"}

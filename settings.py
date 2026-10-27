# -*- coding: utf-8 -*-
"""Persisted application settings for Herta Station.

Two groups:
  ui      — client-side preferences (UI scale, background animation)
  server  — llama-server launch parameters (context, KV type, port, …)

Environment variables still work as *defaults*; anything saved here wins, because
the settings panel is the visible source of truth for the user.
"""

import json
import math
import os
import tempfile
import threading

import hostos

# Per-user writable location: beside the exe on Windows (portable), XDG on Linux,
# Application Support on macOS — a .app bundle itself may be read-only.
DATA_DIR = os.path.abspath(os.environ.get("HERTA_DATA_DIR") or hostos.app_dir())
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
_lock = threading.RLock()

DEFAULTS = {
    "ui": {
        "theme": "herta",      # herta | ollama | xp
        "lang": "zh-Hant",     # zh-Hant | zh-Hans | en
        "hfSort": "downloads", # downloads | likes — Hugging Face result ordering
        "scale": 1.0,          # 0.8 … 1.35, applied as CSS zoom
        "animations": True,    # starfield / pulsing highlights
    },
    "server": {
        "ctx": int(os.environ.get("HERTA_CTX", "65536")),
        "ctk": os.environ.get("HERTA_CTK", "q4_0"),
        "ctv": os.environ.get("HERTA_CTV", "q4_0"),
        "parallel": int(os.environ.get("HERTA_PARALLEL", "1")),
        "bind": os.environ.get("HERTA_BIND", "0.0.0.0"),
        "port": int(os.environ.get("HERTA_PORT", "11433")),
        "apiKey": os.environ.get("HERTA_API_KEY", ""),
        "ngl": int(os.environ.get("HERTA_NGL", "99")),
        "autoPlan": True,
        "anthropicEnabled": True,
        "anthropicPort": int(os.environ.get("HERTA_ANTHROPIC_PORT", "11435")),
    },
}

KV_TYPES = {"f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"}
BINDS = {"0.0.0.0", "127.0.0.1"}
THEMES = {"herta", "ollama", "xp"}
LANGS = {"zh-Hant", "zh-Hans", "en"}
HF_SORTS = {"downloads", "likes"}


def _merge(base, patch):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (patch if isinstance(patch, dict) else {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key].update({k: v for k, v in value.items() if k in out[key]})
    return out


def _clamp_int(value, low, high, fallback):
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return max(low, min(high, n))


def _bool(value, fallback):
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        value = str(value).strip().lower()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
    return fallback


def validate(data):
    """Coerce everything into a safe shape; bad input falls back to defaults."""
    data = data if isinstance(data, dict) else {}
    ui = data.get("ui") if isinstance(data.get("ui"), dict) else {}
    srv = data.get("server") if isinstance(data.get("server"), dict) else {}

    try:
        scale = float(ui.get("scale", DEFAULTS["ui"]["scale"]))
    except (TypeError, ValueError):
        scale = DEFAULTS["ui"]["scale"]
    if not math.isfinite(scale):
        scale = DEFAULTS["ui"]["scale"]
    ui = {"theme": str(ui.get("theme", "herta")) if str(ui.get("theme", "herta")) in THEMES else "herta",
          "lang": str(ui.get("lang", "zh-Hant")) if str(ui.get("lang", "zh-Hant")) in LANGS else "zh-Hant",
          "hfSort": str(ui.get("hfSort", "downloads")) if str(ui.get("hfSort", "downloads")) in HF_SORTS else "downloads",
          "scale": round(max(0.8, min(1.35, scale)), 2),
          "animations": _bool(ui.get("animations"), True)}

    ctk = str(srv.get("ctk", DEFAULTS["server"]["ctk"]))
    ctv = str(srv.get("ctv", DEFAULTS["server"]["ctv"]))
    bind = str(srv.get("bind", DEFAULTS["server"]["bind"]))
    srv = {
        "ctx": _clamp_int(srv.get("ctx"), 2048, 1048576, DEFAULTS["server"]["ctx"]),
        "ctk": ctk if ctk in KV_TYPES else DEFAULTS["server"]["ctk"],
        "ctv": ctv if ctv in KV_TYPES else DEFAULTS["server"]["ctv"],
        "parallel": _clamp_int(srv.get("parallel"), 1, 8, DEFAULTS["server"]["parallel"]),
        "bind": bind if bind in BINDS else DEFAULTS["server"]["bind"],
        "port": _clamp_int(srv.get("port"), 1024, 65535, DEFAULTS["server"]["port"]),
        "apiKey": str(srv.get("apiKey", "") or "").strip(),
        "ngl": _clamp_int(srv.get("ngl"), 0, 999, DEFAULTS["server"]["ngl"]),
        "autoPlan": _bool(srv.get("autoPlan"), True),
        "anthropicEnabled": _bool(srv.get("anthropicEnabled"), True),
        "anthropicPort": _clamp_int(srv.get("anthropicPort"), 1024, 65535,
                                    DEFAULTS["server"]["anthropicPort"]),
    }
    return {"ui": ui, "server": srv}


def load():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        stored = {}
    return validate(_merge(DEFAULTS, stored))


def atomic_json(path, data):
    """Replace only after the complete JSON is flushed; propagate write errors."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="herta-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save(patch):
    if not isinstance(patch, dict) or any(
            key in patch and not isinstance(patch[key], dict) for key in ("ui", "server")):
        raise ValueError("settings, ui and server must be JSON objects")
    with _lock:
        data = validate(_merge(load(), patch))
        atomic_json(SETTINGS_FILE, data)
        return data

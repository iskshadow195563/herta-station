# -*- coding: utf-8 -*-
"""Persisted application settings for Herta Station.

Two groups:
  ui      — client-side preferences (UI scale, background animation)
  server  — llama-server launch parameters (context, KV type, port, …)

Environment variables still work as *defaults*; anything saved here wins, because
the settings panel is the visible source of truth for the user.
"""

import json
import os
import sys

DATA_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

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
    for key, value in (patch or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key].update({k: v for k, v in value.items() if k in out[key]})
        elif key in out:
            out[key] = value
    return out


def _clamp_int(value, low, high, fallback):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, n))


def validate(data):
    """Coerce everything into a safe shape; bad input falls back to defaults."""
    ui = data.get("ui") or {}
    srv = data.get("server") or {}

    try:
        scale = float(ui.get("scale", DEFAULTS["ui"]["scale"]))
    except (TypeError, ValueError):
        scale = DEFAULTS["ui"]["scale"]
    ui = {"theme": str(ui.get("theme", "herta")) if str(ui.get("theme", "herta")) in THEMES else "herta",
          "lang": str(ui.get("lang", "zh-Hant")) if str(ui.get("lang", "zh-Hant")) in LANGS else "zh-Hant",
          "hfSort": str(ui.get("hfSort", "downloads")) if str(ui.get("hfSort", "downloads")) in HF_SORTS else "downloads",
          "scale": round(max(0.8, min(1.35, scale)), 2),
          "animations": bool(ui.get("animations", True))}

    ctk = str(srv.get("ctk", DEFAULTS["server"]["ctk"]))
    ctv = str(srv.get("ctv", DEFAULTS["server"]["ctv"]))
    bind = str(srv.get("bind", DEFAULTS["server"]["bind"]))
    srv = {
        "ctx": _clamp_int(srv.get("ctx"), 2048, 262144, DEFAULTS["server"]["ctx"]),
        "ctk": ctk if ctk in KV_TYPES else DEFAULTS["server"]["ctk"],
        "ctv": ctv if ctv in KV_TYPES else DEFAULTS["server"]["ctv"],
        "parallel": _clamp_int(srv.get("parallel"), 1, 8, DEFAULTS["server"]["parallel"]),
        "bind": bind if bind in BINDS else DEFAULTS["server"]["bind"],
        "port": _clamp_int(srv.get("port"), 1024, 65535, DEFAULTS["server"]["port"]),
        "apiKey": str(srv.get("apiKey", "") or "").strip(),
        "ngl": _clamp_int(srv.get("ngl"), 0, 999, DEFAULTS["server"]["ngl"]),
        "anthropicEnabled": bool(srv.get("anthropicEnabled", True)),
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


def save(patch):
    data = validate(_merge(load(), patch))
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return data

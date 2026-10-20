# -*- coding: utf-8 -*-
"""Herta Space Station - desktop app entry point.

A single executable that embeds the control API and the Herta UI:
  1. boots the tiny control server (server.py) on 127.0.0.1:8899
  2. opens a native WebView2 window pointing at it
  3. start / stop buttons drive llama-server (and its VRAM) from the window
"""

import ctypes
import os
import socket
import sys
import threading
from http.server import ThreadingHTTPServer

import server
import webview

TITLE = "HERTA SPACE STATION"
WINDOW_W, WINDOW_H = 1462, 960


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.35)
        return sock.connect_ex((server.HOST, port)) != 0


def _existing_console() -> str | None:
    """Reuse an already running Herta console instead of double binding."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://{server.HOST}:{server.UI_PORT}/api/status", timeout=1.2) as r:
            if r.status == 200:
                return f"http://{server.HOST}:{server.UI_PORT}/"
    except Exception:
        return None
    return None


def _pick_port() -> int:
    for port in range(server.UI_PORT, server.UI_PORT - 20, -1):
        if _port_free(port):
            return port
    with socket.socket() as sock:
        sock.bind((server.HOST, 0))
        return sock.getsockname()[1]


def main() -> None:
    try:  # give the window its own taskbar identity / icon group
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Herta.Station.Terminal")
    except Exception:
        pass

    url = _existing_console()
    if url is None:
        port = _pick_port()
        httpd = ThreadingHTTPServer((server.HOST, port), server.Handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://{server.HOST}:{port}/"

    webview.create_window(
        TITLE,
        url,
        width=WINDOW_W,
        height=WINDOW_H,
        min_size=(1150, 760),
        background_color="#05070f",
        text_select=True,
    )
    webview.start()


if __name__ == "__main__":
    sys.exit(main())

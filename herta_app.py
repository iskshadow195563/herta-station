# -*- coding: utf-8 -*-
"""Herta Space Station - desktop app entry point.

One executable that embeds the control API and the Herta UI:
  1. boots the tiny control server (server.py) on 127.0.0.1:8899
  2. opens a native webview window (WebView2 / WKWebView / WebKitGTK)
  3. start / stop buttons drive llama-server (and its memory) from the window

Works on Windows, Linux and macOS. When no webview backend can be imported or
created, the UI is served normally and opened in the default browser instead —
the app stays usable on a bare server install.

    HertaStation                # normal desktop window
    HertaStation --browser      # force the browser, no native window
    HertaStation --serve        # API + UI only, never open anything
    HertaStation --port 8901    # control-plan port (default 8899)
    HertaStation --check        # print environment report and exit
"""

import argparse
import os
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import hostos
import server

TITLE = "HERTA SPACE STATION"
WINDOW_W, WINDOW_H = 1462, 960


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.35)
        return sock.connect_ex((server.HOST, port)) != 0


def _existing_console() -> str | None:
    """Reuse an already running Herta console instead of double binding."""
    if hostos.console_alive(server.UI_PORT):
        return f"http://{server.HOST}:{server.UI_PORT}/"
    return None


def _pick_port() -> int:
    for port in range(server.UI_PORT, server.UI_PORT - 20, -1):
        if _port_free(port):
            return port
    with socket.socket() as sock:
        sock.bind((server.HOST, 0))
        return sock.getsockname()[1]


def _serve(port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((server.HOST, port), server.Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _open_window(url: str) -> bool:
    """Native window. Returns False when no backend is available."""
    try:
        import webview
    except Exception as exc:                     # missing GTK/Qt on Linux is normal
        print(f"[herta] no webview backend ({exc}); falling back to the browser",
              file=sys.stderr)
        return False
    try:
        webview.create_window(
            TITLE, url,
            width=WINDOW_W, height=WINDOW_H,
            min_size=(1150, 760),
            background_color="#05070f",
            text_select=True,
        )
        webview.start()
        return True
    except Exception as exc:
        print(f"[herta] webview failed ({exc}); falling back to the browser", file=sys.stderr)
        return False


def _browser_until_stopped(url: str) -> None:
    """No native window: open a tab and keep serving until Ctrl+C."""
    hostos.open_url(url)
    print(f"[herta] serving the UI at {url} — Ctrl+C to quit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="HertaStation", add_help=True)
    parser.add_argument("--browser", action="store_true", help="open the UI in a browser")
    parser.add_argument("--serve", action="store_true", help="serve only, open nothing")
    parser.add_argument("--port", type=int, default=None, help="control-plane port")
    parser.add_argument("--check", action="store_true", help="print environment report")
    args = parser.parse_args(argv)

    if args.check:
        print(hostos.describe())
        return 0
    if args.port:
        server.UI_PORT = args.port

    if hostos.IS_WIN:
        try:  # give the window its own taskbar identity / icon group
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Herta.Station.Terminal")
        except Exception:
            pass

    url = _existing_console()
    if url is None:
        port = _pick_port()
        _serve(port)
        url = f"http://{server.HOST}:{port}/"

    # Anthropic-compatible /v1/messages shim (own port, same underlying model API)
    server.start_anthropic_proxy()

    if args.serve:
        print(f"[herta] serving the UI at {url} — Ctrl+C to quit")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0

    if args.browser or not _open_window(url):
        _browser_until_stopped(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())

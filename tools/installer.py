# -*- coding: utf-8 -*-
"""Herta Station installer (Windows).

This is the small PyInstaller one-file front end that ships as
``HertaStation-Setup-<version>.exe``.  It carries the whole application plus a
llama.cpp build as a single ``payload.zip`` resource, unpacks it into a stable
per-user folder, creates shortcuts, and starts the app.

    HertaStation-Setup.exe              install (or update) and run
    HertaStation-Setup.exe --silent     install only, no shortcuts prompt
    HertaStation-Setup.exe --uninstall  remove the installed copy

Nothing here talks to the network: the payload is inside this executable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

APP_NAME = "Herta Station"
FOLDER = "HertaStation"
VERSION = "{version}"


def resource(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def target_dir() -> str:
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(root, "Programs", FOLDER)


def say(text: str = "") -> None:
    print(text, flush=True)


def extract(payload: str, dest: str) -> None:
    """Unpack the payload, dropping the leading ``HertaStation/`` folder.

    The zip mirrors the PyInstaller onedir layout, so entries look like
    ``HertaStation/HertaStation.exe``; the install dir already plays that role.
    """
    os.makedirs(dest, exist_ok=True)
    prefix = FOLDER + "/"
    with zipfile.ZipFile(payload) as zf:
        names = zf.namelist()
        total = len(names)
        say(f"  unpacking {total} files ...")
        for index, name in enumerate(names, 1):
            rel = name[len(prefix):] if name.startswith(prefix) else name
            if rel:
                out = os.path.join(dest, *rel.split("/"))
                if name.endswith("/"):
                    os.makedirs(out, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
                    with zf.open(name) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            if index % 200 == 0 or index == total:
                print(f"\r  {index}/{total}", end="", flush=True)
    say()


def shortcut(target_exe: str, link_path: str, workdir: str) -> bool:
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
        "$s.TargetPath='{exe}';$s.WorkingDirectory='{wd}';"
        "$s.IconLocation='{exe},0';$s.Description='{desc}';$s.Save()"
    ).format(link=link_path.replace("'", "''"), exe=target_exe.replace("'", "''"),
             wd=workdir.replace("'", "''"), desc=APP_NAME)
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except Exception:
        return False


def make_shortcuts(exe: str, workdir: str) -> None:
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    start = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                         "Start Menu", "Programs")
    for folder, label in ((desktop, APP_NAME), (start, APP_NAME)):
        if folder and os.path.isdir(folder):
            link = os.path.join(folder, label + ".lnk")
            if shortcut(exe, link, workdir):
                say(f"  shortcut: {link}")


def uninstall(root: str) -> int:
    home = os.path.expanduser("~")
    start = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                         "Start Menu", "Programs")
    for link in (os.path.join(home, "Desktop", APP_NAME + ".lnk"),
                 os.path.join(start, APP_NAME + ".lnk")):
        try:
            os.remove(link)
        except OSError:
            pass
    say(f"removing {root} ...")
    shutil.rmtree(root, ignore_errors=True)
    say("uninstalled")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = target_dir()
    exe = os.path.join(root, "HertaStation.exe")

    say(f"{APP_NAME} installer {VERSION}")
    say(f"target: {root}")

    if "--uninstall" in argv:
        return uninstall(root)

    payload = resource("payload.zip")
    if not os.path.isfile(payload):
        say(f"payload missing: {payload}")
        return 1

    # Close a running copy so the payload can replace it.
    subprocess.run(["taskkill", "/F", "/IM", "HertaStation.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        extract(payload, root)
    except Exception as exc:
        say(f"extraction failed: {exc}")
        return 1

    if "--silent" not in argv:
        make_shortcuts(exe, root)

    if os.path.isfile(exe):
        say("starting ...")
        subprocess.Popen([exe], cwd=root, close_fds=True)
    else:
        say(f"installed, but {exe} is missing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Build the installable distributions.

    python tools/make_dist.py --platform win           # HertaStation-Setup-<v>.exe
    python tools/make_dist.py --platform linux         # herta-station_<v>_amd64.deb
    python tools/make_dist.py --platform macos-arm64   # HertaStation.app -> .zip
    python tools/make_dist.py --platform macos-x64
    python tools/make_dist.py --platform all

Every package carries its own llama.cpp build (from ``vendor/llama.cpp/<platform>``)
so the target machine needs nothing installed but a GPU driver — plus python3 on
Linux/macOS, because those bundles ship the app as a zipapp instead of a
PyInstaller binary (PyInstaller cannot cross-compile from Windows).

    --variant vulkan|cuda|metal   which bundled backend to use (default: the one
                                  folder present for that platform)
    --no-app                      reuse the previous PyInstaller onedir build
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "dist"
VENDOR = ROOT / "vendor" / "llama.cpp"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip() if (ROOT / "VERSION").exists() else "1.1.0"
APP_SOURCES = ["hostos.py", "models.py", "settings.py", "server.py", "anthropic_proxy.py",
               "herta_app.py", "ui.html"]
SLUGS = {"win": "windows-x86_64", "linux": "linux-x86_64",
         "macos-arm64": "macos-arm64", "macos-x64": "macos-x86_64"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    log(f"\n==> {msg}")


def run(cmd, **kw) -> None:
    log("    " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


# --------------------------------------------------------------------------- app

def stage_app() -> Path:
    """Copy the Python sources + UI into a folder that becomes a zipapp."""
    out = BUILD / "app"
    rm(out)
    out.mkdir(parents=True)
    for name in APP_SOURCES:
        shutil.copy2(ROOT / name, out / name)
    (out / "__main__.py").write_text(
        "import sys\nfrom herta_app import main\nsys.exit(main())\n", encoding="utf-8")
    return out


def build_pyz(stage: Path) -> Path:
    import zipapp
    target = BUILD / f"herta-station-{VERSION}.pyz"
    rm(target)
    zipapp.create_archive(stage, target=target, interpreter="/usr/bin/env python3",
                          compressed=True)
    log(f"    {target.name}: {target.stat().st_size/1e6:.1f} MB")
    return target


def llama_dir(platform: str, variant: str | None = None) -> Path:
    slug = SLUGS[platform]
    base = VENDOR / slug
    if not base.is_dir():
        raise SystemExit(f"missing bundled llama.cpp: {base}")
    if variant:
        cand = base / variant
        if not cand.is_dir():
            raise SystemExit(f"no variant {variant} in {base}")
        return cand
    subdirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    if subdirs:
        return subdirs[-1]
    return base


def rust_bin(platform: str) -> Path | None:
    """Prebuilt herta-rs for a platform, if cargo has produced one here."""
    name = "herta-rs.exe" if platform == "win" else "herta-rs"
    for cand in (ROOT / "tools" / "herta-rs" / "target" / "release" / name,
                 ROOT / "vendor" / "herta-rs" / SLUGS[platform] / name):
        if cand.is_file():
            return cand
    return None


def copy_tree(src: Path, dst: Path, exec_globs=("*",)) -> int:
    """Copy, dereferencing symlinks so the result is portable across archives."""
    count = 0
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.is_dir():
            count += copy_tree(entry, target, exec_globs)
        else:
            shutil.copy2(entry, target)
            count += 1
    return count


# ----------------------------------------------------------------------- windows

def build_windows(variant: str | None, rebuild: bool = True) -> Path:
    step("Windows: PyInstaller onedir")
    work = BUILD / "win"
    onedir = work / "HertaStation"
    if rebuild or not (onedir / "HertaStation.exe").exists():
        rm(onedir)
        run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
             "--name", "HertaStation", "--icon", ROOT / "herta.ico",
             "--add-data", f"{ROOT / 'ui.html'};.",
             "--paths", ROOT, "--collect-all", "webview", "--collect-all", "clr_loader",
             "--distpath", work, "--workpath", BUILD / "win-work",
             "--specpath", BUILD, ROOT / "herta_app.py"], cwd=str(ROOT))

    step("Windows: bundle llama.cpp")
    llama = llama_dir("win", variant)
    dest = onedir / "_internal" / "llama.cpp"
    rm(dest)
    count = copy_tree(llama, dest)
    log(f"    {count} files from {llama.name} -> _internal/llama.cpp")

    step("Windows: bundle herta-rs")
    helper = rust_bin("win")
    if helper:
        shutil.copy2(helper, onedir / helper.name)
        log(f"    {helper.name} {helper.stat().st_size/1e6:.2f} MB -> next to HertaStation.exe")
    else:
        log("    no herta-rs build for this OS; the Python downloader is used instead")

    step("Windows: payload.zip")
    payload = work / "payload.zip"
    rm(payload)
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(onedir.rglob("*")):
            if path.is_file():
                zf.write(path, str(Path("HertaStation") / path.relative_to(onedir)))
    log(f"    {payload.stat().st_size/1e6:.1f} MB")

    step("Windows: installer exe")
    inst_src = BUILD / "installer_build" / "installer.py"
    inst_src.parent.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "tools" / "installer.py").read_text(encoding="utf-8")
    inst_src.write_text(text.replace("{version}", VERSION), encoding="utf-8")
    out_name = f"HertaStation-Setup-{VERSION}"
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--console",
         "--name", out_name, "--icon", ROOT / "herta.ico",
         "--add-data", f"{payload};.",
         "--distpath", DIST, "--workpath", BUILD / "installer-work",
         "--specpath", BUILD, inst_src], cwd=str(ROOT))
    exe = DIST / (out_name + ".exe")
    log(f"    {exe.name}: {exe.stat().st_size/1e6:.1f} MB")
    return exe


# ------------------------------------------------------------------------- linux

def _ar_write(path: Path, members) -> None:
    """Write a plain Unix ``ar`` archive — that is all a .deb container is."""
    with open(path, "wb") as out:
        out.write(b"!<arch>\n")
        for name, data in members:
            header = "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n".format(
                name, int(time.time()), "0", "0", "100644", len(data))
            out.write(header.encode("ascii"))
            out.write(data)
            if len(data) % 2:
                out.write(b"\n")


def _tar_gz(files) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        dirs = set()
        for _, arc_path, _ in files:
            parent = str(Path(arc_path).parent).replace(os.sep, "/")
            while parent not in (".", "", "/"):
                dirs.add(parent)
                parent = str(Path(parent).parent).replace(os.sep, "/")
        for arc_path in sorted(dirs):
            info = tarfile.TarInfo(arc_path)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, int(time.time())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tf.addfile(info)
        for disk_path, arc_path, mode in files:
            with open(disk_path, "rb") as fh:
                data = fh.read()
            info = tarfile.TarInfo(arc_path)
            info.size, info.mode, info.mtime = len(data), mode, int(time.time())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tf.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), 6)


LINUX_LAUNCHER = """#!/bin/sh
# Herta Station launcher (installed by the .deb)
HERE=/opt/herta-station
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Herta Station needs python3 >= 3.10 (try: sudo apt install python3)" >&2
  exit 1
fi
export HERTA_LLAMA_DIR="$HERE/llama.cpp"
[ -x "$HERE/bin/herta-rs" ] && export HERTA_RS="$HERE/bin/herta-rs"
exec "$PY" "$HERE/herta-station.pyz" "$@"
"""

LINUX_DESKTOP = """[Desktop Entry]
Type=Application
Name=Herta Space Station
Name[zh_TW]=黑塔空間站
Name[zh_CN]=黑塔空间站
GenericName=Local model switchboard
Comment=Start and stop local llama.cpp models (OpenAI compatible)
Comment[zh_TW]=啟動／停止本機 llama.cpp 模型（OpenAI 相容）
Comment[zh_CN]=启动／停止本机 llama.cpp 模型（OpenAI 兼容）
Exec=/usr/bin/herta-station
Icon=herta-station
Terminal=false
Categories=Development;Utility;
Keywords=llm;llama;local;ai;
StartupWMClass=HertaStation
"""

LINUX_POSTINST = """#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t /usr/share/icons/hicolor >/dev/null 2>&1 || true
fi
exit 0
"""


def build_linux(variant: str | None, pyz: Path | None = None) -> Path:
    step("Linux: stage /opt/herta-station")
    stage = BUILD / "linux"
    rm(stage)
    (stage / "opt/herta-station").mkdir(parents=True)
    src_pyz = pyz or build_pyz(stage_app())
    shutil.copy2(src_pyz, stage / "opt/herta-station/herta-station.pyz")
    llama = llama_dir("linux", variant)
    count = copy_tree(llama, stage / "opt/herta-station/llama.cpp")
    log(f"    {count} files from {llama.name}")
    helper = rust_bin("linux")
    if helper:
        (stage / "opt/herta-station/bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(helper, stage / "opt/herta-station/bin/herta-rs")
        (stage / "opt/herta-station/bin/herta-rs").chmod(0o755)
        log("    herta-rs bundled (/opt/herta-station/bin/herta-rs)")

    launcher = stage / "usr/bin/herta-station"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(LINUX_LAUNCHER, encoding="utf-8", newline="\n")
    desktop = stage / "usr/share/applications/herta-station.desktop"
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(LINUX_DESKTOP, encoding="utf-8", newline="\n")
    icon_src = ROOT / "packaging" / "herta-station.svg"
    if icon_src.exists():
        icon = stage / "usr/share/icons/hicolor/scalable/apps/herta-station.svg"
        icon.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon_src, icon)
    postinst = stage / "postinst"
    postinst.write_text(LINUX_POSTINST, encoding="utf-8", newline="\n")
    postinst.chmod(0o755)

    step("Linux: control + data tarballs")
    arch = "amd64"
    installed_kb = int(sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1024)
    control = f"""Package: herta-station
Version: {VERSION}
Section: utils
Priority: optional
Architecture: {arch}
Depends: python3 (>= 3.10)
Recommends: mesa-vulkan-drivers | vulkan-loader
Installed-Size: {installed_kb}
Maintainer: GGbone <iskshadow195563@users.noreply.github.com>
Homepage: https://github.com/iskshadow195563/herta-station
Description: Herta Space Station - local llama.cpp switchboard
 A desktop control panel for local GGUF models: start and stop a bundled
 llama-server, pick models, auto-size the context and KV cache to the
 detected GPU, and expose an OpenAI-compatible API on port 11433 plus an
 Anthropic-compatible endpoint on 11435.
 .
 This package bundles its own llama.cpp build (Vulkan backend); the target
 machine only needs a GPU driver. Models (*.gguf) are not included.
"""

    control_tar = BUILD / "control.tar.gz"
    # control.tar.gz needs the control file from a real path
    ctl_file = BUILD / "control"
    ctl_file.write_text(control, encoding="utf-8", newline="\n")
    control_tar.write_bytes(_tar_gz([
        (ctl_file, "./control", 0o644),
        (postinst, "./postinst", 0o755),
    ]))

    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in ("postinst", "control"):
            arc = "./" + str(path.relative_to(stage)).replace(os.sep, "/")
            mode = 0o755 if (path.name in ("herta-station", "llama-server", "herta-rs")
                             or path.suffix in (".so",) or ".so." in path.name) else 0o644
            files.append((path, arc, mode))
    data_tar = BUILD / "data.tar.gz"
    data_tar.write_bytes(_tar_gz(files))
    log(f"    {len(files)} files, {installed_kb/1024:.1f} MB installed")

    step("Linux: .deb")
    deb = DIST / f"herta-station_{VERSION}_{arch}.deb"
    DIST.mkdir(exist_ok=True)
    _ar_write(deb, [
        ("debian-binary", b"2.0\n"),
        ("control.tar.gz", control_tar.read_bytes()),
        ("data.tar.gz", data_tar.read_bytes()),
    ])
    log(f"    {deb.name}: {deb.stat().st_size/1e6:.1f} MB")
    return deb


# ------------------------------------------------------------------------- macos

MAC_LAUNCHER = """#!/bin/bash
# Herta Station - app bundle launcher
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../Resources" && pwd)"
PY=""
for candidate in \\
  /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \\
  /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 \\
  /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  [ -x "$candidate" ] || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  osascript -e 'display alert "Herta Station" message "Python 3.10+ is required. Install it with: brew install python@3.12"' >/dev/null 2>&1 || true
  echo "python3 >= 3.10 not found (brew install python@3.12)" >&2
  exit 1
fi
export HERTA_LLAMA_DIR="$HERE/llama.cpp"
[ -x "$HERE/bin/herta-rs" ] && export HERTA_RS="$HERE/bin/herta-rs"
export HERTA_DATA_DIR="${HERTA_DATA_DIR:-$HOME/Library/Application Support/HertaStation}"
exec "$PY" "$HERE/herta-station.pyz" "$@"
"""

MAC_INFO = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Herta Station</string>
  <key>CFBundleDisplayName</key><string>Herta Space Station</string>
  <key>CFBundleIdentifier</key><string>com.herta.station</string>
  <key>CFBundleExecutable</key><string>HertaStation</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
"""


def build_macos(platform: str, variant: str | None, pyz: Path | None = None) -> Path:
    step(f"macOS ({platform}): app bundle")
    stage = BUILD / platform
    rm(stage)
    app = stage / "HertaStation.app"
    contents = app / "Contents"
    macos_dir = contents / "MacOS"
    res = contents / "Resources"
    macos_dir.mkdir(parents=True)
    res.mkdir(parents=True)

    launcher = macos_dir / "HertaStation"
    launcher.write_text(MAC_LAUNCHER, encoding="utf-8", newline="\n")
    launcher.chmod(0o755)
    (contents / "Info.plist").write_text(MAC_INFO.replace("{version}", VERSION),
                                         encoding="utf-8", newline="\n")
    src_pyz = pyz or build_pyz(stage_app())
    shutil.copy2(src_pyz, res / "herta-station.pyz")
    llama = llama_dir("macos-arm64" if platform == "macos-arm64" else "macos-x64", variant)
    count = copy_tree(llama, res / "llama.cpp")
    for lib in (res / "llama.cpp").iterdir():
        if lib.name == "llama-server" or lib.suffix == ".dylib" or ".dylib." in lib.name:
            lib.chmod(0o755)
    log(f"    {count} files from {llama.name}")
    helper = rust_bin(platform)
    if helper:
        (res / "bin").mkdir(exist_ok=True)
        shutil.copy2(helper, res / "bin/herta-rs")
        (res / "bin/herta-rs").chmod(0o755)
        log("    herta-rs bundled (Resources/bin/herta-rs)")

    step(f"macOS ({platform}): zip")
    DIST.mkdir(exist_ok=True)
    out = DIST / f"HertaStation-{VERSION}-{platform}.zip"
    rm(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(app.rglob("*")):
            arc = str(path.relative_to(stage)).replace(os.sep, "/")
            info = zipfile.ZipInfo(arc)
            info.create_system = 3
            # Windows cannot store POSIX exec bits, so decide them from the name:
            # Gatekeeper refuses to launch a non-executable launcher/llama-server.
            info.external_attr = (_macos_mode(path, arc) | (0o40000 if path.is_dir() else 0)) << 16
            info.date_time = time.localtime(path.stat().st_mtime)[:6]
            if path.is_dir():
                zf.writestr(info, b"")
            else:
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, path.read_bytes())
    log(f"    {out.name}: {out.stat().st_size/1e6:.1f} MB")
    return out


def _macos_mode(path: Path, arc: str) -> int:
    """POSIX mode for a zip entry coming from a Windows filesystem."""
    if path.is_dir():
        return 0o755
    if "/MacOS/" in arc or "/bin/" in arc:
        return 0o755
    if "/llama.cpp/" in arc:
        name = path.name
        if ".dylib" in name or "." not in name:
            return 0o755
        return 0o644
    return 0o644


# -------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build installable distributions")
    ap.add_argument("--platform", default="all",
                    choices=["all", "win", "linux", "macos-arm64", "macos-x64"])
    ap.add_argument("--variant", default=None,
                    help="bundled llama.cpp backend folder (vulkan/cuda/metal)")
    ap.add_argument("--no-app", action="store_true", help="reuse the existing onedir app")
    args = ap.parse_args(argv)

    DIST.mkdir(exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    log(f"Herta Station {VERSION} -> {DIST}")

    if args.platform in ("all", "win"):
        build_windows(args.variant, rebuild=not args.no_app)
    if args.platform in ("all", "linux"):
        build_linux(args.variant)
    if args.platform in ("all", "macos-arm64"):
        build_macos("macos-arm64", args.variant)
    if args.platform in ("all", "macos-x64"):
        build_macos("macos-x64", args.variant)

    log("\ndone:")
    for f in sorted(DIST.glob("*")):
        if f.is_file() and f.suffix in (".exe", ".deb", ".zip", ".dmg"):
            log(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

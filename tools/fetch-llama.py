#!/usr/bin/env python3
"""Fetch the llama.cpp release binaries that `make_dist.py` bundles.

The vendor tree is deliberately NOT committed (it is ~360 MB of binaries), so
CI — and any fresh clone — runs this once before packaging.

    python tools/fetch-llama.py --platform win linux macos-arm64 macos-x64
    python tools/fetch-llama.py --platform linux --tag b11934
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
DL = VENDOR / "dl"
LLAMA = VENDOR / "llama.cpp"
DEFAULT_TAG = "b11934"

# platform -> (asset name, destination slug, variant folder)
ASSETS = {
    "win": ("llama-{tag}-bin-win-vulkan-x64.zip", "windows-x86_64", "vulkan"),
    "linux": ("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", "linux-x86_64", "vulkan"),
    "macos-arm64": ("llama-{tag}-bin-macos-arm64.tar.gz", "macos-arm64", "metal"),
    "macos-x64": ("llama-{tag}-bin-macos-x64.tar.gz", "macos-x86_64", ""),
}

URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"  cached {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    log(f"  {dest.stat().st_size/1e6:.1f} MB -> {dest}")
    return dest


def extract(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                if not name:
                    continue
                with zf.open(info) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    else:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                if not name:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
                mode = member.mode & 0o777
                if mode:
                    os.chmod(dest / name, mode)
                count += 1
    for path in dest.iterdir():
        if path.name == "llama-server" or path.name == "llama-server.exe":
            os.chmod(path, 0o755)
    return count


def fetch(platform: str, tag: str, force: bool = False) -> Path:
    asset_tpl, slug, variant = ASSETS[platform]
    asset = asset_tpl.format(tag=tag)
    archive = DL / asset
    target = LLAMA / slug / (variant or "")
    if force and target.exists():
        shutil.rmtree(target)
    if target.is_dir() and any(target.iterdir()):
        log(f"{platform}: already present at {target}")
        return target
    log(f"{platform}: {asset}")
    download(URL.format(tag=tag, asset=asset), archive)
    count = extract(archive, target)
    log(f"{platform}: {count} files -> {target}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", nargs="+", default=list(ASSETS),
                    choices=list(ASSETS) + [["all"]], help="which platform binaries to fetch")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    platforms = list(ASSETS) if "all" in args.platform else args.platform
    for platform in platforms:
        fetch(platform, args.tag, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

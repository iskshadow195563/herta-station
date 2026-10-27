# -*- coding: utf-8 -*-
"""Portable host layer: the one place that knows which OS we run on.

Everything the console needs from the operating system lives here so that
``server.py`` / ``models.py`` stay platform agnostic:

  * process image path  (Win32 API / /proc / ps)
  * port -> PID         (netstat / ss / lsof / /proc/net)
  * PID -> name list    (tasklist / pgrep)
  * kill a process tree (taskkill /T /F, SIGTERM->SIGKILL on POSIX)
  * GPU + VRAM          (nvidia-smi, Windows CIM, AMD sysfs, Apple unified)
  * where settings and models may live (portable / XDG / macOS Library)

The parsers are pure functions so they can be unit-tested with captured command
output from every platform — see ``tests/test_hostos.py``.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

EXE_SUFFIX = ".exe" if IS_WIN else ""
SERVER_NAME = "llama-server"
APP_NAME = "HertaStation"

# Same numbers everywhere; Windows' signal module simply omits SIGKILL.
SIG_TERM = getattr(signal, "SIGTERM", 15)
SIG_KILL = getattr(signal, "SIGKILL", 9)

# macOS / Linux app bundles may be read-only; settings must not live there.
_BUNDLE_ROOT = getattr(sys, "_MEIPASS", None)


# --------------------------------------------------------------------------- #
# process plumbing
# --------------------------------------------------------------------------- #
def spawn_kwargs():
    """Extra Popen keyword arguments: no console window, own session on POSIX."""
    if IS_WIN:
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def cmdline(args):
    """Render an argv list the way the platform's shell would display it."""
    return subprocess.list2cmdline(args) if IS_WIN else shlex.join(args)


def run(args, timeout=15):
    """Run a console tool without ever blocking on a lingering grandchild.

    ``ollama ps`` starts the daemon when it is not running; the daemon inherits
    our stdout pipe, so ``subprocess.run(timeout=)`` kills the CLI and then waits
    forever for the pipe to close. Drive Popen directly and kill the tree.
    """
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **spawn_kwargs())
    except Exception as exc:  # missing binary, permission denied, …
        return f"{exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
        return out.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        kill_tree(proc.pid)
        try:
            out, _ = proc.communicate(timeout=3)
        except Exception:
            out = b""
        return out.decode("utf-8", "replace")
    except Exception as exc:
        return f"{exc}"


def kill_tree(pid, force=True):
    """Terminate ``pid`` and its children. Returns True when the call succeeded."""
    if not pid or pid <= 1:
        return False
    if IS_WIN:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           timeout=10)
            return True
        except Exception:
            return False

    ok = True
    # Own session (spawned by us) -> kill the whole group in one shot.
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, SIG_TERM)
            time.sleep(0.25)
            os.killpg(pid, SIG_KILL)
            return True
    except Exception:
        pass
    # Someone else's process: children first, then the process itself.
    for child in _child_pids(pid):
        kill_tree(child, force)
    try:
        os.kill(pid, SIG_TERM)
    except ProcessLookupError:
        return True
    except Exception:
        ok = False
    if force:
        for _ in range(20):                      # up to ~2 s to exit politely
            time.sleep(0.1)
            if not pid_alive(pid):
                return True
        try:
            os.kill(pid, SIG_KILL)
        except Exception:
            ok = False
    return ok


def _child_pids(pid):
    out = run(["pgrep", "-P", str(pid)], timeout=5)
    return [int(x) for x in out.split() if x.isdigit()]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def process_image(pid):
    """Absolute path of the executable behind ``pid``, or None."""
    if not pid:
        return None
    if IS_WIN:
        return _windows_image(pid)
    if IS_LINUX:
        try:
            path = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None
        return path[:-len(" (deleted)")] if path.endswith(" (deleted)") else path
    text = run(["ps", "-o", "command=", "-p", str(pid)], timeout=5).strip()
    if not text:
        return None
    first = text.splitlines()[0].split(" ", 1)[0]
    return first.lstrip("-") or None


def _windows_image(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                  wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return buf.value
    finally:
        kernel.CloseHandle(handle)


# --------------------------------------------------------------------------- #
# port -> PID
# --------------------------------------------------------------------------- #
def listener_pid(port, runner=None):
    """PID owning a LISTEN socket on ``port`` (loopback or wildcard), else None.

    ``runner`` lets the caller inject its own command runner (server._run), which
    keeps process spawning mockable in the console's unit tests.
    """
    run = runner or globals()["run"]
    if IS_WIN:
        return parse_netstat(run(["netstat", "-ano", "-p", "TCP"], timeout=10), port)
    if IS_LINUX:
        pid = parse_ss(run(["ss", "-ltnpH"], timeout=6), port)
        if pid:
            return pid
    pid = parse_lsof_pids(run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], timeout=6))
    if pid:
        return pid
    if IS_LINUX:
        return _proc_net_pid(port)
    return None


def parse_netstat(text, port):
    """Windows ``netstat -ano -p TCP`` -> PID."""
    for line in (text or "").splitlines():
        fields = line.split()
        if (len(fields) >= 5 and fields[0].upper() == "TCP"
                and fields[1].rsplit(":", 1)[-1] == str(port)
                and fields[-2].upper() == "LISTENING" and fields[-1].isdigit()):
            return int(fields[-1])
    return None


def parse_ss(text, port):
    """``ss -ltnpH`` -> PID. Matches 0.0.0.0:PORT, *:PORT, [::]:PORT."""
    for line in (text or "").splitlines():
        if ":" + str(port) not in line:
            continue
        fields = line.split()
        local = next((f for f in fields if re.search(r":%d$" % port, f)), None)
        if not local:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def parse_lsof_pids(text):
    """``lsof -t`` prints bare PIDs, one per line (may be comma separated)."""
    for token in re.split(r"[,\s]+", (text or "").strip()):
        if token.isdigit():
            return int(token)
    return None


def _proc_net_pid(port):
    """Fallback with no external tools: /proc/net/tcp{,6} inode -> /proc/*/fd."""
    target = "%04X" % int(port)
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, "r", encoding="utf-8", errors="replace") as fh:
                next(fh, None)
                for row in fh:
                    cols = row.split()
                    if len(cols) > 9 and cols[1].rsplit(":", 1)[-1] == target and cols[3] == "0A":
                        inodes.add(cols[9])
        except OSError:
            continue
    if not inodes:
        return None
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    link = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                found = re.fullmatch(r"socket:\[(\d+)\]", link)
                if found and found.group(1) in inodes:
                    return int(entry)
        except OSError:
            continue
    return None


def pids_by_name(name, runner=None):
    """Every live PID whose executable basename is ``name`` (no .exe needed)."""
    run = runner or globals()["run"]
    bare = name[:-4] if name.lower().endswith(".exe") else name
    if IS_WIN:
        return parse_tasklist(run(["tasklist", "/FI", f"IMAGENAME eq {bare}.exe",
                                  "/FO", "CSV", "/NH"], timeout=8), bare)
    pids = set()
    for args in (["pgrep", "-x", bare], ["pgrep", "-f", bare]):
        out = run(args, timeout=6)
        for token in out.split():
            if token.isdigit():
                pids.add(int(token))
    # pgrep -f can match our own command line; verify the image basename.
    keep = []
    for pid in sorted(pids):
        image = process_image(pid) or ""
        if os.path.basename(image).lower() in (bare.lower(), bare.lower() + ".exe"):
            keep.append(pid)
    return keep


def parse_tasklist(text, bare):
    """``tasklist /FO CSV /NH`` -> PIDs whose image matches ``bare``."""
    pids = []
    for line in (text or "").splitlines():
        parts = [p.strip('" ') for p in line.split(",")]
        if len(parts) >= 2 and parts[0].lower().rstrip(".exe").startswith(bare.lower()):
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


# --------------------------------------------------------------------------- #
# GPU / memory
# --------------------------------------------------------------------------- #
def gpu_info():
    """Best-effort GPU description: {gpu, vram_gb, driver, source, unified, ram_gb}."""
    info = {"gpu": "unknown", "vram_gb": 0.0, "driver": "", "source": "none",
            "unified": False, "ram_gb": round(total_ram_gb(), 2)}
    nv = _nvidia()
    if nv:
        info.update(nv)
        return info
    if IS_MAC:
        info.update(_apple())
        return info
    if IS_LINUX:
        amd = _amd_sysfs()
        if amd:
            info.update(amd)
            return info
    if IS_WIN:
        win = _windows_wmi()
        if win:
            info.update(win)
    return info


def _nvidia():
    """Works on Windows, Linux and (rare) eGPU Macs."""
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits"], timeout=8)
    line = (out or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3 or not parts[1].isdigit():
        return None
    return {"gpu": parts[0], "vram_gb": round(int(parts[1]) / 1024, 2),
            "driver": parts[2], "source": "nvidia-smi", "unified": False}


def total_ram_gb():
    try:
        if IS_MAC:
            out = run(["sysctl", "-n", "hw.memsize"], timeout=5).strip()
            return int(out) / (1024 ** 3) if out.isdigit() else 0.0
        if IS_LINUX:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for row in fh:
                    if row.startswith("MemTotal:"):
                        return int(row.split()[1]) / (1024 ** 2)
        if IS_WIN:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _apple():
    """Apple Silicon shares RAM with the GPU: llama.cpp sees one memory pool.

    Metal refuses to allocate beyond the recommended working set (~75 % of RAM),
    so ``vram_gb`` reports that usable ceiling instead of a physical VRAM number.
    """
    ram = total_ram_gb()
    chip = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5).strip()
    name = run(["system_profiler", "SPDisplaysDataType"], timeout=12) if not chip else ""
    match = re.search(r"Chipset Model:\s*(.+)", name or "")
    gpu = match.group(1).strip() if match else (chip or "Apple GPU")
    gpu = re.sub(r"^Apple\s+", "Apple ", gpu)
    usable = round(ram * 0.75, 2) if ram else 0.0
    return {"gpu": gpu, "vram_gb": usable, "driver": "Metal", "source": "unified-memory",
            "unified": True, "ram_gb": round(ram, 2)}


def _amd_sysfs(root="/sys/class/drm"):
    """AMD/Intel discrete GPU on Linux via /sys/class/drm (``root`` is injectable)."""
    import glob
    for card in sorted(glob.glob(os.path.join(root, "card*/device"))):
        total = os.path.join(card, "mem_info_vram_total")
        if not os.path.isfile(total):
            continue
        try:
            with open(total, "r", encoding="utf-8") as fh:
                size = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if size <= 0:
            continue
        name = ""
        try:
            with open(os.path.join(card, "product_name"), "r",
                      encoding="utf-8", errors="replace") as fh:
                name = fh.readline().strip()
        except OSError:
            name = ""
        if not name:
            try:
                with open(os.path.join(card, "uevent"), "r",
                          encoding="utf-8", errors="replace") as fh:
                    found = re.search(r"^PRODUCT=(.+)$", fh.read(), re.MULTILINE)
                    name = found.group(1).strip() if found else ""
            except OSError:
                name = ""
        return {"gpu": name or "AMD GPU", "vram_gb": round(size / (1024 ** 3), 2),
                "driver": "amdgpu", "source": "sysfs", "unified": False}
    return None


def _windows_wmi():
    out = run(["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending | "
               "Select-Object -First 1) | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }"],
              timeout=20).strip()
    if "|" not in out:
        return None
    name, ram = out.rsplit("|", 1)
    try:
        size = int(ram.strip() or 0)
    except ValueError:
        size = 0
    return {"gpu": name.strip(), "vram_gb": round(size / (1024 ** 3), 2),
            "driver": "", "source": "Win32_VideoController", "unified": False}


def vram_usage(runner=None):
    """Used/total VRAM in MiB for the dashboard, or None where it is meaningless."""
    run = runner or globals()["run"]
    if IS_MAC:
        # Unified memory: report the process-side figure llama.cpp can use.
        info = gpu_info()
        total = int((info.get("vram_gb") or 0) * 1024)
        used = _mac_gpu_used(total)
        return {"used": used, "total": total, "gpu": info.get("gpu") or "Apple GPU"}
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name",
               "--format=csv,noheader,nounits"], timeout=8).strip().splitlines()
    if out:
        parts = [p.strip() for p in out[0].split(",")]
        try:
            return {"used": int(parts[0]), "total": int(parts[1]), "gpu": parts[2]}
        except (ValueError, IndexError):
            pass
    if IS_LINUX:
        amd = _amd_sysfs()
        if amd:
            used_path = "/sys/class/drm/card0/device/mem_info_vram_used"
            used = 0
            try:
                with open(used_path, "r", encoding="utf-8") as fh:
                    used = int(fh.read().strip()) // (1024 ** 2)
            except (OSError, ValueError):
                pass
            return {"used": used, "total": int(amd["vram_gb"] * 1024), "gpu": amd["gpu"]}
    return {"used": None, "total": None, "gpu": "n/a"}


def _mac_gpu_used(total_mib):
    """No per-GPU counter on macOS; derive llama.cpp's RSS from ps."""
    pids = pids_by_name(SERVER_NAME)
    if not pids:
        return 0
    rss = run(["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)], timeout=6)
    kb = sum(int(x) for x in rss.split() if x.isdigit())
    return min(total_mib, int(kb / 1024)) if kb else 0


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def config_home():
    if IS_WIN:
        return os.environ.get("APPDATA") or os.path.expanduser("~")
    if IS_MAC:
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")


def data_home():
    if IS_WIN:
        return os.environ.get("LOCALAPPDATA") or config_home()
    if IS_MAC:
        return config_home()
    return os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")


def app_dir():
    """Writable per-user directory for settings, logs, PID and downloaded models."""
    env = os.environ.get("HERTA_DATA_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    frozen = getattr(sys, "frozen", False)
    if frozen and (IS_MAC or IS_LINUX):
        # A bundle can be read-only (AppImage, /Applications, quarantined .app).
        return os.path.join(data_home(), APP_NAME)
    if frozen:
        return os.path.dirname(os.path.abspath(sys.executable))
    here = os.path.dirname(os.path.abspath(__file__))
    # Running from a zipapp / pyz: __file__ points inside the archive, which is
    # not a directory we can write to.
    if os.path.isdir(here) and os.access(here, os.W_OK):
        return here
    return os.path.join(data_home(), APP_NAME)


def default_models_dir():
    env = os.environ.get("HERTA_MODELS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    preferred = os.path.join(os.path.expanduser("~"), "llama.cpp", "models")
    beside = os.path.join(app_dir(), "models")
    if not IS_WIN and os.path.isdir(preferred):
        return preferred
    # An installed copy lives in %LOCALAPPDATA%\Programs\HertaStation: keep its
    # models in the data dir so uninstalling the app does not delete them.
    local = os.environ.get("LOCALAPPDATA")
    if IS_WIN and local:
        installed = os.path.join(local, "Programs", APP_NAME)
        if os.path.normcase(os.path.abspath(app_dir())) == os.path.normcase(os.path.abspath(installed)):
            return os.path.join(data_home(), APP_NAME, "models")
    return beside


def bundled_llama_dirs():
    """Folders that may carry a llama.cpp build shipped with the app.

    Order matters: the copy packed next to (or inside) this app wins over
    anything the user installed machine-wide.
    """
    dirs = []
    env = os.environ.get("HERTA_LLAMA_DIR")
    if env:
        dirs.append(os.path.abspath(os.path.expanduser(env)))
    slug = f"{kind_name()}-{arch_name()}"
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(meipass)                                    # onefile and onedir
        roots.append(os.path.dirname(os.path.abspath(meipass)))   # onedir: beside _internal
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    roots += [exe_dir, app_dir()]
    if IS_MAC:
        roots.append(os.path.join(exe_dir, "..", "Resources"))    # .app bundle
    here = os.path.dirname(os.path.abspath(__file__))
    roots += [
        os.path.join(here, "vendor", "llama.cpp", slug),          # dev tree, best variant
        os.path.join(here, "vendor", "llama.cpp"),
        os.path.join(here, "llama.cpp"),
        "/opt/herta-station/llama.cpp",
        "/usr/local/lib/herta-station/llama.cpp",
    ]
    for root in roots:
        if not root:
            continue
        for sub in ("", "llama.cpp", os.path.join("llama.cpp", slug), slug):
            dirs.append(os.path.normpath(os.path.join(root, sub)))
        # vendor/<slug>/<variant>/ : the same platform may ship several backends
        for base in (os.path.join(root, slug), os.path.join(root, "llama.cpp", slug)):
            try:
                for entry in sorted(os.scandir(base), key=lambda e: e.name, reverse=True):
                    if entry.is_dir():
                        dirs.append(os.path.normpath(entry.path))
            except OSError:
                pass
    return dirs


def _bundled_llama_exe():
    name = SERVER_NAME + EXE_SUFFIX
    for folder in bundled_llama_dirs():
        path = os.path.join(folder, name)
        if os.path.isfile(path) and (IS_WIN or os.access(path, os.X_OK)):
            return path
    return None


def default_llama_exe():
    """Locate llama-server without hardcoding a Windows path."""
    env = os.environ.get("HERTA_LLAMA_EXE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    bundled = _bundled_llama_exe()
    if bundled:
        return bundled
    names = [SERVER_NAME + EXE_SUFFIX]
    found = shutil.which(names[0])
    if found:
        return found
    home = os.path.expanduser("~")
    candidates = [
        f"/opt/llama.cpp/build/bin/{names[0]}",
        f"/usr/local/bin/{names[0]}",
        f"/opt/homebrew/bin/{names[0]}",
        os.path.join(home, "llama.cpp", "build", "bin", names[0]),
        os.path.join(home, ".local", "bin", names[0]),
        os.path.join(home, "llama.cpp", names[0]),
        os.path.join(app_dir(), names[0]),
        os.path.join(os.path.dirname(os.path.abspath(sys.executable)), names[0]),
    ]
    if IS_WIN:
        candidates += [r"D:\llama.cpp\b11756\llama-server.exe"]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return names[0] if not IS_WIN else candidates[0]


def default_model_path():
    env = os.environ.get("HERTA_MODEL")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return ""


def default_mmproj_path():
    env = os.environ.get("HERTA_MMPROJ_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return ""


def open_url(url):
    """Open ``url`` in the user's browser (GUI fallback when no WebView exists)."""
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:
        return False


def console_alive(port, timeout=1.2):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def kind_name():
    """Platform slug used in paths and in the status payload."""
    if IS_WIN:
        return "windows"
    if IS_MAC:
        return "macos"
    if IS_LINUX:
        return "linux"
    return sys.platform


def arch_name():
    """CPU architecture as ll users expect it ("x86_64", "arm64", "aarch64")."""
    machine = (os.uname().machine if hasattr(os, "uname") else os.environ.get("PROCESSOR_ARCHITECTURE", "")) or ""
    lowered = machine.lower()
    if lowered in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if lowered in ("arm64", "aarch64"):
        return "arm64"
    return machine


def describe():
    """One-line platform banner for logs and the self-test."""
    info = gpu_info()
    return json.dumps({"platform": kind_name(), "arch": arch_name(),
                       "gpu": info.get("gpu"), "vram_gb": info.get("vram_gb"),
                       "ram_gb": info.get("ram_gb"), "unified": bool(info.get("unified")),
                       "data_dir": app_dir(), "models_dir": default_models_dir(),
                       "llama_exe": default_llama_exe()}, ensure_ascii=False)

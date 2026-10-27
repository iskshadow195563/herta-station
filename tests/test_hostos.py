# -*- coding: utf-8 -*-
"""Host-layer tests: every platform branch, driven by captured command output.

The console only ever runs on one machine, so the other two platforms are
exercised by feeding the real parsers the exact text each tool prints there and
by flipping ``hostos.IS_WIN`` / ``IS_MAC`` / ``IS_LINUX``.

    python tests/test_hostos.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hostos  # noqa: E402


NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:8899           0.0.0.0:0              LISTENING       4412
  TCP    127.0.0.1:11433        0.0.0.0:0              LISTENING       7788
  TCP    192.168.31.10:52110    20.205.243.166:443     ESTABLISHED     9001
  TCP    [::]:11433             [::]:0                 LISTENING       7788
"""

SS_LINUX = """
LISTEN 0      4096         0.0.0.0:8899       0.0.0.0:*    users:(("python3",pid=4412,fd=5))
LISTEN 0      4096         0.0.0.0:11433      0.0.0.0:*    users:(("llama-server",pid=7788,fd=3))
LISTEN 0      4096               *:22             *:*      users:(("sshd",pid=913,fd=3))
LISTEN 0      4096            [::]:11435          [::]:*   users:(("python3",pid=7788,fd=9))
"""

TASKLIST = """
"llama-server.exe","7788","Console","1","1,234,567 K"
"llama-server.exe","9911","Console","1","987,654 K"
INFO: No tasks are running which match the specified criteria.
"""


class PortLookupTests(unittest.TestCase):
    def test_netstat_wildcard_binding(self):
        self.assertEqual(hostos.parse_netstat(NETSTAT, 11433), 7788)

    def test_netstat_ipv4_loopback(self):
        self.assertEqual(hostos.parse_netstat(NETSTAT, 8899), 4412)

    def test_netstat_ignores_established_and_other_ports(self):
        self.assertIsNone(hostos.parse_netstat(NETSTAT, 52110))
        self.assertIsNone(hostos.parse_netstat(NETSTAT, 4000))

    def test_ss_reports_owner_pid(self):
        self.assertEqual(hostos.parse_ss(SS_LINUX, 11433), 7788)
        self.assertEqual(hostos.parse_ss(SS_LINUX, 8899), 4412)

    def test_ss_handles_wildcard_and_ipv6(self):
        self.assertEqual(hostos.parse_ss(SS_LINUX, 22), 913)
        self.assertEqual(hostos.parse_ss(SS_LINUX, 11435), 7788)

    def test_ss_without_users_column(self):
        text = "LISTEN 0 4096 0.0.0.0:11433 0.0.0.0:*\n"
        self.assertIsNone(hostos.parse_ss(text, 11433))

    def test_lsof_bare_pid(self):
        self.assertEqual(hostos.parse_lsof_pids("7788\n"), 7788)
        self.assertEqual(hostos.parse_lsof_pids("4412,7788\n"), 4412)
        self.assertIsNone(hostos.parse_lsof_pids(""))

    def test_listener_pid_uses_the_injected_runner(self):
        calls = []

        def runner(args, timeout=15):
            calls.append(args[0])
            return NETSTAT

        with patch.object(hostos, "IS_WIN", True), patch.object(hostos, "IS_LINUX", False):
            self.assertEqual(hostos.listener_pid(11433, runner=runner), 7788)
        self.assertEqual(calls, ["netstat"])

    def test_listener_pid_prefers_ss_on_linux(self):
        seen = []

        def runner(args, timeout=15):
            seen.append(args[0])
            return SS_LINUX if args[0] == "ss" else ""

        with patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_LINUX", True):
            self.assertEqual(hostos.listener_pid(11433, runner=runner), 7788)
        self.assertEqual(seen[0], "ss")

    def test_listener_pid_falls_back_to_lsof_without_ss(self):
        seen = []

        def runner(args, timeout=15):
            seen.append(args[0])
            return "7788\n" if args[0] == "lsof" else ""

        with patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_LINUX", False):
            self.assertEqual(hostos.listener_pid(11433, runner=runner), 7788)
        self.assertEqual(seen, ["lsof"])


class ProcessTests(unittest.TestCase):
    def test_tasklist_rows(self):
        self.assertEqual(hostos.parse_tasklist(TASKLIST, "llama-server"), [7788, 9911])

    def test_tasklist_no_match_banner(self):
        self.assertEqual(hostos.parse_tasklist("INFO: No tasks are running\n", "llama-server"), [])

    def test_pids_by_name_uses_pgrep_and_verifies_image(self):
        def runner(args, timeout=15):
            if args[0] == "pgrep":
                return "7788\n9911\n"
            return ""

        def image(pid):
            return "/opt/llama.cpp/build/bin/llama-server" if pid == 7788 else "/usr/bin/vim"

        with patch.object(hostos, "IS_WIN", False), \
                patch.object(hostos, "process_image", side_effect=image):
            self.assertEqual(hostos.pids_by_name("llama-server", runner=runner), [7788])

    def test_process_image_ps_on_macos(self):
        def runner(args, timeout=15):
            return "/Applications/HertaStation.app/Contents/MacOS/llama-server --port 11433\n"

        with patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_LINUX", False), \
                patch.object(hostos, "run", side_effect=runner):
            self.assertEqual(hostos.process_image(4242),
                             "/Applications/HertaStation.app/Contents/MacOS/llama-server")

    def test_kill_tree_posix_uses_process_group(self):
        with patch.object(hostos, "IS_WIN", False), \
                patch.object(os, "getpgid", return_value=7788, create=True), \
                patch.object(os, "killpg", create=True) as killpg, \
                patch.object(hostos.time, "sleep"):
            self.assertTrue(hostos.kill_tree(7788))
        self.assertEqual(killpg.call_count, 2)      # SIGTERM then SIGKILL

    def test_kill_tree_never_touches_pid_one(self):
        with patch.object(hostos.subprocess, "run") as run_call, \
                patch.object(os, "killpg", create=True) as killpg:
            self.assertFalse(hostos.kill_tree(1))
        run_call.assert_not_called()
        killpg.assert_not_called()

    def test_spawn_kwargs_are_platform_appropriate(self):
        with patch.object(hostos, "IS_WIN", True):
            self.assertIn("creationflags", hostos.spawn_kwargs())
        with patch.object(hostos, "IS_WIN", False):
            self.assertEqual(hostos.spawn_kwargs(), {"start_new_session": True})

    def test_cmdline_quoting(self):
        with patch.object(hostos, "IS_WIN", True):
            self.assertIn('"', hostos.cmdline(["a b", "c"]))
        with patch.object(hostos, "IS_WIN", False):
            self.assertEqual(hostos.cmdline(["a b", "c"]), "'a b' c")


class GpuTests(unittest.TestCase):
    def test_nvidia_is_used_when_present(self):
        def runner(args, timeout=15):
            return "NVIDIA GeForce RTX 5080, 16303, 617.42\n"

        with patch.object(hostos, "run", side_effect=runner):
            info = hostos.gpu_info()
        self.assertEqual(info["source"], "nvidia-smi")
        self.assertEqual(info["vram_gb"], 15.92)
        self.assertFalse(info["unified"])

    def test_apple_silicon_reports_unified_memory(self):
        def runner(args, timeout=15):
            if args[:2] == ["sysctl", "-n"] and args[2] == "hw.memsize":
                return str(36 * 1024 ** 3) + "\n"
            if args[:2] == ["sysctl", "-n"] and args[2] == "machdep.cpu.brand_string":
                return "Apple M3 Pro\n"
            if args[:1] == ["system_profiler"]:
                return "Chipset Model: Apple M3 Pro\n"
            return ""

        with patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_MAC", True), \
                patch.object(hostos, "IS_LINUX", False), patch.object(hostos, "run", side_effect=runner):
            info = hostos.gpu_info()
        self.assertTrue(info["unified"])
        self.assertEqual(info["source"], "unified-memory")
        self.assertEqual(info["ram_gb"], 36.0)
        self.assertAlmostEqual(info["vram_gb"], 27.0, places=2)   # 75 % working set
        self.assertIn("Apple", info["gpu"])

    def test_amd_sysfs_parsing(self):
        with tempfile.TemporaryDirectory(prefix="herta-sys-") as root:
            device = Path(root) / "card0" / "device"
            device.mkdir(parents=True)
            (device / "mem_info_vram_total").write_text(str(16 * 1024 ** 3), encoding="utf-8")
            (device / "product_name").write_text("Navi 31 [Radeon RX 7900 XTX]\n", encoding="utf-8")
            info = hostos._amd_sysfs(root)
        self.assertEqual(info["vram_gb"], 16.0)
        self.assertEqual(info["source"], "sysfs")
        self.assertIn("Radeon", info["gpu"])

    def test_vram_usage_parses_nvidia(self):
        def runner(args, timeout=15):
            return "15452, 16303, NVIDIA GeForce RTX 5080\n"

        with patch.object(hostos, "IS_MAC", False), patch.object(hostos, "run", side_effect=runner):
            usage = hostos.vram_usage()
        self.assertEqual(usage["used"], 15452)
        self.assertEqual(usage["total"], 16303)

    def test_vram_usage_is_honest_when_nothing_is_detectable(self):
        with patch.object(hostos, "IS_MAC", False), patch.object(hostos, "IS_LINUX", False), \
                patch.object(hostos, "run", return_value=""):
            self.assertEqual(hostos.vram_usage(), {"used": None, "total": None, "gpu": "n/a"})


class PathTests(unittest.TestCase):
    def test_linux_data_dir_follows_xdg(self):
        with patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_MAC", False), \
                patch.object(hostos, "IS_LINUX", True), \
                patch.object(hostos.sys, "frozen", True, create=True), \
                patch.dict(os.environ, {"XDG_DATA_HOME": "/tmp/xdg", "HERTA_DATA_DIR": ""}):
            # separators follow the host running the test; the layout is what matters
            self.assertEqual(hostos.app_dir(), os.path.join("/tmp/xdg", "HertaStation"))

    def test_macos_data_dir_uses_application_support(self):
        with tempfile.TemporaryDirectory(prefix="herta-home-") as home, \
                patch.object(hostos, "IS_WIN", False), patch.object(hostos, "IS_MAC", True), \
                patch.object(hostos, "IS_LINUX", False), \
                patch.object(hostos.sys, "frozen", True, create=True), \
                patch.dict(os.environ, {"HERTA_DATA_DIR": "", "HOME": home,
                                        "USERPROFILE": home}):
            self.assertEqual(hostos.app_dir(),
                             os.path.join(home, "Library", "Application Support", "HertaStation"))

    def test_env_override_wins_everywhere(self):
        with patch.dict(os.environ, {"HERTA_DATA_DIR": "/srv/herta"}):
            self.assertEqual(hostos.app_dir(), os.path.abspath("/srv/herta"))

    def test_source_run_keeps_files_beside_the_code(self):
        with patch.object(hostos.sys, "frozen", False, create=True), \
                patch.dict(os.environ, {"HERTA_DATA_DIR": ""}):
            self.assertEqual(hostos.app_dir(),
                             os.path.dirname(os.path.abspath(hostos.__file__)))

    def test_models_dir_env_override(self):
        with patch.dict(os.environ, {"HERTA_MODELS_DIR": "~/weights"}):
            self.assertEqual(hostos.default_models_dir(),
                             os.path.abspath(os.path.expanduser("~/weights")))

    def test_models_dir_prefers_the_usual_llama_cpp_place_on_posix(self):
        with tempfile.TemporaryDirectory(prefix="herta-home-") as home:
            (Path(home) / "llama.cpp" / "models").mkdir(parents=True)
            with patch.object(hostos, "IS_WIN", False), \
                    patch.dict(os.environ, {"HOME": home, "USERPROFILE": home,
                                            "HERTA_MODELS_DIR": ""}):
                self.assertEqual(hostos.default_models_dir(),
                                 os.path.join(home, "llama.cpp", "models"))

    def test_llama_exe_env_override(self):
        with patch.dict(os.environ, {"HERTA_LLAMA_EXE": "/usr/local/bin/llama-server"}):
            self.assertEqual(hostos.default_llama_exe(), os.path.abspath("/usr/local/bin/llama-server"))


class VramLimitTests(unittest.TestCase):
    """The 15.3 GB cliff is a 16 GB Windows card property, not a global cap."""

    def test_sixteen_gb_windows_card_stays_under_the_cliff(self):
        import models
        with patch.object(hostos, "IS_WIN", True):
            self.assertAlmostEqual(models.vram_limit(15.92), models.VRAM_CLIFF_GB, places=2)

    def test_bigger_windows_cards_keep_the_full_budget(self):
        import models
        with patch.object(hostos, "IS_WIN", True):
            self.assertAlmostEqual(models.vram_limit(24.0), 24.0 * 0.965, places=2)

    def test_linux_and_macos_never_clamp(self):
        import models
        with patch.object(hostos, "IS_WIN", False):
            self.assertAlmostEqual(models.vram_limit(15.92), 15.92 * 0.965, places=2)

    def test_unified_memory_uses_the_same_rule(self):
        import models
        with patch.object(hostos, "IS_WIN", False):
            # Apple Silicon reports 75 % of RAM as the GPU budget already.
            self.assertAlmostEqual(models.vram_limit(27.0), 27.0 * 0.965, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

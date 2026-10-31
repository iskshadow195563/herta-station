"""Isolated regressions: no GPU, real models, user data or external network."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings
import models
import server
import anthropic_proxy


@contextlib.contextmanager
def http_server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="herta-settings-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "settings.json"
        self.override = patch.object(settings, "SETTINGS_FILE", str(self.path))
        self.override.start()
        self.addCleanup(self.override.stop)

    def test_malformed_stored_shapes_fall_back(self):
        for data in ([], "wrong", {"ui": [1], "server": "wrong"}):
            with self.subTest(data=data):
                self.path.write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(settings.load(), settings.validate(settings.DEFAULTS))

    def test_rejects_non_object_patch(self):
        for data in ([1], "wrong", {"ui": []}, {"server": False}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                settings.save(data)
        self.assertFalse(self.path.exists())

    def test_failed_atomic_replace_keeps_previous_file(self):
        settings.save({"ui": {"lang": "en"}})
        previous = self.path.read_bytes()
        with patch.object(settings.os, "replace", side_effect=PermissionError("read only")):
            with self.assertRaises(OSError):
                settings.save({"ui": {"lang": "zh-Hans"}})
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(Path(self.temp.name).glob("*.tmp")), [])

    def test_numeric_and_boolean_inputs(self):
        data = settings.validate({"ui": {"scale": float("nan"), "animations": "false"},
                                  "server": {"ctx": float("inf"), "anthropicEnabled": "false"}})
        self.assertEqual(data["ui"]["scale"], 1.0)
        self.assertFalse(data["ui"]["animations"])
        self.assertEqual(data["server"]["ctx"], settings.DEFAULTS["server"]["ctx"])
        self.assertFalse(data["server"]["anthropicEnabled"])


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows process identity API")
    def test_windows_checks_the_actual_executable(self):
        # Compare against the image Windows reports, not sys.executable: inside a
        # venv the reported path is the venv shim, while the real image is the base
        # interpreter. The product only ever matches llama-server.exe, which is a
        # plain binary, so this keeps the test portable.
        image = server._process_image(os.getpid())
        self.assertTrue(image)
        self.assertTrue(server._matches_exe(os.getpid(), image))
        self.assertFalse(server._matches_exe(os.getpid(), image + ".not-me.exe"))

    def test_launch_passes_but_does_not_log_the_api_key(self):
        with tempfile.TemporaryDirectory(prefix="herta-launch-") as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"GGUFfixture")
            conf = settings.validate(settings.DEFAULTS)
            conf["server"]["apiKey"] = "synthetic-test-key"
            child = Mock(pid=43210)
            child.poll.return_value = None
            with contextlib.ExitStack() as stack:
                for key in ("CTX", "CTK", "CTV", "PARALLEL", "BIND", "PORT", "API_KEY", "NGL",
                            "_proc", "_started_at", "_last_error", "_last_error_code"):
                    stack.enter_context(patch.object(server, key, getattr(server, key)))
                stack.enter_context(patch.object(server, "CONF", conf))
                stack.enter_context(patch.object(server, "DATA_DIR", directory))
                stack.enter_context(patch.object(server, "LOG_PATH", str(Path(directory) / "llama-server.log")))
                stack.enter_context(patch.object(server, "PID_PATH", str(Path(directory) / "llama-server.pid")))
                stack.enter_context(patch.object(server, "EXE", sys.executable))
                stack.enter_context(patch.object(server, "current_model", return_value=str(model)))
                stack.enter_context(patch.object(server, "_alive", return_value=False))
                stack.enter_context(patch.object(server, "_port_pid", return_value=None))
                stack.enter_context(patch.object(server, "start_anthropic_proxy"))
                launch = stack.enter_context(patch.object(server.subprocess, "Popen", return_value=child))
                self.assertTrue(server.start_model()["ok"])
                self.assertIn("synthetic-test-key", launch.call_args.args[0])
                log = Path(server.LOG_PATH).read_text(encoding="utf-8")
                self.assertNotIn("synthetic-test-key", log)
                self.assertIn("***", log)

    def test_port_lookup_uses_current_port_and_local_endpoint(self):
        output = (" TCP 127.0.0.1:4000 127.0.0.1:5001 LISTENING 11\n"
                  " TCP [::]:5001 [::]:0 LISTENING 22\n")
        # The fixture is netstat output, so pin the Windows branch: without this
        # the lookup follows the host platform's path (lsof on macOS) and the
        # test fails on every developer Mac and in the macOS CI jobs.
        with patch.object(server, "PORT", 5001), patch.object(server, "_run", return_value=output), \
                patch.object(server.hostos, "IS_WIN", True), \
                patch.object(server.hostos, "IS_LINUX", False):
            self.assertEqual(server._port_pid(), 22)

    def test_stale_pid_file_does_not_override_live_listener(self):
        with tempfile.TemporaryDirectory(prefix="herta-pid-") as directory:
            path = Path(directory) / "llama-server.pid"
            path.write_text("99999", encoding="utf-8")
            with patch.object(server, "PID_PATH", str(path)), patch.object(server, "_proc", None), \
                    patch.object(server, "_port_pid", return_value=None):
                self.assertIsNone(server._pid())

    def test_ui_only_settings_do_not_require_model_restart(self):
        conf = settings.validate(settings.DEFAULTS)
        conf["ui"]["lang"] = "en"
        with patch.object(server, "CONF", settings.validate(settings.DEFAULTS)), \
                patch.object(server.appsettings, "save", return_value=conf), \
                patch.object(server, "_health", return_value=True), \
                patch.object(server, "_alive", return_value=True), \
                patch.object(server, "start_anthropic_proxy"):
            result = server.apply_settings({"ui": {"lang": "en"}})
            self.assertFalse(result["restart_required"])

    def test_saving_new_port_preserves_live_runtime_until_reload(self):
        conf = settings.validate(settings.DEFAULTS)
        conf["server"]["port"] = 12345
        with patch.object(server, "CONF", settings.validate(settings.DEFAULTS)), \
                patch.object(server, "PORT", 11433), \
                patch.object(server.appsettings, "save", return_value=conf), \
                patch.object(server, "_health", return_value=True), \
                patch.object(server, "_alive", return_value=True), \
                patch.object(server, "start_anthropic_proxy"):
            result = server.apply_settings({"server": {"port": 12345}})
            self.assertTrue(result["restart_required"])
            self.assertEqual(server.PORT, 11433)

    def test_log_tail_reads_a_bounded_suffix(self):
        with tempfile.TemporaryDirectory(prefix="herta-log-") as directory:
            path = Path(directory) / "test.log"
            path.write_bytes(b"old line\n" * 100000 + "last line 中文\n".encode())
            with patch.object(server, "LOG_PATH", str(path)):
                self.assertEqual(server._tail(1), ["last line 中文"])

    def test_stop_does_not_kill_an_unowned_listener(self):
        with tempfile.TemporaryDirectory(prefix="herta-owner-") as directory:
            with patch.object(server, "PID_PATH", str(Path(directory) / "missing.pid")), \
                    patch.object(server, "_proc", None), patch.object(server, "_started_at", None), \
                    patch.object(server, "_port_pid", return_value=42), \
                    patch.object(server.subprocess, "run") as run, \
                    patch.object(server, "_apply_server_settings"), \
                    patch.object(server, "start_anthropic_proxy"):
                self.assertTrue(server.stop_model()["ok"])
                run.assert_not_called()

    def test_launch_oom_is_visible_in_status(self):
        child = Mock(returncode=1)
        child.poll.return_value = 1
        with patch.object(server, "_proc", child), patch.object(server, "_started_at", 1), \
                patch.object(server, "_last_error", ""), patch.object(server, "_last_error_code", ""), \
                patch.object(server, "_pid", return_value=None), \
                patch.object(server, "_tail", return_value=["CUDA error: out of memory"]), \
                patch.object(server, "_vram_cached", return_value={"used": 0, "total": 0, "gpu": "test"}), \
                patch.object(server, "_lan_ip_cached", return_value="127.0.0.1"), \
                patch.object(server.models, "read_gguf_ctx", return_value=None):
            result = server.status()
        self.assertEqual(result["phase"], "idle")
        self.assertEqual(result["error_code"], "out_of_memory")


class DownloadTests(unittest.TestCase):
    @staticmethod
    def _jobs(directory, size=0):
        return [{
            "repo": "test/model",
            "path": "weights.gguf",
            "target": os.path.join(directory, "test__model", "weights.gguf"),
            "size": size,
        }]

    def test_completed_job_persists_selection_before_becoming_idle(self):
        with tempfile.TemporaryDirectory(prefix="herta-complete-") as directory:
            weights = b"GGUFfixture"
            response = io.BytesIO(weights)
            response.headers = {"Content-Length": str(len(weights))}
            observed = []
            def select(path):
                observed.append(models._dl["active"])
                self.assertEqual(Path(path).read_bytes(), weights)
            with patch.object(models, "MODELS_DIR", directory), \
                    patch.object(models, "_dl", dict(models._dl, active=True, started=1, done=0)), \
                    patch.object(models, "set_active", side_effect=select), \
                    patch.object(models.urllib.request, "urlopen", return_value=response):
                models._dl_cancel.clear()
                models._download_worker(self._jobs(directory))
                result = models.download_state()
                self.assertTrue(result["finished"])
                self.assertFalse(result["active"])
                self.assertEqual(result["error"], "")
                self.assertEqual(result["percent"], 100.0)
            self.assertEqual(observed, [True])

    def test_short_download_is_not_published_or_activated(self):
        with tempfile.TemporaryDirectory(prefix="herta-download-") as directory:
            response = io.BytesIO(b"GGUFshort")
            response.headers = {"Content-Length": "100"}
            response.status = 200
            with patch.object(models, "MODELS_DIR", directory), \
                    patch.object(models, "_dl", dict(models._dl, active=True, started=1, done=0)), \
                    patch.object(models, "set_active") as activate, \
                    patch.object(models.urllib.request, "urlopen", return_value=response):
                models._dl_cancel.clear()
                models._download_worker(self._jobs(directory))
                self.assertTrue(models.download_state()["error"])
                self.assertFalse((Path(directory) / "test__model" / "weights.gguf").exists())
                activate.assert_not_called()

    def test_cancel_during_last_read_does_not_publish(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "0"}
            def read(self, n=-1):
                models._dl_cancel.set()
                return b""
        with tempfile.TemporaryDirectory(prefix="herta-download-") as directory:
            with patch.object(models, "MODELS_DIR", directory), \
                    patch.object(models, "_dl", dict(models._dl, active=True, started=1, done=0)), \
                    patch.object(models, "set_active") as activate, \
                    patch.object(models.urllib.request, "urlopen", return_value=Response()):
                models._dl_cancel.clear()
                models._download_worker(self._jobs(directory))
                self.assertTrue(models.download_state()["error"])
                self.assertFalse((Path(directory) / "test__model" / "weights.gguf").exists())
                activate.assert_not_called()
        models._dl_cancel.clear()


class ProxyTests(unittest.TestCase):
    def test_bearer_key_reaches_upstream(self):
        seen = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                seen.append(self.headers.get("Authorization"))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if seen[-1] != "Bearer regression-test-key":
                    self.send_error(401)
                    return
                data = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        with http_server(Upstream) as upstream:
            proxy = anthropic_proxy.AnthropicProxy("127.0.0.1", upstream.server_port, "test", "regression-test-key")
            result = proxy.complete({"model": "test", "messages": []})
        self.assertEqual(result["content"][0]["text"], "ok")
        self.assertEqual(seen, ["Bearer regression-test-key"])

    def test_invalid_json_shape_returns_400(self):
        proxy = anthropic_proxy.AnthropicProxy("127.0.0.1", 1, "test")
        with http_server(anthropic_proxy._make_handler(proxy)) as httpd:
            request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/messages", data=b"[]",
                                             headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=3)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(json.loads(error.exception.read())["error"]["type"], "invalid_request_error")

    def test_sse_upstream_error_is_one_json_response(self):
        proxy = anthropic_proxy.AnthropicProxy("127.0.0.1", 1, "test")
        error = urllib.error.HTTPError("http://127.0.0.1", 401, "Unauthorized", {}, io.BytesIO(b"wrong key"))
        with patch.object(proxy, "_call", side_effect=error), \
                http_server(anthropic_proxy._make_handler(proxy)) as httpd:
            request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/messages",
                                             data=b'{"stream":true,"messages":[]}',
                                             headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=3)
            self.assertEqual(caught.exception.code, 502)
            body = json.loads(caught.exception.read())
            self.assertIn("upstream 401", body["error"]["message"])

    def test_count_tokens_includes_message_and_system_text(self):
        proxy = anthropic_proxy.AnthropicProxy("127.0.0.1", 1, "test")
        with http_server(anthropic_proxy._make_handler(proxy)) as httpd:
            data = {"system": "s" * 1000, "messages": [{"role": "user", "content": "hello" * 1000}]}
            request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/messages/count_tokens",
                                             data=json.dumps(data).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3) as response:
                result = json.loads(response.read())
        self.assertGreater(result["input_tokens"], 1000)
        self.assertTrue(result["estimated"])

    def test_stream_preserves_unicode_text_and_tool_arguments(self):
        chunks = [
            {"choices": [{"delta": {"content": "中文"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "toolu_test",
                          "function": {"name": "search", "arguments": '{"q":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                          "function": {"arguments": '"中文"}'}}]}, "finish_reason": "tool_calls"}]},
        ]
        raw = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
        response = io.BytesIO((raw + "data: [DONE]\n\n").encode())
        proxy = anthropic_proxy.AnthropicProxy("127.0.0.1", 1, "test")
        with patch.object(proxy, "_call", return_value=response):
            events = list(proxy.stream({"model": "test"}))
        data = [json.loads(event.split("data: ", 1)[1]) for event in events]
        self.assertEqual(data[0]["type"], "message_start")
        self.assertEqual(data[-1]["type"], "message_stop")
        self.assertIn("中文", "".join(event.get("delta", {}).get("text", "") for event in data))
        arguments = "".join(event.get("delta", {}).get("partial_json", "") for event in data)
        self.assertEqual(json.loads(arguments), {"q": "中文"})


class ControlContractTests(unittest.TestCase):
    def test_control_routes_and_failed_save_contract(self):
        with tempfile.TemporaryDirectory(prefix="herta-control-") as directory:
            conf = settings.validate(settings.DEFAULTS)
            with patch.object(settings, "SETTINGS_FILE", str(Path(directory) / "settings.json")), \
                    patch.object(server, "CONF", conf), \
                    patch.object(server, "_pid", return_value=None), \
                    patch.object(server, "_alive", return_value=False), \
                    patch.object(server, "_health", return_value=False), \
                    patch.object(server, "_tail", return_value=[]), \
                    patch.object(server, "_vram_cached", return_value={"used": 0, "total": 0, "gpu": "test"}), \
                    patch.object(server, "_lan_ip_cached", return_value="127.0.0.1"), \
                    patch.object(models, "MODELS_DIR", directory), \
                    patch.object(models, "gpu_total_gb", return_value=16), \
                    patch.object(models, "scan_local", return_value=[]), \
                    patch.object(models, "read_gguf_ctx", return_value=None), \
                    http_server(server.Handler) as httpd:
                base = f"http://127.0.0.1:{httpd.server_port}"
                for route, keys in (
                        ("status", ("phase", "pid", "vram", "log", "error", "endpoint")),
                        ("models", ("dir", "gpu_total_gb", "ctx", "items")),
                        ("settings", ("ok", "settings", "defaults")),
                        ("download/status", ("active", "done", "total", "percent", "error"))):
                    with urllib.request.urlopen(base + "/api/" + route, timeout=3) as response:
                        data = json.loads(response.read())
                    self.assertTrue(set(keys).issubset(data), route)
                with patch.object(settings, "atomic_json", side_effect=PermissionError("test denial")):
                    request = urllib.request.Request(base + "/api/settings", data=b'{"ui":{"lang":"en"}}')
                    with urllib.request.urlopen(request, timeout=3) as response:
                        data = json.loads(response.read())
                    self.assertFalse(data["ok"])
                    self.assertEqual(data["error_code"], "settings_write_failed")
                    self.assertIs(server.CONF, conf)


if __name__ == "__main__":
    unittest.main()

class HardeningTests(unittest.TestCase):
    """Regressions found while reviewing the previous drop."""

    def test_missing_pid_file_adopts_the_live_listener(self):
        missing = str(Path(tempfile.gettempdir()) / "herta-no-such.pid")
        with patch.object(server, "_proc", None), \
             patch.object(server, "PID_PATH", missing), \
             patch.object(server, "_port_pid", return_value=4242), \
             patch.object(server, "_matches_exe", return_value=True):
            self.assertEqual(server._pid(), 4242)

    def test_adoption_refuses_a_foreign_listener(self):
        missing = str(Path(tempfile.gettempdir()) / "herta-no-such.pid")
        with patch.object(server, "_proc", None), \
             patch.object(server, "PID_PATH", missing), \
             patch.object(server, "_port_pid", return_value=4242), \
             patch.object(server, "_matches_exe", return_value=False):
            self.assertIsNone(server._pid())

    def test_run_times_out_even_when_the_child_spawns_children(self):
        import time
        started = time.time()
        out = server._run([sys.executable, "-c", "import subprocess,sys,time; "
                           "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                           "time.sleep(30)"], timeout=1)
        self.assertLess(time.time() - started, 15)
        self.assertIsInstance(out, str)

    def test_context_plan_scales_kv_to_the_native_limit(self):
        with tempfile.TemporaryDirectory(prefix="herta-plan-") as directory:
            model = Path(directory) / "fixture.gguf"
            model.write_bytes(b"GGUF" + b"\0" * 4096)
            meta = {"ctx_train": 131072, "layers": 32, "heads": 16,
                    "kv_heads": 4, "head_dim": 256, "hybrid": True}
            with patch.object(models, "read_gguf_meta", return_value=meta), \
                 patch.object(models.os.path, "getsize", return_value=4 * 1024 ** 3):
                plan = models.context_plan(str(model), total_gb=16.0)
                self.assertEqual(plan["ctx"], 131072)      # native max is reachable
                self.assertIn(plan["ctk"], models.KV_BYTES)
                self.assertEqual(plan["ctk"], plan["ctv"])
                # A tighter budget must trade KV quality, then context length
                small = models.context_plan(str(model), total_gb=7.5, ctx_train=262144)
                self.assertLess(small["ctx"], 262144)
                self.assertEqual(small["ctk"], "q4_0")

    def test_auto_plan_can_be_switched_off(self):
        with tempfile.TemporaryDirectory(prefix="herta-plan-") as directory:
            with patch.object(settings, "SETTINGS_FILE", str(Path(directory) / "s.json")):
                self.assertTrue(settings.load()["server"]["autoPlan"])
                saved = settings.save({"server": {"autoPlan": False}})
                self.assertFalse(saved["server"]["autoPlan"])


class PresetTests(unittest.TestCase):
    def test_every_preset_points_at_a_real_hf_file(self):
        ids = [p["id"] for p in models.PRESETS]
        self.assertEqual(len(ids), len(set(ids)))       # ids are unique
        for preset in models.PRESETS:
            self.assertRegex(preset["repo"], r"^[\w.-]+/[\w.-]+$")
            self.assertTrue(preset["files"])
            for item in preset["files"]:
                self.assertTrue(item["path"].endswith(".gguf"), item["path"])
                self.assertGreater(item["size"], 100 * 1024 ** 2)
        # the engine preset is explicitly multi-shard, the qwen ones are single file
        self.assertEqual(len(models.find_preset("strata-flash-next-q2")["files"]), 2)
        self.assertEqual(len(models.find_preset("gemma3-4b-qat")["files"]), 2)   # model + projector
        self.assertEqual(len(models.find_preset("qwen35-9b-q4")["files"]), 1)

    def test_shard_files_collapse_to_one_library_entry(self):
        with tempfile.TemporaryDirectory(prefix="herta-shard-") as directory:
            folder = Path(directory) / "someone__big-model"
            folder.mkdir()
            for index in (1, 2):
                target = folder / f"model-0000{index}-of-00002.gguf"
                target.write_bytes(b"GGUF" + b"\0" * 32)
            with patch.object(models, "MODELS_DIR", directory), \
                 patch.object(models, "read_gguf_meta", return_value={}):
                rows = models.scan_local(0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["shards"], 2)
            self.assertTrue(rows[0]["name"].endswith("-00001-of-00002.gguf"))

    def test_preset_fit_prefers_shorter_context_before_failing(self):
        self.assertEqual(models.preset_fit(66.42, 65536, total=15.92)["level"], "over")
        mid = models.preset_fit(13.15, 65536, total=15.92)
        self.assertEqual(mid["level"], "tight")          # weights fit, KV must shrink
        self.assertLessEqual(mid["weights_only"], models.vram_limit(15.92))
        self.assertEqual(models.preset_fit(5.97, 65536, total=15.92)["level"], "ok")

    def test_preset_state_reports_install_and_disk_room(self):
        fake = [{"id": "tiny", "name": "Tiny", "repo": "someone/tiny", "entry": "tiny.gguf",
                 "quant": "Q4_K_M", "params_label": "1B", "params_b": 1,
                 "files": [{"path": "tiny.gguf", "size": 8}]}]
        with tempfile.TemporaryDirectory(prefix="herta-preset-") as directory:
            with patch.object(models, "MODELS_DIR", directory), \
                 patch.object(models, "PRESETS", fake), \
                 patch.object(models, "_free_gb", return_value=500.0), \
                 patch.object(models, "active_model_path", return_value=""):
                row = models.preset_state(65536, total=15.92)[0]
                self.assertFalse(row["installed"])
                self.assertTrue(row["disk_ok"])          # 500 GB free
                self.assertEqual(row["files"], 1)
                target = models.preset_files(models.find_preset("tiny"))[0]["path"]
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_bytes(b"GGUF\x0a\x00\x0a\x00")
                row = models.preset_state(65536, total=15.92)[0]
                self.assertTrue(row["installed"])
                self.assertFalse(row["partial"])

    def test_preset_state_flags_a_tight_disk(self):
        with tempfile.TemporaryDirectory(prefix="herta-preset-") as directory:
            with patch.object(models, "MODELS_DIR", directory), \
                 patch.object(models, "_free_gb", return_value=2.0), \
                 patch.object(models, "active_model_path", return_value=""):
                rows = {row["id"]: row for row in models.preset_state(65536, total=15.92)}
                self.assertFalse(rows["strata-flash-next-q2"]["disk_ok"])   # 66 GB > 2 GB
                self.assertTrue(rows["qwen3-1_7b-q4"]["disk_ok"])            # 1.1 GB fits

    def test_start_preset_download_rejects_unknown_id(self):
        res = models.start_preset_download("nope-nothing")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_code"], "invalid_download")


if __name__ == "__main__":
    unittest.main()
"""Image-engine regressions: no GPU, no ComfyUI process, no external network.

Everything here is pure logic plus monkeypatched probes, so the suite stays fast
and never allocates VRAM - the live end-to-end path lives in tools/image_e2e.py
and is run by hand on a real machine.

Discovery is pinned to a temporary checkout in every test, otherwise ``find_root``
would happily pick up the developer's real D:\\AI\\ComfyUI.
"""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import imagestation as I
import models
import server
import settings


@contextlib.contextmanager
def http_server(handler):
    """Ephemeral loopback server, same shape as the one in test_regressions."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def fake_checkout(temp_dir, weights=()):
    """A directory that passes ``is_checkout`` plus a venv python inside it."""
    root = Path(temp_dir) / "ComfyUI"
    (root / "comfy").mkdir(parents=True, exist_ok=True)
    for folder in ("diffusion_models", "text_encoders", "vae", "checkpoints", "loras"):
        (root / "models" / folder).mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# fake\n", encoding="utf-8")
    venv = root / "venv" / "Scripts"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "python.exe").write_bytes(b"")
    (venv / "python").write_bytes(b"")
    for folder, name, megabytes in weights:
        target = root / "models" / folder / name
        with open(target, "wb") as fh:
            fh.write(b"\0" * (int(megabytes * 1024 * 1024)))
    return str(root)


class ImageTest(unittest.TestCase):
    """Common scaffolding: isolated candidate list and clean module state."""

    root = ""

    def setUp(self):
        I._roots.update({"root": "", "python": ""})
        I._proc = None
        I._started_at = None
        I._state.update({"active": False, "done": 0, "total": 0, "finished": False,
                         "error": "", "error_code": "", "repo": "", "file": "",
                         "role": "", "target": "", "speed": 0.0})
        self._pin([])

    def _pin(self, roots):
        patcher = patch.object(I, "_candidates", lambda: list(roots))
        patcher.start()
        self.addCleanup(patcher.stop)

    def conf(self, root=None, **extra):
        group = {"comfyDir": root if root is not None else self.root}
        group.update(extra)
        return {"image": group}


class ImageConfTests(unittest.TestCase):
    def test_defaults_fill_in_and_unknown_keys_are_dropped(self):
        conf = I.image_conf({"image": {"port": 9999, "nonsense": True}})
        self.assertEqual(conf["port"], 9999)
        self.assertNotIn("nonsense", conf)
        self.assertEqual(conf["vramMode"], settings.DEFAULTS["image"]["vramMode"])

    def test_accepts_a_bare_group(self):
        self.assertEqual(I.image_conf({"port": 9000})["port"], 9000)

    def test_bad_input_falls_back_to_defaults(self):
        for value in (None, "garbage", 42, []):
            self.assertEqual(I.image_conf(value)["port"], settings.DEFAULTS["image"]["port"])

    def test_group_is_a_real_settings_key(self):
        self.assertIn("image", settings.DEFAULTS)
        for key in ("port", "vramMode", "autoOpen", "browser", "listen",
                    "comfyDir", "pythonExe", "extraArgs", "loadTimeout", "genTimeout"):
            self.assertIn(key, settings.DEFAULTS["image"], key)


class DiscoveryTests(ImageTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-img-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name)
        self._pin([self.root])

    def test_is_checkout_needs_main_and_comfy(self):
        self.assertTrue(I.is_checkout(self.root))
        self.assertFalse(I.is_checkout(str(Path(self.temp.name) / "nope")))
        bare = Path(self.temp.name) / "bare"
        bare.mkdir()
        (bare / "main.py").write_text("", encoding="utf-8")
        self.assertFalse(I.is_checkout(str(bare)))

    def test_find_root_prefers_the_configured_path(self):
        self.assertEqual(I.find_root(self.root), self.root)
        self.assertEqual(I.find_root(""), self.root)      # falls back to candidates

    def test_find_python_locates_the_venv(self):
        found = I.find_python(self.root).replace("\\", "/")
        self.assertTrue(found.endswith(("venv/Scripts/python.exe", "venv/Scripts/python",
                                        "venv/bin/python", "venv/bin/python3")), found)
        self.assertTrue(I.find_python(self.root, prefer="").replace("\\", "/").endswith(
            "python.exe" if sys.platform.startswith("win") else "python"))

    def test_inspect_reports_what_is_missing(self):
        info = I.inspect(self.conf())
        self.assertTrue(info["has_root"] and info["has_python"])
        self.assertEqual(info["models_dir"], str(Path(self.root) / "models"))
        self.assertFalse(info["custom_nodes_gguf"])

        empty = Path(self.temp.name) / "empty"
        empty.mkdir()
        self._pin([])
        missing = I.inspect(self.conf(str(empty)))
        self.assertFalse(missing["has_root"])
        self.assertFalse(missing["has_python"])
        self.assertEqual(missing["root"], "")


class FamilyTests(unittest.TestCase):
    def test_family_of_recognises_real_weight_names(self):
        cases = {
            "z-image-turbo-Q4_K_M.gguf": "z-image",
            "Qwen3-4B-Q8_0.gguf": "other",           # a text encoder, not a family
            "qwen-image-2.1-turbo-Q6_K.gguf": "qwen-image",
            "illustriousXL_v11.safetensors": "sdxl-anime",
            "noobaiXLNAIXL_vPred10.safetensors": "sdxl-anime",
            "sd_xl_base_1.0.safetensors": "sdxl",
            "flux1-schnell.safetensors": "flux",
            "chroma1-hd.safetensors": "chroma",
            "v1-5-pruned-emaonly.safetensors": "sd15",
            "random-thing.safetensors": "other",
        }
        for name, expected in cases.items():
            self.assertEqual(I.family_of(name), expected, name)

    def test_every_family_declares_prompt_languages(self):
        for key, _pattern, params, min_vram, langs, note in I.FAMILIES:
            meta = I.family_meta(key)
            self.assertEqual(meta["key"], key)
            self.assertTrue(langs, key)
            self.assertEqual(meta["lang_labels"], [I.LANG_LABEL[x] for x in langs])
            self.assertGreater(min_vram, 0, key)
            self.assertGreaterEqual(meta["min_vram_gb"], min_vram)
            self.assertTrue(meta["note"])
            self.assertTrue(params)

    def test_unknown_family_falls_back_to_english(self):
        meta = I.family_meta("nothing-here")
        self.assertEqual(meta["key"], "other")
        self.assertEqual(meta["langs"], ["en"])
        self.assertEqual(len(meta["lang_labels"]), 1)

    def test_language_labels_are_readable_strings(self):
        for code, label in I.LANG_LABEL.items():
            self.assertIsInstance(label, str)
            self.assertTrue(label.strip())
            self.assertGreater(len(label), 1)

    def test_language_sets_for_the_interesting_families(self):
        self.assertEqual(I.family_meta("qwen-image")["langs"], ["en", "zh"])
        self.assertEqual(I.family_meta("z-image")["langs"], ["en"])
        self.assertIn("tags", I.family_meta("sdxl-anime")["langs"])
        self.assertEqual(I.family_meta("flux")["langs"], ["en"])


class ScanTests(ImageTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-scan-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3),
            ("diffusion_models", "qwen-image-2.1-turbo-Q6_K.gguf", 5),
            ("vae", "ae.safetensors", 2),
            ("vae", "tiny.safetensors", 0.02),
        ])
        self._pin([self.root])

    def test_scan_reports_roles_families_and_sizes(self):
        out = I.scan_models(self.conf())
        self.assertTrue(out["ok"])
        self.assertEqual(out["root"], self.root)
        by_name = {i["name"]: i for i in out["items"]}
        self.assertNotIn("tiny.safetensors", by_name)          # below the 1 MB floor
        self.assertEqual(by_name["ae.safetensors"]["role"], "vae")
        self.assertEqual(by_name["ae.safetensors"]["folder"], "vae")
        self.assertEqual(by_name["z-image-turbo-Q4_K_M.gguf"]["family"], "z-image")
        self.assertEqual(by_name["z-image-turbo-Q4_K_M.gguf"]["lang_labels"],
                         [I.LANG_LABEL["en"]])
        self.assertEqual(by_name["qwen-image-2.1-turbo-Q6_K.gguf"]["langs"], ["en", "zh"])
        self.assertAlmostEqual(out["total_gb"],
                               round(sum(i["size_gb"] for i in out["items"]), 2), places=2)

    def test_items_carry_the_fields_the_ui_paints(self):
        out = I.scan_models(self.conf())
        for item in out["items"]:
            for key in ("name", "role", "folder", "path", "size_gb", "family",
                        "langs", "lang_labels", "params", "min_vram_gb"):
                self.assertIn(key, item, key)
            self.assertTrue(item["lang_labels"])

    def test_scan_without_engine_is_a_clean_failure(self):
        self._pin([])
        out = I.scan_models(self.conf(str(Path(self.temp.name) / "nope")))
        self.assertFalse(out["ok"])
        self.assertEqual(out["items"], [])
        self.assertEqual(out["total_gb"], 0)


class RecommendTests(ImageTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-reco-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name)
        self._pin([self.root])

    def _recommend(self, total_gb):
        with patch.object(models, "gpu_total_gb", return_value=total_gb), \
             patch.object(models, "hardware",
                          return_value={"gpu": "TEST GPU", "unified": False}):
            return I.recommend(self.conf())

    def test_picks_are_complete(self):
        for total in (8.0, 12.0, 16.0, 24.0):
            out = self._recommend(total)
            self.assertEqual(out["gpu"], "TEST GPU")
            self.assertEqual(out["gpu_total_gb"], total)
            self.assertEqual([p["tier"] for p in out["picks"]],
                             ["首選", "速度優先", "動漫特化", "畫質上限"])
            for pick in out["picks"]:
                self.assertTrue(pick["lang_labels"], pick["family"])
                self.assertTrue(pick["search"])
                self.assertIsInstance(pick["installed"], list)
                self.assertIn(pick["family"], [f[0] for f in I.FAMILIES])
                self.assertTrue(pick["reason"])

    def test_small_cards_get_small_models(self):
        small = self._recommend(8.0)
        self.assertNotIn("chroma", {p["family"] for p in small["picks"]})
        self.assertNotIn("flux", {p["family"] for p in small["picks"]})
        big = self._recommend(24.0)
        self.assertIn("qwen-image", {p["family"] for p in big["picks"]})

    def test_picks_declare_whether_they_fit(self):
        for total in (4.0, 6.0, 8.0, 16.0, 24.0):
            out = self._recommend(total)
            for pick in out["picks"]:
                expected = pick["min_vram_gb"] <= max(0.0, total - 1.5)
                self.assertEqual(pick["fits"], expected, f"{pick['family']} on {total} GB")
            self.assertEqual(bool(out["warning"]), not out["picks"][0]["fits"], total)
            if total >= 8.0:
                self.assertTrue(out["picks"][0]["fits"], total)

    def test_tiny_card_warns_instead_of_lying(self):
        out = self._recommend(4.0)
        self.assertTrue(out["warning"])
        self.assertIn("GB", out["warning"])
        self.assertFalse(out["picks"][0]["fits"])
        self.assertIn("顯存不足", out["picks"][0]["reason"])

    def test_installed_weights_are_listed(self):
        fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3)])
        out = self._recommend(16.0)
        self.assertEqual(out["installed_families"], ["z-image"])
        self.assertEqual([i["name"] for i in out["local"]], ["z-image-turbo-Q4_K_M.gguf"])
        self.assertEqual(out["local_gb"], round(sum(i["size_gb"] for i in out["local"]), 2))
        self.assertTrue(out["known"])
        self.assertTrue(out["hint"])

    def test_empty_library_is_reported_as_empty(self):
        out = self._recommend(16.0)
        self.assertEqual(out["local"], [])
        self.assertEqual(out["local_gb"], 0)
        self.assertEqual(out["installed_families"], [])

    def test_english_switches_the_prose_not_the_pick_order(self):
        zh = self._recommend(16.0)
        with patch.object(models, "gpu_total_gb", return_value=16.0), \
             patch.object(models, "hardware",
                          return_value={"gpu": "TEST GPU", "unified": False}):
            en = I.recommend(self.conf(), "en")
        self.assertEqual([p["tier"] for p in en["picks"]],
                         [p["tier"] for p in zh["picks"]])
        self.assertEqual(en["lang"], "en")
        for pick in en["picks"]:
            self.assertTrue(pick["note"].isascii(), pick["note"])
            self.assertTrue(pick["reason"].isascii(), pick["reason"])
        for pick in zh["picks"]:
            self.assertFalse(pick["note"].isascii(), pick["note"])
        self.assertTrue(all(k["note"].isascii() for k in en["known"]))
        self.assertTrue(en["hint"].isascii())

    def test_vram_flags_avoid_the_removed_normalvram_switch(self):
        flags = {mode: I.vram_flags(mode) for mode in ("high", "normal", "low", "verylow")}
        self.assertEqual(flags["high"], ["--highvram"])
        self.assertEqual(flags["normal"], [])
        self.assertEqual(flags["low"], ["--lowvram"])
        self.assertEqual(flags["verylow"], ["--novram"])
        for value in flags.values():
            self.assertNotIn("--normalvram", value)


class HuggingFaceTests(unittest.TestCase):
    ROWS = [
        {"id": "Qwen/Qwen-Image", "downloads": 9000, "likes": 42,
         "pipeline_tag": "text-to-image", "lastModified": "2025-09-01T00:00:00.000Z",
         "tags": ["gguf", "text-to-image"], "gated": False,
         "cardData": {"language": ["en", "zh", "zh-Hans"]}},
        {"id": "someone/Illustrious-XL-v1.1", "downloads": 100, "likes": 1,
         "tags": ["text-to-image"], "gated": True},
    ]

    def test_search_parses_rows_and_labels_languages(self):
        with patch.object(I, "_hf_json", return_value=self.ROWS):
            items = I.hf_search("qwen", limit=500)
        self.assertEqual(len(items), 2)
        first, second = items
        self.assertEqual(first["repo"], "Qwen/Qwen-Image")
        self.assertEqual(first["family"], "qwen-image")
        self.assertEqual(first["langs"], ["en", "zh"])
        self.assertEqual(first["lang_labels"], [I.LANG_LABEL["en"], I.LANG_LABEL["zh"]])
        self.assertEqual(first["build"], "gguf")
        self.assertEqual(first["updated"], "2025-09-01")
        self.assertEqual(second["family"], "sdxl-anime")
        self.assertTrue(second["gated"])
        self.assertIn("tags", second["langs"])

    def test_search_clamps_limit_and_forwards_the_query(self):
        seen = {}

        def fake(url, timeout=25):
            seen["url"] = url
            return []

        with patch.object(I, "_hf_json", fake):
            I.hf_search("flux", limit=999)
        self.assertIn("filter=text-to-image", seen["url"])
        self.assertIn("search=flux", seen["url"])
        self.assertIn("limit=60", seen["url"])

    def test_search_survives_a_failing_endpoint(self):
        def boom(url, timeout=25):
            raise OSError("no network")
        with patch.object(I, "_hf_json", boom):
            with self.assertRaises(OSError):
                I.hf_search("flux")     # the server layer turns this into a 502

    def test_files_filter_sort_and_assign_roles(self):
        rows = [
            {"type": "file", "path": "split_files/diffusion_models/big.gguf",
             "size": 7 * 2 ** 30},
            {"type": "file", "path": "split_files/vae/ae.safetensors", "size": 320 * 2 ** 20},
            {"type": "file", "path": "text_encoders/qwen3vl_8b_w4a8.safetensors",
             "lfs": {"size": 6 * 2 ** 30}},
            {"type": "file", "path": "README.md", "size": 10},
            {"type": "directory", "path": "split_files"},
        ]
        with patch.object(I, "_hf_json", return_value=rows):
            items = I.hf_files("Comfy-Org/z_image_turbo")
        names = [i["path"] for i in items]
        self.assertNotIn("README.md", names)
        self.assertEqual(names[0], "split_files/diffusion_models/big.gguf")
        roles = {i["path"]: i["role"] for i in items}
        self.assertEqual(roles["split_files/vae/ae.safetensors"], "vae")
        self.assertEqual(roles["text_encoders/qwen3vl_8b_w4a8.safetensors"], "te")
        sizes = {i["path"]: i["size_gb"] for i in items}
        self.assertAlmostEqual(sizes["split_files/diffusion_models/big.gguf"], 7.0, places=2)
        self.assertAlmostEqual(sizes["split_files/vae/ae.safetensors"], 0.31, places=2)
        self.assertGreater(sizes["split_files/diffusion_models/big.gguf"],
                           sizes["text_encoders/qwen3vl_8b_w4a8.safetensors"])

    def test_files_rejects_traversal_without_touching_the_network(self):
        def boom(url, timeout=25):
            raise AssertionError("network must not be used for an invalid repo")

        with patch.object(I, "_hf_json", boom):
            for repo in ("../etc", "a/..", "..", "no-slash", "a/b/c", "", None):
                with self.assertRaises(ValueError):
                    I.hf_files(repo)

    def test_guess_role(self):
        cases = {
            "models/vae/ae.safetensors": "vae",
            "split_files/vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
            "text_encoders/qwen3vl_8b_w4a8.safetensors": "te",
            "clip_l.safetensors": "te",
            "loras/mylora.safetensors": "lora",
            "sd_xl_base_1.0.safetensors": "checkpoint",
            "z-image-turbo-Q4_K_M.gguf": "unet",
        }
        for path, role in cases.items():
            self.assertEqual(I.guess_role(path), role, path)


class DownloadTests(ImageTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-dl-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name)
        self._pin([self.root])
        worker = patch.object(I, "_worker", lambda *a, **k: None)
        worker.start()
        self.addCleanup(worker.stop)
        self.addCleanup(I.reset_download)

    def test_rejects_bad_repo_and_extension(self):
        cases = [
            ("no-slash", "a.safetensors"),
            ("a/b", "weights.txt"),
            ("a/b", "../escape.safetensors"),
            ("a/../b", "x.safetensors"),
            ("../etc", "x.safetensors"),
            ("a/b", ""),
        ]
        for repo, name in cases:
            out = I.start_download(self.conf(), repo, name)
            self.assertFalse(out["ok"], (repo, name))
            self.assertEqual(out["error_code"], "invalid_download", (repo, name))

    def test_missing_engine_is_reported(self):
        self._pin([])
        out = I.start_download(self.conf(str(Path(self.temp.name) / "nope")),
                               "Qwen/Qwen-Image", "model.safetensors")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "missing_engine")

    def test_role_chooses_the_models_subfolder(self):
        out = I.start_download(self.conf(), "Comfy-Org/z_image_turbo",
                               "split_files/vae/ae.safetensors")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["role"], "vae")
        self.assertEqual(out["target"].replace("\\", "/"),
                         self.root.replace("\\", "/") + "/models/vae/ae.safetensors")

        I._state.update({"active": False})
        out = I.start_download(self.conf(), "unsloth/Z-Image-Turbo-GGUF",
                               "z-image-turbo-Q4_K_M.gguf")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["role"], "unet")
        self.assertTrue(out["target"].replace("\\", "/").endswith(
            "/models/diffusion_models/z-image-turbo-Q4_K_M.gguf"))

    def test_explicit_role_wins(self):
        out = I.start_download(self.conf(), "a/b", "clip_l.safetensors", role="lora")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["role"], "lora")
        self.assertIn("/models/loras/", out["target"].replace("\\", "/"))

    def test_one_job_at_a_time(self):
        first = I.start_download(self.conf(), "a/b", "one.safetensors")
        self.assertTrue(first["ok"], first)
        second = I.start_download(self.conf(), "a/b", "two.safetensors")
        self.assertFalse(second["ok"])
        self.assertEqual(second["error_code"], "download_busy")
        self.assertTrue(I.cancel_download()["ok"])
        self.assertFalse(I.reset_download()["ok"])          # still marked active
        I._state.update({"active": False})
        self.assertTrue(I.reset_download()["ok"])
        self.assertEqual(I.download_state()["percent"], 0.0)

    def test_percent_math(self):
        I._state.update({"active": True, "done": 50, "total": 200})
        self.assertEqual(I.download_state()["percent"], 25.0)
        I._state.update({"done": 999, "total": 200})
        self.assertEqual(I.download_state()["percent"], 100.0)
        I._state.update({"total": 0})
        self.assertEqual(I.download_state()["percent"], 0.0)

    def test_state_snapshot_is_a_copy(self):
        snap = I.download_state()
        snap["active"] = True
        self.assertFalse(I.download_state()["active"])


class StatusContractTests(ImageTest):
    """The console paints /api/image/status unconditionally: every key must exist."""

    KEYS = ("phase", "engine", "port", "host", "url", "pid", "owned", "uptime",
            "device", "device_name", "engine_version", "torch", "python_ver",
            "root", "python", "models_dir", "engine_installed", "gguf_nodes",
            "queue", "vram", "auto_open", "browser", "vram_mode", "listen",
            "log", "error", "error_code", "platform")

    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-status-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name)
        self._pin([self.root])
        for target, value in (("stats", None), ("_pid", None), ("_listener", None),
                              ("_log_tail", [])):
            patcher = patch.object(I, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        vram = patch.object(I.hostos, "vram_usage",
                            return_value={"used": 512, "total": 16303, "gpu": "TEST GPU"})
        vram.start()
        self.addCleanup(vram.stop)

    def test_down_engine_still_returns_every_key(self):
        out = I.status(self.conf())
        for key in self.KEYS:
            self.assertIn(key, out, key)
        self.assertEqual(out["phase"], "idle")
        self.assertEqual(out["queue"], {"running": 0, "pending": 0})
        self.assertEqual(out["vram"], {"used": 512, "total": 16303})
        self.assertEqual(out["url"], f"http://{I.DEFAULT_HOST}:{out['port']}")
        self.assertTrue(out["engine_installed"])
        self.assertEqual(out["log"], [])
        self.assertIn(out["platform"], ("windows", "macos", "linux"))

    def test_running_engine_reports_device_and_version(self):
        info = {"devices": [{"type": "cuda", "name": "NVIDIA GeForce RTX 5080",
                             "vram_total": 16 * 2 ** 30, "vram_free": 8 * 2 ** 30}],
                "system": {"comfyui_version": "0.43.0", "pytorch_version": "2.11.0+cu128",
                           "python_version": "3.12.9 (main, Oct 1 2025)"}}
        with patch.object(I, "stats", return_value=info), \
             patch.object(I, "queue", return_value={"running": 1, "pending": 2}), \
             patch.object(I, "_started_at", __import__("time").time() - 12):
            out = I.status(self.conf())
        self.assertEqual(out["phase"], "running")
        self.assertEqual(out["device"], "cuda")
        self.assertEqual(out["device_name"], "NVIDIA GeForce RTX 5080")
        self.assertEqual(out["engine_version"], "0.43.0")
        self.assertEqual(out["torch"], "2.11.0+cu128")
        self.assertEqual(out["python_ver"], "3.12.9")
        self.assertEqual(out["queue"], {"running": 1, "pending": 2})
        self.assertGreaterEqual(out["uptime"], 10)

    def test_loading_phase_when_pid_without_stats(self):
        with patch.object(I, "_pid", return_value=1234):
            out = I.status(self.conf())
        self.assertEqual(out["phase"], "loading")
        self.assertEqual(out["pid"], 1234)

    def test_start_without_engine_reports_missing_engine(self):
        self._pin([])
        out = I.start(self.conf(str(Path(self.temp.name) / "nope")))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "missing_engine")
        self.assertEqual(I.status(self.conf())["error_code"], "missing_engine")

    def test_selftest_without_engine_is_a_clean_failure(self):
        out = I.selftest(self.conf())
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "engine_not_running")

    def test_open_ui_without_engine_is_a_clean_failure(self):
        out = I.open_ui(self.conf())
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "engine_not_running")

    def test_stop_when_idle_is_not_an_error(self):
        out = I.stop(self.conf(), quiet=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["freed_gb"], 0.0)

    def test_stop_refuses_a_foreign_listener(self):
        with patch.object(I, "_listener", return_value=4242), \
             patch.object(I, "_is_engine", return_value=False):
            out = I.stop(self.conf())
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "port_in_use")

    def test_stop_kills_a_foreign_comfyui_listener(self):
        killed = []
        calls = {"n": 0}

        def listener(port, *a, **k):
            # the listener disappears once the process tree is killed
            calls["n"] += 1
            return 4242 if calls["n"] <= 1 else None

        with patch.object(I, "_listener", listener), \
             patch.object(I, "_is_engine", return_value=True), \
             patch.object(I.hostos, "kill_tree",
                          lambda pid, force=True: killed.append(pid) or True):
            out = I.stop(self.conf())
        self.assertTrue(out["ok"])
        self.assertEqual(killed, [4242])
        self.assertEqual(out["freed_gb"], 0.0)

    def test_is_engine_recognises_a_command_line(self):
        with patch.object(I, "_cmdline",
                          return_value='"C:/py/python.exe" main.py --listen 127.0.0.1'):
            self.assertTrue(I._is_engine(1234))
        with patch.object(I, "_cmdline", return_value='"C:/py/python.exe" other.py'):
            self.assertFalse(I._is_engine(1234))
        with patch.object(I, "_cmdline", return_value=""):
            self.assertFalse(I._is_engine(1234))


class AutoGraphTests(ImageTest):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-graph-")
        self.addCleanup(self.temp.cleanup)

    def test_builds_a_gguf_graph_when_the_nodes_exist(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3),
            ("text_encoders", "Qwen3-4B-Q8_0.gguf", 3),
            ("vae", "ae.safetensors", 2),
        ])
        objects = {name: {} for name in ("UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader",
                                         "CLIPTextEncode", "EmptySD3LatentImage",
                                         "KSampler", "VAEDecode", "SaveImage")}
        graph = I._auto_graph(objects, root)
        self.assertIsNotNone(graph)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "z-image-turbo-Q4_K_M.gguf")
        self.assertEqual(graph["2"]["inputs"]["type"], "lumina2")
        self.assertEqual(graph["6"]["inputs"]["steps"], 4)
        self.assertEqual(graph["5"]["inputs"]["width"], 512)

    def test_needs_all_three_weight_roles(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3)])
        self.assertIsNone(I._auto_graph({"UnetLoaderGGUF": {}}, root))

    def test_returns_none_without_the_loader_nodes(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3),
            ("text_encoders", "Qwen3-4B-Q8_0.gguf", 3),
            ("vae", "ae.safetensors", 2),
        ])
        self.assertIsNone(I._auto_graph({"VAELoader": {}}, root))

    def test_qwen_picks_the_dedicated_encoder_node(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "qwen-image-2.1-turbo-Q6_K.gguf", 3),
            ("text_encoders", "qwen3vl_8b_w4a8.safetensors", 3),
            ("vae", "qwen_image_2.1_vae_bf16.safetensors", 2),
        ])
        objects = {name: {} for name in ("UnetLoaderGGUF", "CLIPLoader", "CLIPLoaderGGUF",
                                         "VAELoader", "CLIPTextEncode", "TextEncodeQwenImage21",
                                         "EmptySD3LatentImage", "KSampler", "VAEDecode",
                                         "SaveImage")}
        graph = I._auto_graph(objects, root)
        # A safetensors encoder must go through the core loader, not the GGUF one.
        self.assertEqual(graph["2"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["2"]["inputs"]["type"], "qwen_image")
        self.assertEqual(graph["4"]["class_type"], "TextEncodeQwenImage21")
        self.assertEqual(graph["4"]["inputs"]["resolution"], 512)
        self.assertNotIn("vae", graph["4"]["inputs"])      # its packed latent stays unused
        self.assertEqual(graph["5"]["class_type"], "EmptySD3LatentImage")
        self.assertEqual(graph["6"]["inputs"]["negative"], ["4", 1])
        self.assertEqual(graph["6"]["inputs"]["latent_image"], ["5", 0])

    def test_qwen_vae_is_preferred_over_the_shared_ae(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "qwen-image-2.1-turbo-Q6_K.gguf", 3),
            ("text_encoders", "qwen3vl_8b_w4a8.safetensors", 3),
            ("vae", "ae.safetensors", 1),
            ("vae", "qwen_image_2.1_vae_bf16.safetensors", 2),
        ])
        objects = {name: {} for name in ("UnetLoaderGGUF", "CLIPLoader", "VAELoader",
                                         "TextEncodeQwenImage21", "EmptySD3LatentImage",
                                         "KSampler", "VAEDecode", "SaveImage")}
        graph = I._auto_graph(objects, root)
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "qwen_image_2.1_vae_bf16.safetensors")

    def test_zimage_still_uses_the_shared_ae(self):
        root = fake_checkout(self.temp.name, weights=[
            ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", 3),
            ("text_encoders", "Qwen3-4B-Q8_0.gguf", 3),
            ("vae", "ae.safetensors", 1),
            ("vae", "qwen_image_2.1_vae_bf16.safetensors", 2),
        ])
        objects = {name: {} for name in ("UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader",
                                         "CLIPTextEncode", "EmptySD3LatentImage",
                                         "KSampler", "VAEDecode", "SaveImage")}
        graph = I._auto_graph(objects, root)
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "ae.safetensors")
        self.assertEqual(graph["2"]["class_type"], "CLIPLoaderGGUF")


class ServerImageRouteTests(ImageTest):
    """The HTTP layer around the engine: routes forward (CONF) and stay additive."""

    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="herta-route-")
        self.addCleanup(self.temp.cleanup)
        self.root = fake_checkout(self.temp.name)
        self._pin([self.root])
        self.conf = settings.validate(settings.DEFAULTS)
        self.conf["image"]["comfyDir"] = self.root
        for target, value in (
                ("SETTINGS_FILE", str(Path(self.temp.name) / "settings.json")),
        ):
            patcher = patch.object(settings, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        conf_patch = patch.object(server, "CONF", self.conf)
        conf_patch.start()
        self.addCleanup(conf_patch.stop)

    def _get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return json.loads(response.read())

    def _post(self, base, path, body=None):
        data = json.dumps(body or {}).encode("utf-8")
        request = urllib.request.Request(base + path, data=data,
                                        headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def test_status_route_forwards_the_conf(self):
        seen = {}

        def fake_status(conf):
            seen["conf"] = conf
            return {"phase": "idle", "engine": "ComfyUI"}

        with patch.object(I, "status", fake_status), http_server(server.Handler) as httpd:
            data = self._get(f"http://127.0.0.1:{httpd.server_port}", "/api/image/status")
        self.assertEqual(data["phase"], "idle")
        self.assertIs(seen["conf"], self.conf)

    def test_status_route_survives_an_engine_explosion(self):
        with patch.object(I, "status", side_effect=RuntimeError("boom")), \
             http_server(server.Handler) as httpd:
            data = self._get(f"http://127.0.0.1:{httpd.server_port}", "/api/image/status")
        self.assertEqual(data["phase"], "idle")
        self.assertEqual(data["error_code"], "image_status_failed")
        self.assertIn("boom", data["error"])

    def test_models_and_recommend_routes(self):
        seen = {}

        def fake_reco(conf, lang="zh-Hant"):
            seen["lang"] = lang
            return {"picks": [{"tier": "首選"}]}

        with patch.object(I, "scan_models", lambda conf: {"ok": True, "items": [{"name": "a"}]}), \
             patch.object(I, "recommend", fake_reco), \
             http_server(server.Handler) as httpd:
            base = f"http://127.0.0.1:{httpd.server_port}"
            models_data = self._get(base, "/api/image/models")
            reco = self._get(base, "/api/image/recommend?lang=en")
        self.assertTrue(models_data["ok"])
        self.assertEqual(models_data["items"][0]["name"], "a")
        self.assertEqual(reco["picks"][0]["tier"], "首選")
        self.assertEqual(seen["lang"], "en")

    def test_hf_search_route_forwards_the_query(self):
        seen = {}

        def fake_search(query, sort, limit):
            seen.update({"query": query, "sort": sort, "limit": limit})
            return []

        with patch.object(I, "hf_search", fake_search), http_server(server.Handler) as httpd:
            data = self._get(f"http://127.0.0.1:{httpd.server_port}",
                             "/api/image/hf/search?q=flux&sort=likes&limit=7")
        self.assertTrue(data["ok"])
        self.assertEqual(seen, {"query": "flux", "sort": "likes", "limit": "7"})

    def test_hf_files_route_reports_a_bad_repo(self):
        with http_server(server.Handler) as httpd:
            data = self._get(f"http://127.0.0.1:{httpd.server_port}",
                             "/api/image/hf/files?repo=no-slash")
        self.assertFalse(data["ok"])
        self.assertEqual(data["items"], [])

    def test_download_route_needs_repo_and_file(self):
        with patch.object(I, "start_download",
                          lambda conf, repo, name, role: {"ok": True, "repo": repo}) as fake, \
             http_server(server.Handler) as httpd:
            base = f"http://127.0.0.1:{httpd.server_port}"
            bad = self._post(base, "/api/image/hf/download", {})
            good = self._post(base, "/api/image/hf/download",
                              {"repo": "a/b", "file": "x.safetensors"})
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["error_code"], "invalid_download")
        self.assertTrue(good["ok"])
        self.assertEqual(good["repo"], "a/b")

    def test_download_status_route_shape(self):
        I.reset_download()
        with http_server(server.Handler) as httpd:
            data = self._get(f"http://127.0.0.1:{httpd.server_port}",
                             "/api/image/download/status")
        for key in ("active", "done", "total", "percent", "error", "target", "role"):
            self.assertIn(key, data, key)

    def test_start_and_stop_routes_pass_conf(self):
        seen = []

        def fake_start(conf):
            seen.append(("start", conf))
            return {"ok": False, "error_code": "missing_engine"}

        def fake_stop(conf):
            seen.append(("stop", conf))
            return {"ok": True, "freed_gb": 0.0}

        with patch.object(I, "start", fake_start), patch.object(I, "stop", fake_stop), \
             http_server(server.Handler) as httpd:
            base = f"http://127.0.0.1:{httpd.server_port}"
            started = self._post(base, "/api/image/start")
            stopped = self._post(base, "/api/image/stop")
        self.assertEqual(started["error_code"], "missing_engine")
        self.assertTrue(stopped["ok"])
        self.assertEqual([name for name, _ in seen], ["start", "stop"])
        self.assertTrue(all(conf is self.conf for _, conf in seen))

    def test_text_routes_still_answer(self):
        """The image work must not disturb the existing control contract."""
        with patch.object(server, "_pid", return_value=None), \
             patch.object(server, "_alive", return_value=False), \
             patch.object(server, "_health", return_value=False), \
             patch.object(server, "_tail", return_value=[]), \
             patch.object(server, "_vram_cached",
                          return_value={"used": 0, "total": 0, "gpu": "t"}), \
             patch.object(server, "_lan_ip_cached", return_value="127.0.0.1"), \
             patch.object(models, "MODELS_DIR", self.temp.name), \
             patch.object(models, "gpu_total_gb", return_value=16), \
             patch.object(models, "scan_local", return_value=[]), \
             http_server(server.Handler) as httpd:
            base = f"http://127.0.0.1:{httpd.server_port}"
            status = self._get(base, "/api/status")
            library = self._get(base, "/api/models")
            settings_data = self._get(base, "/api/settings")
        self.assertIn("phase", status)
        self.assertIn("endpoint", status)
        self.assertIn("items", library)
        self.assertTrue(settings_data["ok"])
        self.assertIn("image", settings_data["defaults"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Юнит-тесты логики servstart (llm.py) — без GTK, без сети, без /mnt/models.

Запуск:  /usr/bin/python3 -m unittest discover -s tests -v
(или просто  python3 tests/test_llm.py)
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402


# ---------------------------------------------------------------------------
# Минимальный валидный GGUF для тестов (только metadata, тензоры не читаются)
# ---------------------------------------------------------------------------

def _kv_str(key, value):
    b = value.encode("utf-8")
    return struct.pack("<Q", len(key)) + key.encode("utf-8") + \
        struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b


def make_gguf(path, arch, name, size=1024, gtype="model"):
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))          # version
        f.write(struct.pack("<Q", 0))          # tensor_count
        kvs = [("general.architecture", arch),
               ("general.name", name),
               ("general.type", gtype)]
        f.write(struct.pack("<Q", len(kvs)))
        for k, v in kvs:
            f.write(_kv_str(k, v))
        # дописываем до нужного размера (имитация тензоров)
        f.write(b"\x00" * max(0, size - f.tell()))
    return path


CATALOG = {
    "backends": [
        {"id": "llamacpp", "name": "llama.cpp", "binaries": ["llama-server"],
         "search_paths": ["~/llama.cpp/build-hip/bin"], "formats": ["gguf"],
         "exclude_arch": ["deepseek4", "deepseek4-dflash-draft"],
         "port_default": 8080, "health": "/health", "models_route": "/v1/models",
         "env": {"HIP_VISIBLE_DEVICES": "0"},
         "argv_template": ["{binary}", "-m", "{model}", "--alias", "{alias}",
                           "--host", "{host}", "--port", "{port}",
                           "-ngl", "{gpu_layers}", "-fa", "{flash_attention}",
                           "-t", "{threads}", "-c", "{context}",
                           "--jinja", "--metrics", "{mtp_args}", "{mmproj_args}", "{extra_args}"],
         "settings": [
             {"key": "host", "type": "str", "default": "127.0.0.1"},
             {"key": "port", "type": "int", "default": 8080, "min": 1, "max": 65535},
             {"key": "context", "type": "int", "default": 65536, "min": 2048, "max": 262144},
             {"key": "threads", "type": "int", "default": 16},
             {"key": "gpu_layers", "type": "int", "default": 999},
             {"key": "flash_attention", "type": "bool", "default": True,
              "on_value": "on", "off_value": "off"},
             {"key": "mtp", "type": "bool", "default": False},
             {"key": "mmproj", "type": "bool", "default": True},
             {"key": "extra_args", "type": "str", "split": True, "default": ""},
         ]},
        {"id": "lucebox", "name": "Lucebox", "binaries": ["luce_server"],
         "search_paths": ["~/lucebox/server/build-hip"], "formats": ["gguf"],
         "arch": ["deepseek4", "deepseek4-dflash-draft"],
         "port_default": 8081, "health": "/health", "models_route": "/v1/models",
         "env": {},
         "argv_template": ["{binary}", "{model}", "--target-device", "{target}",
                           "--port", "{port}", "--max-ctx", "{context}",
                           "--profile", "{profile}", "{draft_args}", "{extra_args}"],
         "settings": [
             {"key": "target", "type": "str", "default": "hip:0"},
             {"key": "port", "type": "int", "default": 8081},
             {"key": "context", "type": "int", "default": 131072},
             {"key": "profile", "type": "str", "default": "ds4-strix"},
             {"key": "draft", "type": "bool", "default": True},
             {"key": "extra_args", "type": "str", "split": True, "default": ""},
         ]},
    ]
}


def load_catalog():
    d = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(CATALOG, d)
    d.close()
    return llm.load_catalog(d.name), d.name


class TestGguf(unittest.TestCase):
    def test_read_meta(self):
        p = make_gguf("/tmp/_t_gguf.gguf", "qwen4exp", "Qwen3.8 Flash Next")
        try:
            m = llm.read_gguf_meta(p)
            self.assertEqual(m["general.architecture"], "qwen4exp")
            self.assertEqual(m["general.name"], "Qwen3.8 Flash Next")
            self.assertEqual(m["gguf_version"], 3)
        finally:
            os.remove(p)

    def test_not_gguf(self):
        p = "/tmp/_t_notgguf.bin"
        with open(p, "wb") as f:
            f.write(b"NOTGGUF!!")
        try:
            m = llm.read_gguf_meta(p)
            self.assertIn("_error", m)
        finally:
            os.remove(p)


class TestMatch(unittest.TestCase):
    def setUp(self):
        self.catalog, self.catfile = load_catalog()

    def tearDown(self):
        os.remove(self.catfile)

    def _m(self, arch):
        return llm.Model("/x/%s.gguf" % arch, "M", arch, 100, None)

    def test_qwen_to_llamacpp(self):
        self.assertEqual(llm.match_model_to_backend(self._m("qwen4exp"), self.catalog),
                         "llamacpp")

    def test_llama_arch_to_llamacpp(self):
        self.assertEqual(llm.match_model_to_backend(self._m("llama"), self.catalog),
                         "llamacpp")

    def test_deepseek4_to_lucebox(self):
        self.assertEqual(llm.match_model_to_backend(self._m("deepseek4"), self.catalog),
                         "lucebox")

    def test_deepseek_draft_to_lucebox(self):
        self.assertEqual(llm.match_model_to_backend(self._m("deepseek4-dflash-draft"),
                                                    self.catalog), "lucebox")

    def test_deepseek4_not_llamacpp(self):
        # глубокий контроль: llamacpp исключает deepseek4
        b = self.catalog["llamacpp"]
        self.assertIn("deepseek4", b.exclude_arch)


class TestBuildArgv(unittest.TestCase):
    def setUp(self):
        self.catalog, self.catfile = load_catalog()
        self.cfg = {}

    def tearDown(self):
        os.remove(self.catfile)

    def _bind(self, bid):
        b = self.catalog[bid]
        b.binary_path = "/fake/bin/%s" % ("llama-server" if bid == "llamacpp" else "luce_server")
        return b

    def test_llamacpp_defaults(self):
        b = self._bind("llamacpp")
        m = llm.Model("/m/q.gguf", "Qwen", "qwen4exp", 1, "llamacpp")
        argv = llm.build_argv(b, m, self.cfg)
        self.assertEqual(argv[0], "/fake/bin/llama-server")
        self.assertIn("-m", argv); self.assertIn("/m/q.gguf", argv)
        self.assertIn("-c", argv); self.assertIn("65536", argv)
        self.assertIn("-t", argv); self.assertIn("16", argv)
        self.assertIn("-ngl", argv); self.assertIn("999", argv)
        self.assertIn("-fa", argv); self.assertIn("on", argv)
        self.assertIn("--jinja", argv); self.assertIn("--metrics", argv)
        # mtp выключен -> нет -md
        self.assertNotIn("-md", argv)

    def test_llamacpp_mtp_and_mmproj(self):
        b = self._bind("llamacpp")
        m = llm.Model("/m/q.gguf", "Qwen", "qwen4exp", 1, "llamacpp",
                      aux={"mtp": "/m/MTP/mtp.gguf", "mmproj": "/m/mmproj.gguf"})
        cfg = {"backends": {"llamacpp": {"mtp": True}}}
        argv = llm.build_argv(b, m, cfg)
        self.assertIn("-md", argv)
        self.assertIn("/m/MTP/mtp.gguf", argv)
        self.assertIn("--spec-type", argv)
        self.assertIn("--mmproj", argv)
        self.assertIn("/m/mmproj.gguf", argv)

    def test_llamacpp_override_port_context(self):
        b = self._bind("llamacpp")
        m = llm.Model("/m/q.gguf", "Qwen", "qwen4exp", 1, "llamacpp")
        cfg = {"backends": {"llamacpp": {"port": 9000, "context": 32768}}}
        argv = llm.build_argv(b, m, cfg)
        self.assertIn("9000", argv)
        self.assertIn("32768", argv)
        self.assertNotIn("65536", argv)

    def test_lucebox_defaults(self):
        b = self._bind("lucebox")
        m = llm.Model("/m/d.gguf", "DS4", "deepseek4", 1, "lucebox")
        argv = llm.build_argv(b, m, self.cfg)
        self.assertEqual(argv[0], "/fake/bin/luce_server")
        self.assertIn("--target-device", argv); self.assertIn("hip:0", argv)
        self.assertIn("--profile", argv); self.assertIn("ds4-strix", argv)
        self.assertIn("--max-ctx", argv); self.assertIn("131072", argv)
        # draft без файла -> нет --draft
        self.assertNotIn("--draft", argv)

    def test_lucebox_draft(self):
        b = self._bind("lucebox")
        m = llm.Model("/m/d.gguf", "DS4", "deepseek4", 1, "lucebox",
                      aux={"draft": "/m-draft/draft.gguf"})
        argv = llm.build_argv(b, m, self.cfg)
        self.assertIn("--draft", argv)
        self.assertIn("/m-draft/draft.gguf", argv)

    def test_extra_args_split(self):
        b = self._bind("llamacpp")
        m = llm.Model("/m/q.gguf", "Qwen", "qwen4exp", 1, "llamacpp")
        cfg = {"backends": {"llamacpp": {"extra_args": "-ot per_layer_token_embd.weight=CPU --n-cpu-moe 2"}}}
        argv = llm.build_argv(b, m, cfg)
        self.assertIn("-ot", argv)
        self.assertIn("per_layer_token_embd.weight=CPU", argv)
        self.assertIn("--n-cpu-moe", argv)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.catalog, self.catfile = load_catalog()

    def tearDown(self):
        os.remove(self.catfile)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            data = {"backends": {"llamacpp": {"port": 9999}}}
            llm.save_config(data, p)
            self.assertEqual(llm.load_config(p), data)

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(llm.load_config(os.path.join(d, "nope.json")), {})

    def test_load_corrupt(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            with open(p, "w") as f:
                f.write("{not json")
            self.assertEqual(llm.load_config(p), {})

    def test_effective_defaults_then_overrides(self):
        m = llm.Model("/m/q.gguf", "Qwen", "qwen4exp", 1, "llamacpp")
        b = self.catalog["llamacpp"]
        # только defaults
        s = llm.effective_settings(b, m, {})
        self.assertEqual(s["port"], 8080)
        self.assertEqual(s["context"], 65536)
        # backend override
        s = llm.effective_settings(b, m, {"backends": {"llamacpp": {"port": 9000}}})
        self.assertEqual(s["port"], 9000)
        # model override побеждает
        s = llm.effective_settings(b, m, {
            "backends": {"llamacpp": {"port": 9000}},
            "models": {"/m/q.gguf": {"port": 7777}},
        })
        self.assertEqual(s["port"], 7777)


class TestFindModels(unittest.TestCase):
    def setUp(self):
        self.catalog, self.catfile = load_catalog()

    def tearDown(self):
        os.remove(self.catfile)

    def test_grouping_and_aux(self):
        with tempfile.TemporaryDirectory() as root:
            q = os.path.join(root, "qwen38-fn")
            os.makedirs(os.path.join(q, "MTP"))
            make_gguf(os.path.join(q, "Qwen-merged.gguf"), "qwen4exp", "Qwen", 2048)
            make_gguf(os.path.join(q, "mmproj-F16.gguf"), "clip", "mmproj", 900, gtype="mmproj")
            make_gguf(os.path.join(q, "MTP", "mtp.gguf"), "qwen4exp", "Ckpt", 700)
            d4 = os.path.join(root, "ds4-flash-0731")
            os.makedirs(d4)
            make_gguf(os.path.join(d4, "DS4.gguf"), "deepseek4", "DS4", 4096)
            dd = os.path.join(root, "ds4-flash-0731-draft")
            os.makedirs(dd)
            make_gguf(os.path.join(dd, "draft.gguf"), "deepseek4-dflash-draft", "draft", 1000)

            models = llm.find_models([root])
            # mmproj (clip) исключён из моделей; MTP-каталог приклеен к qwen
            self.assertEqual(len(models), 2)
            by_arch = {m.arch: m for m in models}
            self.assertIn("qwen4exp", by_arch)
            self.assertIn("deepseek4", by_arch)
            qwen = by_arch["qwen4exp"]
            self.assertEqual(qwen.name, "Qwen")
            self.assertIn("mmproj", qwen.aux)
            self.assertIn("mtp", qwen.aux)
            ds4 = by_arch["deepseek4"]
            self.assertIn("draft", ds4.aux)
            self.assertTrue(ds4.aux["draft"].endswith("draft.gguf"))

    def test_match_after_find(self):
        with tempfile.TemporaryDirectory() as root:
            make_gguf(os.path.join(root, "m.gguf"), "deepseek4", "DS4", 2048)
            models = llm.find_models([root])
            self.assertEqual(len(models), 1)
            self.assertEqual(llm.match_model_to_backend(models[0], self.catalog), "lucebox")


if __name__ == "__main__":
    unittest.main(verbosity=2)

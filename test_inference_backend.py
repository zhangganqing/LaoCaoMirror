# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as core


class InferenceBackendTests(unittest.TestCase):
    def test_cpu_backend_forces_cpu_even_when_cuda_is_available(self):
        providers = core.execution_providers(
            "cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_gpu_backend_requires_cuda_provider(self):
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            core.execution_providers("gpu", ["CPUExecutionProvider"])

    def test_gpu_backend_prefers_cuda_without_silent_cpu_fallback(self):
        providers = core.execution_providers(
            "gpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.assertEqual(providers, ["CUDAExecutionProvider"])

    def test_auto_backend_uses_available_acceleration(self):
        self.assertEqual(
            core.resolve_compute_backend(
                "auto", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
            "gpu")
        self.assertEqual(
            core.resolve_compute_backend("auto", ["CPUExecutionProvider"]),
            "cpu")

    def test_environment_selects_fixed_distribution_backend(self):
        with patch.dict(os.environ, {"LCSB_COMPUTE_BACKEND": "cpu"}):
            self.assertEqual(core.requested_compute_backend(), "cpu")
        with patch.dict(os.environ, {"LCSB_COMPUTE_BACKEND": "gpu"}):
            self.assertEqual(core.requested_compute_backend(), "gpu")

    def test_new_cpu_install_uses_low_load_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            with patch.dict(os.environ, {"LCSB_COMPUTE_BACKEND": "cpu"}):
                cfg = core.load_config(str(config_path))

        self.assertEqual(cfg["detect_imgsz"], 640)
        self.assertEqual(cfg["fps"], 8)
        self.assertEqual(cfg["camera_resolution"], [1280, 720])

    def test_real_cpu_session_uses_cpu_provider(self):
        model = Path(__file__).parent / "models" / "scrfd_2.5g.onnx"

        session = core.create_onnx_session(str(model), "cpu")

        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])

    def test_gpu_session_rejects_runtime_fallback_to_cpu(self):
        fallback_session = type("Session", (), {
            "get_providers": lambda self: ["CPUExecutionProvider"]})()
        with (patch.object(
                core.ort, "get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
              patch.object(core.ort, "InferenceSession",
                           return_value=fallback_session)):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*加载失败"):
                core.create_onnx_session("model.onnx", "gpu")

    def test_model_cache_key_includes_resolved_backend(self):
        cpu_key = core.model_cache_key(
            {"models_dir": "models", "gallery_dir": "laocao"}, "cpu")
        gpu_key = core.model_cache_key(
            {"models_dir": "models", "gallery_dir": "laocao"}, "gpu")

        self.assertNotEqual(cpu_key, gpu_key)

    def test_runtime_report_lists_all_three_model_providers(self):
        detector = type("Detector", (), {"provider": "CUDAExecutionProvider"})()
        aligner = type("Aligner", (), {"provider": "CUDAExecutionProvider"})()
        recognizer = type("Recognizer", (), {
            "provider": "CUDAExecutionProvider", "aligner": aligner})()

        report = core.model_runtime_report(detector, recognizer, "GPU")

        self.assertIn("GPU版", report)
        self.assertIn("检测=CUDAExecutionProvider", report)
        self.assertIn("对齐=CUDAExecutionProvider", report)
        self.assertIn("识别=CUDAExecutionProvider", report)


class DistributionProfileTests(unittest.TestCase):
    def test_cpu_and_gpu_dependency_profiles_are_mutually_exclusive(self):
        root = Path(__file__).parent
        cpu = (root / "requirements-cpu.txt").read_text(encoding="utf-8").lower()
        gpu = (root / "requirements-gpu.txt").read_text(encoding="utf-8").lower()

        self.assertIn("onnxruntime==", cpu)
        self.assertNotIn("onnxruntime-gpu", cpu)
        self.assertIn("onnxruntime-gpu==", gpu)
        for forbidden in ("torch", "torchvision", "ultralytics"):
            self.assertNotIn(forbidden, cpu)
            self.assertNotIn(forbidden, gpu)

    def test_fixed_launchers_select_their_own_backend_before_importing_ui(self):
        root = Path(__file__).parent
        cpu = (root / "ui_cpu.py").read_text(encoding="utf-8")
        gpu = (root / "ui_gpu.py").read_text(encoding="utf-8")

        self.assertLess(cpu.index('LCSB_COMPUTE_BACKEND"] = "cpu"'),
                        cpu.index("from ui import main"))
        self.assertLess(gpu.index('LCSB_COMPUTE_BACKEND"] = "gpu"'),
                        gpu.index("from ui import main"))

    def test_build_profiles_use_distinct_entries_without_pytorch(self):
        root = Path(__file__).parent
        cpu = (root / "LaoCaoMirrorCPU.spec").read_text(encoding="utf-8")
        gpu = (root / "LaoCaoMirrorGPU.spec").read_text(encoding="utf-8")

        self.assertIn("ui_cpu.py", cpu)
        self.assertIn("name='LaoCaoMirror-CPU'", cpu)
        self.assertIn("ui_gpu.py", gpu)
        self.assertIn("name='LaoCaoMirror-GPU'", gpu)
        for profile in (cpu, gpu):
            self.assertIn("yolov8n-face.onnx", profile)
            self.assertNotIn("yolov8n-face.pt", profile)
            for forbidden in ("collect_all('ultralytics')", "collect_all('torchvision')"):
                self.assertNotIn(forbidden, profile)

    def test_build_scripts_target_their_matching_specs(self):
        root = Path(__file__).parent
        cpu = (root / "build_cpu.ps1").read_text(encoding="utf-8")
        gpu = (root / "build_gpu.ps1").read_text(encoding="utf-8")

        self.assertIn("requirements-cpu.txt", cpu)
        self.assertIn("LaoCaoMirrorCPU.spec", cpu)
        self.assertIn("requirements-gpu.txt", gpu)
        self.assertIn("LaoCaoMirrorGPU.spec", gpu)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as core
import ui as app_ui


class InferenceBackendTests(unittest.TestCase):
    def test_cpu_backend_forces_cpu_even_when_cuda_is_available(self):
        providers = core.execution_providers(
            "cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_gpu_backend_requires_cuda_provider(self):
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider") as caught:
            core.execution_providers("gpu", ["CPUExecutionProvider"])
        self.assertIn("GPU安装教程.md", str(caught.exception))
        self.assertIn("CUDA 12.x", str(caught.exception))

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

    def test_gpu_session_preloads_cuda_and_cudnn_dlls(self):
        cuda_session = type("Session", (), {
            "get_providers": lambda self: ["CUDAExecutionProvider"]})()
        with (patch.object(
                core.ort, "get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
              patch.object(core.ort, "preload_dlls") as preload,
              patch.object(core.ort, "InferenceSession",
                           return_value=cuda_session)):
            core.create_onnx_session("model.onnx", "gpu")

        preload.assert_called_once_with()

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

    def test_gpu_setup_message_contains_actionable_install_steps(self):
        message = core.gpu_setup_message(
            "找不到 cublasLt64_12.dll", guide_path=r"C:\app\GPU安装教程.md")

        for expected in (
                "CUDA 12.x", "cuDNN 9.x", "Visual C++",
                "onnxruntime-gpu", r"C:\app\GPU安装教程.md",
                "nvidia-smi", "where.exe cublasLt64_12.dll",
                "where.exe cudnn64_9.dll"):
            self.assertIn(expected, message)


class DistributionProfileTests(unittest.TestCase):
    def test_cpu_and_gpu_dependency_profiles_are_mutually_exclusive(self):
        root = Path(__file__).parent
        cpu = (root / "requirements-cpu.txt").read_text(encoding="utf-8").lower()
        gpu = (root / "requirements-gpu.txt").read_text(encoding="utf-8").lower()
        gpu_source = (root / "requirements-gpu-source.txt").read_text(
            encoding="utf-8").lower()

        self.assertIn("onnxruntime==", cpu)
        self.assertNotIn("onnxruntime-gpu", cpu)
        self.assertIn("onnxruntime-gpu==", gpu)
        self.assertNotIn("[cuda,cudnn]", gpu)
        self.assertIn("onnxruntime-gpu[cuda,cudnn]==", gpu_source)
        for forbidden in ("torch", "torchvision", "ultralytics"):
            self.assertNotIn(forbidden, cpu)
            self.assertNotIn(forbidden, gpu)
            self.assertNotIn(forbidden, gpu_source)

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

    def test_gpu_model_failure_emits_setup_help_signal(self):
        engine = app_ui.Engine({}, source=None)
        messages = []
        engine.gpu_setup_required.connect(messages.append)

        with (patch.dict(os.environ, {"LCSB_COMPUTE_BACKEND": "gpu"}),
              patch.object(app_ui.core, "build_models",
                           side_effect=RuntimeError("GPU 环境未安装完整"))):
            engine.run()

        self.assertEqual(messages, ["GPU 环境未安装完整"])

    def test_gpu_ui_exposes_install_help_button_and_handler(self):
        source = Path(app_ui.__file__).read_text(encoding="utf-8")

        self.assertIn("GPU 安装帮助", source)
        self.assertIn("open_gpu_setup_guide", source)
        self.assertIn("show_gpu_setup_help", source)

    def test_gpu_guide_contains_official_links_and_verification_commands(self):
        root = Path(__file__).parent
        guide = (root / "GPU安装教程.md").read_text(encoding="utf-8")

        for expected in (
                "打包版", "源码版", "requirements-gpu-source.txt",
                "onnxruntime-gpu[cuda,cudnn]",
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                "https://developer.nvidia.com/cuda-downloads",
                "https://developer.nvidia.com/cudnn",
                "https://aka.ms/vs/17/release/vc_redist.x64.exe",
                "nvidia-smi", "where.exe cublasLt64_12.dll",
                "where.exe cudnn64_9.dll"):
            self.assertIn(expected, guide)

    def test_gpu_package_and_readme_include_install_guide(self):
        root = Path(__file__).parent
        spec = (root / "LaoCaoMirrorGPU.spec").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertIn("GPU安装教程.md", spec)
        self.assertIn("GPU安装教程.md", readme)


if __name__ == "__main__":
    unittest.main()

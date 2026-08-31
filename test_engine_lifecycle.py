# -*- coding: utf-8 -*-
"""设置保存和摄像头重启不能阻塞 Qt 界面线程。"""
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import main as core
import ui


class _SlowEngine:
    def __init__(self):
        self.stop_calls = 0
        self.wait_calls = 0

    def stop(self):
        self.stop_calls += 1

    def wait(self, _milliseconds):
        self.wait_calls += 1
        time.sleep(0.05)
        return False


class EngineLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, gallery_dir):
        cfg = dict(core.DEFAULTS)
        cfg["gallery_dir"] = gallery_dir
        with patch("ui.list_visible_windows", return_value=[]):
            return ui.MainWindow(cfg)

    def test_stop_request_never_waits_in_gui_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self.make_window(tmp)
            engine = _SlowEngine()
            window.engine = engine

            started = time.perf_counter()
            window.stop_engine()
            elapsed = time.perf_counter() - started

            self.assertEqual(engine.stop_calls, 1)
            self.assertEqual(engine.wait_calls, 0, "Qt 主线程不应同步等待摄像头线程")
            self.assertLess(elapsed, 0.03)
            self.assertIs(window.engine, engine, "线程真正结束前必须保留引用")
            window.engine = None

    def test_stale_engine_completion_does_not_clear_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self.make_window(tmp)
            old_engine = _SlowEngine()
            replacement = _SlowEngine()
            window.engine = replacement

            window.on_engine_finished(old_engine)

            self.assertIs(window.engine, replacement)
            window.engine = None

    def test_camera_restart_waits_for_old_engine_finished_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self.make_window(tmp)
            old_engine = _SlowEngine()
            replacement = _SlowEngine()
            window.engine = old_engine
            window.start_engine = Mock(side_effect=lambda: setattr(window, "engine", replacement))

            window.restart_engine()

            self.assertEqual(old_engine.stop_calls, 1)
            self.assertIs(window.engine, old_engine)
            window.start_engine.assert_not_called()

            window.on_engine_finished(old_engine)

            window.start_engine.assert_called_once_with()
            self.assertIs(window.engine, replacement)
            window.engine = None


if __name__ == "__main__":
    unittest.main()

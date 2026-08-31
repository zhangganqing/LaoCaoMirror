# -*- coding: utf-8 -*-
"""人工复核按钮的界面状态与信号转发。"""
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import main as core
from ui import MainWindow


class _DecisionRecorder:
    def __init__(self):
        self.restored = 0

    def confirm_restore(self):
        self.restored += 1


class ReviewControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        cfg = dict(core.DEFAULTS)
        cfg["guard_window_keyword"] = ""
        cfg["gallery_dir"] = self.temp.name
        self.window = MainWindow(cfg)

    def tearDown(self):
        self.window.engine = None
        self.window.close()
        self.temp.cleanup()

    def test_restore_button_is_only_enabled_during_defense(self):
        self.assertTrue(hasattr(self.window, "btn_restore"), "恢复按钮尚未实现")
        self.assertFalse(self.window.btn_restore.isEnabled())

        self.window.show_state("DEFEND")
        self.assertTrue(self.window.btn_restore.isEnabled())

        self.window.show_state("PATROL")
        self.assertFalse(self.window.btn_restore.isEnabled())

    def test_restore_button_only_forwards_restore(self):
        self.assertTrue(hasattr(self.window, "btn_restore"), "恢复按钮尚未实现")
        recorder = _DecisionRecorder()
        self.window.engine = recorder
        self.window.show_state("DEFEND")

        self.window.btn_restore.click()

        self.assertEqual(recorder.restored, 1)

    def test_review_button_displays_persistent_queue_count(self):
        self.assertTrue(hasattr(self.window, "btn_review"), "审核记录按钮尚未实现")
        queue = core.ReviewQueue(self.temp.name)
        queue.start_event(__import__("numpy").zeros((20, 20, 3), dtype="uint8"), 0.61)

        self.window.refresh_review_count()

        self.assertIn("1", self.window.btn_review.text())


if __name__ == "__main__":
    unittest.main()

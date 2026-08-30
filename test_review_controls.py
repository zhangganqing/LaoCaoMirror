# -*- coding: utf-8 -*-
"""人工复核按钮的界面状态与信号转发。"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import main as core
from ui import MainWindow


class _DecisionRecorder:
    def __init__(self):
        self.confirmed = 0
        self.rejected = 0

    def confirm_and_learn(self):
        self.confirmed += 1

    def reject_and_restore(self):
        self.rejected += 1


class ReviewControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        cfg = dict(core.DEFAULTS)
        cfg["guard_window_keyword"] = ""
        self.window = MainWindow(cfg)

    def tearDown(self):
        self.window.engine = None
        self.window.close()

    def test_buttons_are_only_enabled_while_waiting_for_review(self):
        self.assertTrue(hasattr(self.window, "btn_confirm_learn"), "确认学习按钮尚未实现")
        self.assertTrue(hasattr(self.window, "btn_reject"), "误报按钮尚未实现")
        self.assertFalse(self.window.btn_confirm_learn.isEnabled())
        self.assertFalse(self.window.btn_reject.isEnabled())

        self.window.show_state("DEFEND")
        self.assertTrue(self.window.btn_confirm_learn.isEnabled())
        self.assertTrue(self.window.btn_reject.isEnabled())

        self.window.show_state("PATROL")
        self.assertFalse(self.window.btn_confirm_learn.isEnabled())
        self.assertFalse(self.window.btn_reject.isEnabled())

    def test_each_button_forwards_only_its_own_decision(self):
        self.assertTrue(hasattr(self.window, "btn_confirm_learn"), "确认学习按钮尚未实现")
        recorder = _DecisionRecorder()
        self.window.engine = recorder
        self.window.show_state("DEFEND")

        self.window.btn_confirm_learn.click()
        self.window.show_state("DEFEND")
        self.window.btn_reject.click()

        self.assertEqual(recorder.confirmed, 1)
        self.assertEqual(recorder.rejected, 1)


if __name__ == "__main__":
    unittest.main()

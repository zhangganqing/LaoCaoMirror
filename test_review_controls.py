# -*- coding: utf-8 -*-
"""人工复核按钮的界面状态与信号转发。"""
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import main as core
import ui
from ui import MainWindow


class _DecisionRecorder:
    def __init__(self):
        self.restored = 0

    def confirm_restore(self):
        self.restored += 1


class _AlwaysLearnRecognizer:
    def learn_face(self, _image, **_kwargs):
        return {"status": "learned", "path": "learned_ui_test.jpg"}


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

    def test_review_dialog_lists_records_and_confirms_checked_items(self):
        self.assertTrue(hasattr(ui, "ReviewDialog"), "批量审核对话框尚未实现")
        queue = core.ReviewQueue(self.temp.name)
        np = __import__("numpy")
        first = queue.start_event(np.full((20, 20, 3), 80, dtype="uint8"), 0.61)
        keep = queue.start_event(np.full((20, 20, 3), 100, dtype="uint8"), 0.71)
        dialog = ui.ReviewDialog(queue, _AlwaysLearnRecognizer())

        self.assertEqual(dialog.items.count(), 2)
        for index in range(dialog.items.count()):
            item = dialog.items.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == first:
                item.setCheckState(Qt.CheckState.Checked)
        dialog.confirm_selected()

        self.assertEqual([item["id"] for item in queue.list_items()], [keep])


if __name__ == "__main__":
    unittest.main()

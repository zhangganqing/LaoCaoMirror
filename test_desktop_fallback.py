# -*- coding: utf-8 -*-
"""未配置护身窗口时使用 Win+D 的防御与恢复兜底。"""
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from test_face_learning import (
    _LoopCapture, _LoopCaptureSource, _LoopDetector, _LoopRecognizer,
)

import main as core


class DesktopFallbackTests(unittest.TestCase):
    def _cfg(self, gallery_dir):
        cfg = dict(core.DEFAULTS)
        cfg.update({"gallery_dir": gallery_dir, "fps": 1000, "confirm_frames": 1,
                    "roi": None, "mirror": False, "guard_window_keyword": ""})
        return cfg

    def test_toggle_desktop_sends_windows_d_key_sequence(self):
        self.assertTrue(hasattr(core, "toggle_desktop"), "Win+D 桌面切换尚未实现")
        fake_user32 = Mock()

        with patch.object(core, "user32", fake_user32):
            core.toggle_desktop()

        self.assertEqual(fake_user32.keybd_event.call_count, 4)
        self.assertEqual(fake_user32.keybd_event.call_args_list[0].args[:3], (0x5B, 0, 0))
        self.assertEqual(fake_user32.keybd_event.call_args_list[1].args[:3], (0x44, 0, 0))

    def test_missing_guard_toggles_desktop_on_defense_and_restore(self):
        self.assertTrue(hasattr(core, "toggle_desktop"), "Win+D 桌面切换尚未实现")
        with tempfile.TemporaryDirectory() as gallery_dir:
            restore_event = threading.Event()
            cb = _LoopCapture(restore_event)
            with (patch.object(core, "beep"),
                  patch.object(core, "toggle_desktop") as desktop,
                  patch.object(core, "restore_screen") as restore):
                core.run_loop(
                    self._cfg(gallery_dir), _LoopCaptureSource(), _LoopDetector(),
                    _LoopRecognizer(), None, cb, restore_event=restore_event)

        self.assertEqual(desktop.call_count, 2)
        restore.assert_not_called()
        self.assertTrue(any("桌面" in text for text in cb.events))

    def test_existing_guard_uses_window_switch_without_desktop_toggle(self):
        self.assertTrue(hasattr(core, "toggle_desktop"), "Win+D 桌面切换尚未实现")
        with tempfile.TemporaryDirectory() as gallery_dir:
            restore_event = threading.Event()
            cb = _LoopCapture(restore_event)
            fake_user32 = Mock()
            fake_user32.IsWindow.return_value = True
            fake_user32.GetForegroundWindow.return_value = 123
            with (patch.object(core, "beep"),
                  patch.object(core, "user32", fake_user32),
                  patch.object(core, "toggle_desktop") as desktop,
                  patch.object(core, "switch_screen", return_value=456) as switch,
                  patch.object(core, "restore_screen") as restore):
                core.run_loop(
                    self._cfg(gallery_dir), _LoopCaptureSource(), _LoopDetector(),
                    _LoopRecognizer(), 123, cb, restore_event=restore_event)

        desktop.assert_not_called()
        switch.assert_called_once_with(123)
        restore.assert_called_once_with(123, 456)


if __name__ == "__main__":
    unittest.main()

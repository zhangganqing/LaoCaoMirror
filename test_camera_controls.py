# -*- coding: utf-8 -*-
"""摄像头对焦、曝光控制和检测亮度均衡。"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

import main as core
import ui


class _FakeCapture:
    def __init__(self, unsupported=()):
        self.unsupported = set(unsupported)
        self.values = {}
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop in self.unsupported:
            return False
        self.values[prop] = value
        return True

    def get(self, prop):
        return self.values.get(prop, 0.0)


class CameraControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_manual_controls_set_supported_properties_and_report_unsupported(self):
        self.assertTrue(hasattr(core, "apply_camera_controls"), "摄像头属性控制尚未实现")
        cap = _FakeCapture(unsupported={cv2.CAP_PROP_BACKLIGHT})
        cfg = {
            "camera_autofocus": False,
            "camera_focus": 180,
            "camera_auto_exposure": False,
            "camera_exposure": -6,
            "camera_backlight": 1,
        }

        report = core.apply_camera_controls(cap, cfg)

        self.assertIn((cv2.CAP_PROP_AUTOFOCUS, 0.0), cap.calls)
        self.assertIn((cv2.CAP_PROP_FOCUS, 180.0), cap.calls)
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25), cap.calls)
        self.assertIn((cv2.CAP_PROP_EXPOSURE, -6.0), cap.calls)
        self.assertIn("backlight", report["unsupported"])
        self.assertEqual(report["actual"]["focus"], 180.0)

    def test_automatic_modes_do_not_override_manual_focus_or_exposure(self):
        self.assertTrue(hasattr(core, "apply_camera_controls"), "摄像头属性控制尚未实现")
        cap = _FakeCapture()

        core.apply_camera_controls(cap, {
            "camera_autofocus": True,
            "camera_focus": 200,
            "camera_auto_exposure": True,
            "camera_exposure": -8,
            "camera_backlight": 0,
        })

        props = [prop for prop, _ in cap.calls]
        self.assertNotIn(cv2.CAP_PROP_FOCUS, props)
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, props)

    def test_detection_equalization_improves_local_contrast_without_mutating_input(self):
        self.assertTrue(hasattr(core, "prepare_detection_frame"), "检测亮度均衡尚未实现")
        x = np.tile(np.arange(40, 80, dtype=np.uint8), (80, 2))
        frame = cv2.merge([x, x, x])
        original = frame.copy()

        enhanced = core.prepare_detection_frame(frame, {"detection_equalize": True})

        np.testing.assert_array_equal(frame, original)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertFalse(np.array_equal(enhanced, frame))
        self.assertGreater(float(enhanced.std()), float(frame.std()))

    @patch("ui.list_visible_windows", return_value=[])
    def test_settings_expose_camera_controls_and_save_values(self, _windows):
        cfg = dict(core.DEFAULTS)
        dialog = ui.SettingsDialog(cfg)
        self.assertTrue(hasattr(dialog, "autofocus"), "自动对焦设置尚未实现")
        self.assertTrue(hasattr(dialog, "auto_exposure"), "自动曝光设置尚未实现")

        dialog.autofocus.setChecked(False)
        dialog.focus.setValue(190)
        dialog.auto_exposure.setChecked(False)
        dialog.exposure.setValue(-7)
        dialog.backlight.setChecked(True)
        dialog.equalize.setChecked(True)
        result = dialog.result_cfg()

        self.assertTrue(dialog.focus.isEnabled())
        self.assertTrue(dialog.exposure.isEnabled())
        self.assertFalse(result["camera_autofocus"])
        self.assertEqual(result["camera_focus"], 190)
        self.assertFalse(result["camera_auto_exposure"])
        self.assertEqual(result["camera_exposure"], -7)
        self.assertEqual(result["camera_backlight"], 1)
        self.assertTrue(result["detection_equalize"])

    @patch("ui.list_visible_windows", return_value=[])
    def test_camera_control_changes_emit_live_preview_values(self, _windows):
        cfg = dict(core.DEFAULTS)
        dialog = ui.SettingsDialog(cfg)
        self.assertTrue(hasattr(dialog, "camera_preview"), "摄像头实时预览信号尚未实现")
        emitted = []
        dialog.camera_preview.connect(emitted.append)

        dialog.autofocus.setChecked(False)
        dialog.focus.setValue(201)
        dialog.auto_exposure.setChecked(False)
        dialog.exposure.setValue(-8)

        self.assertTrue(emitted)
        self.assertEqual(emitted[-1]["camera_focus"], 201)
        self.assertEqual(emitted[-1]["camera_exposure"], -8)


if __name__ == "__main__":
    unittest.main()

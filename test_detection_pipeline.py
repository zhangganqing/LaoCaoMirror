# -*- coding: utf-8 -*-
"""检测热路径回归测试：多 ROI 只推理一次，并统一使用全局坐标。"""
import unittest

import numpy as np

import main as core


class RecordingDetector:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, dtype=int)
        self.calls = []

    def detect(self, frame, imgsz=960):
        self.calls.append((frame.shape, imgsz))
        return self.boxes


class DetectionPipelineTests(unittest.TestCase):
    def test_multiple_rois_share_one_inference_and_filter_gap(self):
        self.assertTrue(hasattr(core, "detect_faces"), "main.detect_faces 尚未实现")
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        rois = [[0.0, 0.1, 0.25, 0.7], [0.75, 0.2, 0.2, 0.6]]
        # 合并裁剪区为 x=0..190, y=10..80；框坐标是相对这个裁剪区的。
        detector = RecordingDetector([
            [10, 10, 30, 30],     # 全局 (10,20)-(30,40)，在左 ROI
            [80, 10, 100, 30],    # 全局 (80,20)-(100,40)，在两 ROI 之间，应过滤
            [155, 20, 180, 50],   # 全局 (155,30)-(180,60)，在右 ROI
        ])

        boxes = core.detect_faces(detector, frame, rois, imgsz=640)

        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(detector.calls[0], ((70, 190, 3), 640))
        self.assertEqual(boxes, [(10, 20, 30, 40), (155, 30, 180, 60)])

    def test_no_roi_runs_once_and_keeps_frame_coordinates(self):
        self.assertTrue(hasattr(core, "detect_faces"), "main.detect_faces 尚未实现")
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        detector = RecordingDetector([[4, 5, 20, 30]])

        boxes = core.detect_faces(detector, frame, None, imgsz=960)

        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(detector.calls[0], ((60, 80, 3), 960))
        self.assertEqual(boxes, [(4, 5, 20, 30)])


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""检测热路径回归测试：多 ROI 只推理一次，并统一使用全局坐标。"""
import unittest
from pathlib import Path

import cv2
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
    def test_onnx_detector_runs_real_reference_photo_on_cpu(self):
        root = Path(__file__).parent
        detector = core.FaceDetector(
            str(root / "models" / "yolov8n-face.onnx"), backend="cpu")
        image = cv2.imread(str(Path("laocao") / "ref_front_talking.png"))

        boxes = detector.detect(image, imgsz=640)

        self.assertEqual(detector.provider, "CPUExecutionProvider")
        self.assertGreaterEqual(len(boxes), 1)

    def test_onnx_letterbox_keeps_aspect_ratio_and_reports_padding(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        blob, scale, pad = core.prepare_yolo_input(frame, 640)

        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertAlmostEqual(scale, 3.2)
        self.assertEqual(pad, (0, 160))

    def test_onnx_yolo_output_restores_frame_coordinates_and_applies_nms(self):
        # 原图 200x100 letterbox 到 640x640: scale=3.2, 上下各补 160。
        # 前两个框高度重叠，应由 NMS 只留下 0.9 分框；第三个低于阈值。
        output = np.array([[
            [320.0, 323.2, 64.0],
            [320.0, 320.0, 192.0],
            [320.0, 320.0, 32.0],
            [192.0, 185.6, 32.0],
            [0.90, 0.80, 0.10],
        ]], dtype=np.float32)

        boxes = core.decode_yolo_face_output(
            output, frame_shape=(100, 200), scale=3.2, pad=(0, 160),
            confidence_threshold=0.25, iou_threshold=0.45)

        self.assertEqual(boxes, [(50, 20, 150, 80)])

    def test_overlapping_boxes_from_one_face_are_merged(self):
        """同一张脸的多锚框不能重复进入识别。"""
        boxes = core.merge_fragmented_face_boxes([
            (10, 30, 55, 95),
            (16, 42, 56, 95),
        ])

        self.assertEqual(boxes, [(10, 30, 56, 95)])

    def test_small_adjacent_ear_fragment_is_merged_into_main_face(self):
        """侧脸被拆成主脸和耳侧小框时，合并为完整头部框。"""
        boxes = core.merge_fragmented_face_boxes([
            (10, 30, 55, 95),
            (57, 24, 91, 65),
            (16, 42, 56, 95),
        ])

        self.assertEqual(boxes, [(10, 24, 91, 95)])

    def test_two_similar_sized_adjacent_faces_remain_separate(self):
        """肩并肩的两个真人不能被误合并。"""
        boxes = core.merge_fragmented_face_boxes([
            (0, 10, 40, 60),
            (42, 8, 82, 60),
        ])

        self.assertEqual(boxes, [(0, 10, 40, 60), (42, 8, 82, 60)])

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

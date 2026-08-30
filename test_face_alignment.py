# -*- coding: utf-8 -*-
"""SCRFD 五点对齐回归：同一个人的大姿态照片不应再被直接拉伸拉散。"""
import unittest

import cv2

import main as core


class FaceAlignmentTests(unittest.TestCase):
    def test_landmark_alignment_makes_gallery_views_consistent(self):
        self.assertTrue(hasattr(core, "FaceAligner"), "main.FaceAligner 尚未实现")
        aligner = core.FaceAligner("models/scrfd_2.5g.onnx")
        recognizer = core.FaceRecognizer(
            "models/w600k_mbf.onnx", "__missing_gallery__", aligner=aligner)
        images = [
            cv2.imread("laocao/0f18b067c807a1bbbcdb0e2450cfd880.jpg"),
            cv2.imread("laocao/30f42f7cccac4f2003a1071e198de4da.jpg"),
        ]

        embeddings = [recognizer._embed(image, align_size=640) for image in images]

        self.assertTrue(all(e is not None for e in embeddings))
        self.assertGreater(float(embeddings[0] @ embeddings[1]), 0.65)

    def test_expanded_face_box_adds_context_and_clamps_to_frame(self):
        self.assertTrue(hasattr(core, "expand_face_box"), "main.expand_face_box 尚未实现")

        self.assertEqual(core.expand_face_box((10, 20, 30, 50), (100, 80), 0.25),
                         (5, 12, 35, 58))
        self.assertEqual(core.expand_face_box((0, 0, 20, 20), (100, 80), 0.5),
                         (0, 0, 30, 30))


if __name__ == "__main__":
    unittest.main()

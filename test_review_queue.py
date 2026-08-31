# -*- coding: utf-8 -*-
"""持久化审核队列：累计、最佳照片、跨重启与坏记录隔离。"""
import json
import os
import tempfile
import unittest
from datetime import datetime

import cv2
import numpy as np

import main as core


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gallery_dir = self.temp.name

    def tearDown(self):
        self.temp.cleanup()

    def test_multiple_events_accumulate_and_survive_restart(self):
        self.assertTrue(hasattr(core, "ReviewQueue"), "ReviewQueue 尚未实现")
        queue = core.ReviewQueue(self.gallery_dir)
        first = queue.start_event(
            np.full((20, 20, 3), 60, np.uint8), 0.51,
            now=datetime(2026, 8, 31, 9, 0, 0))
        second = queue.start_event(
            np.full((20, 20, 3), 180, np.uint8), 0.72,
            now=datetime(2026, 8, 31, 9, 1, 0))

        items = core.ReviewQueue(self.gallery_dir).list_items()

        self.assertNotEqual(first, second)
        self.assertEqual({item["id"] for item in items}, {first, second})
        self.assertEqual({round(item["score"], 2) for item in items}, {0.51, 0.72})
        self.assertTrue(all(os.path.exists(item["image_path"]) for item in items))

    def test_each_event_keeps_only_its_highest_score_photo(self):
        self.assertTrue(hasattr(core, "ReviewQueue"), "ReviewQueue 尚未实现")
        queue = core.ReviewQueue(self.gallery_dir)
        low = np.full((20, 20, 3), 30, np.uint8)
        high = np.full((20, 20, 3), 220, np.uint8)
        record_id = queue.start_event(low, 0.55)

        self.assertFalse(queue.consider(record_id, high, 0.50))
        self.assertTrue(queue.consider(record_id, high, 0.80))
        item = queue.list_items()[0]
        saved = cv2.imread(item["image_path"])

        self.assertAlmostEqual(item["score"], 0.80)
        self.assertGreater(float(saved.mean()), 200)
        self.assertEqual(len(list(queue.review_dir.glob("*.jpg"))), 1)

    def test_delete_removes_image_and_metadata(self):
        self.assertTrue(hasattr(core, "ReviewQueue"), "ReviewQueue 尚未实现")
        queue = core.ReviewQueue(self.gallery_dir)
        record_id = queue.start_event(np.full((20, 20, 3), 90, np.uint8), 0.60)
        item = queue.list_items()[0]

        deleted = queue.delete([record_id])

        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.exists(item["image_path"]))
        self.assertEqual(queue.list_items(), [])

    def test_corrupt_metadata_does_not_hide_valid_records(self):
        self.assertTrue(hasattr(core, "ReviewQueue"), "ReviewQueue 尚未实现")
        queue = core.ReviewQueue(self.gallery_dir)
        valid_id = queue.start_event(np.full((20, 20, 3), 120, np.uint8), 0.66)
        queue.review_dir.mkdir(parents=True, exist_ok=True)
        (queue.review_dir / "broken.json").write_text("{not json", encoding="utf-8")
        cv2.imwrite(str(queue.review_dir / "broken.jpg"), np.zeros((10, 10, 3), np.uint8))

        items = core.ReviewQueue(self.gallery_dir).list_items()

        self.assertEqual([item["id"] for item in items], [valid_id])


if __name__ == "__main__":
    unittest.main()

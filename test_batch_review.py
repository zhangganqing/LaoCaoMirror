# -*- coding: utf-8 -*-
"""批量审核：确认学习、重复跳过、失败保留与误报删除。"""
import tempfile
import unittest

import numpy as np

import main as core


class _ResultRecognizer:
    def learn_face(self, image, **_kwargs):
        value = int(round(float(image.mean()) / 10.0) * 10)
        if value == 10:
            return {"status": "learned", "path": "learned_test.jpg"}
        if value == 20:
            return {"status": "duplicate", "similarity": 0.96}
        return {"status": "invalid", "message": "五点对齐失败"}


class BatchReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.queue = core.ReviewQueue(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _add(self, value, score=0.6):
        return self.queue.start_event(np.full((24, 24, 3), value, np.uint8), score)

    def test_confirm_many_counts_results_and_keeps_invalid_record(self):
        self.assertTrue(hasattr(self.queue, "confirm_many"), "批量确认尚未实现")
        learned_id = self._add(10)
        duplicate_id = self._add(20)
        invalid_id = self._add(30)

        result = self.queue.confirm_many(
            [learned_id, duplicate_id, invalid_id], _ResultRecognizer(),
            max_learned=30, duplicate_threshold=0.93)

        self.assertEqual(result["learned"], 1)
        self.assertEqual(result["duplicate"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["error"], 0)
        self.assertEqual([item["id"] for item in self.queue.list_items()], [invalid_id])

    def test_reject_many_deletes_only_selected_records(self):
        self.assertTrue(hasattr(self.queue, "reject_many"), "批量误报删除尚未实现")
        first = self._add(50)
        second = self._add(60)
        keep = self._add(70)

        result = self.queue.reject_many([first, second])

        self.assertEqual(result["deleted"], 2)
        self.assertEqual([item["id"] for item in self.queue.list_items()], [keep])


if __name__ == "__main__":
    unittest.main()

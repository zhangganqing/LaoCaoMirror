# -*- coding: utf-8 -*-
"""人工确认学习：待审照片、去重、热更新与自动样本上限。"""
import os
import tempfile
import threading
import unittest
from datetime import datetime
from unittest.mock import patch

import cv2
import numpy as np

import main as core


def _write_image(path, value):
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(path, image)


def _write_checkerboard(path, low=40, high=220):
    pattern = (np.indices((32, 32)).sum(axis=0) % 2).astype(np.uint8)
    image = np.repeat(
        np.where(pattern[..., None] == 0, low, high), 3, axis=2).astype(np.uint8)
    assert cv2.imwrite(path, image)


def _recognizer_without_models(gallery_dir, embeddings, candidate_embedding):
    recognizer = core.FaceRecognizer.__new__(core.FaceRecognizer)
    recognizer.gallery_dir = gallery_dir
    recognizer.gallery = {}
    recognizer.gallery_embeddings = {
        os.path.basename(gallery_dir): {
            filename: np.asarray(vector, dtype=np.float32)
            for filename, vector in embeddings.items()
        }
    }
    recognizer._embed = lambda _image: np.asarray(candidate_embedding, dtype=np.float32)
    recognizer._refresh_gallery_mean(os.path.basename(gallery_dir))
    return recognizer


class FaceGalleryLearningTests(unittest.TestCase):
    def test_learned_outliers_cannot_drag_reliable_identity_center(self):
        """大量模糊新增样本不能压过用户提供的可靠原始照片。"""
        with tempfile.TemporaryDirectory() as gallery_dir:
            reference_a = np.array([1.0, 0.0], dtype=np.float32)
            reference_b = np.array([0.98, 0.20], dtype=np.float32)
            reference_b /= np.linalg.norm(reference_b)
            good_learned = np.array([0.8, 0.6], dtype=np.float32)
            embeddings = {
                "manual_a.jpg": reference_a,
                "manual_b.jpg": reference_b,
                "learned_good.jpg": good_learned,
            }
            embeddings.update({
                f"learned_outlier_{index}.jpg": np.array([0.0, 1.0], dtype=np.float32)
                for index in range(8)
            })
            recognizer = _recognizer_without_models(
                gallery_dir, embeddings, [1.0, 0.0])
            reference_center = reference_a + reference_b
            reference_center /= np.linalg.norm(reference_center)
            active_center = recognizer.gallery[os.path.basename(gallery_dir)]

            self.assertGreater(float(reference_center @ active_center), 0.97)
            self.assertGreater(
                float(good_learned @ active_center),
                float(good_learned @ reference_center),
                "一致的新增样本仍应产生受限的自适应效果")

    def test_confirm_promotes_photo_and_updates_gallery_without_restart(self):
        self.assertTrue(hasattr(core.FaceRecognizer, "learn_face"), "图库热学习尚未实现")
        with tempfile.TemporaryDirectory() as gallery_dir:
            _write_image(os.path.join(gallery_dir, "manual.jpg"), 80)
            recognizer = _recognizer_without_models(
                gallery_dir, {"manual.jpg": [1.0, 0.0]}, [0.8, 0.6])
            candidate = np.full((32, 32, 3), 160, dtype=np.uint8)

            result = recognizer.learn_face(
                candidate, duplicate_threshold=0.99,
                now=datetime(2026, 8, 30, 19, 1, 2, 345000))

            self.assertEqual(result["status"], "learned")
            self.assertTrue(os.path.basename(result["path"]).startswith("learned_20260830_190102_345"))
            self.assertTrue(os.path.exists(result["path"]))
            name = os.path.basename(gallery_dir)
            self.assertIn(os.path.basename(result["path"]), recognizer.gallery_embeddings[name])
            # 可靠原始照片占 80%，一致的新增样本最多影响 20%。
            expected = np.array([0.96, 0.12], dtype=np.float32)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(recognizer.gallery[name], expected, atol=1e-6)

    def test_duplicate_confirmation_does_not_add_another_photo(self):
        self.assertTrue(hasattr(core.FaceRecognizer, "learn_face"), "图库去重尚未实现")
        with tempfile.TemporaryDirectory() as gallery_dir:
            _write_image(os.path.join(gallery_dir, "manual.jpg"), 80)
            recognizer = _recognizer_without_models(
                gallery_dir, {"manual.jpg": [1.0, 0.0]}, [0.999, 0.01])

            result = recognizer.learn_face(
                np.full((32, 32, 3), 160, dtype=np.uint8),
                duplicate_threshold=0.95)

            self.assertEqual(result["status"], "duplicate")
            self.assertEqual(
                [name for name in os.listdir(gallery_dir) if name.startswith("learned_")], [])

    def test_cap_archives_lowest_quality_and_keeps_old_high_quality_photo(self):
        with tempfile.TemporaryDirectory() as gallery_dir:
            paths = {
                "manual.jpg": os.path.join(gallery_dir, "manual.jpg"),
                "learned_20260901_080000_000.jpg": os.path.join(
                    gallery_dir, "learned_20260901_080000_000.jpg"),
                "learned_20260904_080000_000.jpg": os.path.join(
                    gallery_dir, "learned_20260904_080000_000.jpg"),
            }
            _write_image(paths["manual.jpg"], 100)
            _write_checkerboard(paths["learned_20260901_080000_000.jpg"])
            _write_image(paths["learned_20260904_080000_000.jpg"], 127)
            os.utime(paths["learned_20260901_080000_000.jpg"], (10, 10))
            os.utime(paths["learned_20260904_080000_000.jpg"], (20, 20))
            recognizer = _recognizer_without_models(
                gallery_dir,
                {"manual.jpg": [1.0, 0.0],
                 "learned_20260901_080000_000.jpg": [0.98, 0.20],
                 "learned_20260904_080000_000.jpg": [0.30, 0.954]},
                [0.90, 0.436])

            result = recognizer.learn_face(
                np.repeat(
                    np.where((np.indices((32, 32)).sum(axis=0) % 2)[..., None] == 0,
                             50, 210), 3, axis=2).astype(np.uint8),
                max_learned=2, duplicate_threshold=0.999,
                now=datetime(2026, 9, 5, 9, 30, 0))

            self.assertEqual(result["status"], "learned")
            self.assertTrue(os.path.exists(paths["manual.jpg"]))
            self.assertTrue(os.path.exists(
                paths["learned_20260901_080000_000.jpg"]))
            self.assertFalse(os.path.exists(
                paths["learned_20260904_080000_000.jpg"]))
            archived = os.path.join(
                gallery_dir, "archive", "2026-09-04",
                "learned_20260904_080000_000.jpg")
            self.assertTrue(os.path.exists(archived))
            self.assertIn(archived, result["archived"])
            learned = [name for name in os.listdir(gallery_dir) if name.startswith("learned_")]
            self.assertEqual(len(learned), 2)

    def test_new_low_quality_photo_is_archived_without_polluting_active_gallery(self):
        with tempfile.TemporaryDirectory() as gallery_dir:
            filenames = [
                "learned_20260903_080000_000.jpg",
                "learned_20260904_080000_000.jpg",
            ]
            _write_image(os.path.join(gallery_dir, "manual.jpg"), 100)
            for filename in filenames:
                _write_checkerboard(os.path.join(gallery_dir, filename))
            recognizer = _recognizer_without_models(
                gallery_dir,
                {"manual.jpg": [1.0, 0.0],
                 filenames[0]: [0.99, 0.141],
                 filenames[1]: [0.95, 0.312]},
                [0.0, 1.0])

            result = recognizer.learn_face(
                np.full((32, 32, 3), 127, dtype=np.uint8),
                max_learned=2, duplicate_threshold=0.999,
                now=datetime(2026, 9, 5, 10, 0, 0))

            self.assertEqual(result["status"], "archived")
            self.assertEqual(
                sorted(name for name in os.listdir(gallery_dir)
                       if name.startswith("learned_")), sorted(filenames))
            archived = os.path.join(
                gallery_dir, "archive", "2026-09-05",
                "learned_20260905_100000_000.jpg")
            self.assertTrue(os.path.exists(archived))


class _LoopCapture:
    def __init__(self, restore_event):
        self.restore_event = restore_event
        self.state = None
        self.seen_defend = False
        self.events = []

    def on_event(self, text):
        self.events.append(text)

    def on_state(self, state):
        self.state = state
        if state == "DEFEND" and not self.seen_defend:
            self.seen_defend = True
            self.restore_event.set()

    def on_frame(self, _frame):
        return not (self.seen_defend and self.state == "PATROL")


class _LoopCaptureSource:
    def __init__(self):
        self.frame = np.full((100, 100, 3), 120, dtype=np.uint8)

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class _LoopDetector:
    def detect(self, _frame, imgsz=960):
        return np.array([[25, 20, 75, 80]])


class _SmallFaceDetector:
    def detect(self, _frame, imgsz=960):
        return np.array([[45, 45, 55, 55]])


class _SequenceDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, _frame, imgsz=960):
        if next(self.detections):
            return np.array([[25, 20, 75, 80]])
        return np.empty((0, 4), dtype=np.float32)


class _FrameLimitCapture:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.frames = 0
        self.events = []

    def on_event(self, text):
        self.events.append(text)

    def on_state(self, _state):
        pass

    def on_frame(self, _frame):
        self.frames += 1
        return self.frames < self.frame_count


class _LoopRecognizer:
    def __init__(self, learning_status="learned"):
        self.learn_calls = 0
        self.learning_status = learning_status

    def identify(self, _face):
        return "laocao", 0.80

    def learn_face(self, _face, **_kwargs):
        self.learn_calls += 1
        return {"status": self.learning_status, "path": "learned_test.jpg"}


class _SequenceRecognizer(_LoopRecognizer):
    def __init__(self, scores):
        super().__init__()
        self.scores = iter(scores)

    def identify(self, _face):
        return "laocao", next(self.scores)


class ReviewQueuePipelineTests(unittest.TestCase):
    def _cfg(self, gallery_dir):
        cfg = dict(core.DEFAULTS)
        cfg.update({"gallery_dir": gallery_dir, "fps": 1000, "confirm_frames": 1,
                    "roi": None, "mirror": False, "guard_window_keyword": ""})
        return cfg

    def _run_once(self, gallery_dir):
        restore_event = threading.Event()
        cb = _LoopCapture(restore_event)
        recognizer = _LoopRecognizer()
        with patch.object(core, "beep"), patch.object(core, "restore_screen"):
            core.run_loop(
                self._cfg(gallery_dir), _LoopCaptureSource(), _LoopDetector(),
                recognizer, None, cb, restore_event=restore_event)
        return recognizer, cb.events

    def test_restore_does_not_learn_or_delete_queued_photo(self):
        with tempfile.TemporaryDirectory() as gallery_dir:
            recognizer, events = self._run_once(gallery_dir)
            items = core.ReviewQueue(gallery_dir).list_items()

            self.assertEqual(recognizer.learn_calls, 0)
            self.assertEqual(len(items), 1)
            self.assertTrue(any("[记录]" in text and "待审核" in text for text in events))
            self.assertTrue(any("[恢复]" in text for text in events))

    def test_two_defense_events_accumulate_two_review_records(self):
        with tempfile.TemporaryDirectory() as gallery_dir:
            self._run_once(gallery_dir)
            self._run_once(gallery_dir)

            self.assertEqual(len(core.ReviewQueue(gallery_dir).list_items()), 2)

    def test_reappearance_creates_new_record_without_manual_restore(self):
        """防御界面未恢复时，离开再出现仍应是新的审核事件。"""
        with tempfile.TemporaryDirectory() as gallery_dir:
            detections = [True, True, False, True, True]
            cb = _FrameLimitCapture(len(detections))
            cfg = self._cfg(gallery_dir)
            cfg["review_rearm_seconds"] = 0
            with patch.object(core, "beep"), patch.object(core, "toggle_desktop"):
                core.run_loop(
                    cfg, _LoopCaptureSource(), _SequenceDetector(detections),
                    _LoopRecognizer(), None, cb)

            items = core.ReviewQueue(gallery_dir).list_items()
            self.assertEqual(len(items), 2)
            self.assertEqual(sum("[记录]" in event for event in cb.events), 2)

    def test_small_high_score_face_cannot_trigger_defense(self):
        """低于最小脸占比的模糊目标即使偶然高分也不能切屏。"""
        with tempfile.TemporaryDirectory() as gallery_dir:
            cb = _FrameLimitCapture(2)
            cfg = self._cfg(gallery_dir)
            cfg["min_face_ratio"] = 0.20
            with patch.object(core, "beep"), patch.object(core, "toggle_desktop"):
                core.run_loop(
                    cfg, _LoopCaptureSource(), _SmallFaceDetector(),
                    _LoopRecognizer(), None, cb)

            self.assertEqual(core.ReviewQueue(gallery_dir).list_items(), [])
            self.assertFalse(any("[防御]" in event for event in cb.events))

    def test_high_score_confirmation_uses_fresh_recognition_each_frame(self):
        """一次偶然高分不能靠帧间缓存伪造连续命中。"""
        with tempfile.TemporaryDirectory() as gallery_dir:
            cb = _FrameLimitCapture(2)
            cfg = self._cfg(gallery_dir)
            cfg["confirm_frames"] = 2
            with patch.object(core, "beep"), patch.object(core, "toggle_desktop"):
                core.run_loop(
                    cfg, _LoopCaptureSource(), _LoopDetector(),
                    _SequenceRecognizer([0.8, 0.1]), None, cb)

            self.assertEqual(core.ReviewQueue(gallery_dir).list_items(), [])
            self.assertFalse(any("[防御]" in event for event in cb.events))

    def test_low_score_cache_expires_as_target_approaches(self):
        """远处首帧低分不能永久缓存，走近后必须重新识别。"""
        with tempfile.TemporaryDirectory() as gallery_dir:
            cb = _FrameLimitCapture(3)
            cfg = self._cfg(gallery_dir)
            cfg["confirm_frames"] = 2
            cfg["recognition_cache_frames"] = 1
            with patch.object(core, "beep"), patch.object(core, "toggle_desktop"):
                core.run_loop(
                    cfg, _LoopCaptureSource(), _LoopDetector(),
                    _SequenceRecognizer([0.1, 0.8, 0.8]), None, cb)

            self.assertEqual(len(core.ReviewQueue(gallery_dir).list_items()), 1)
            self.assertTrue(any("[防御]" in event for event in cb.events))


if __name__ == "__main__":
    unittest.main()

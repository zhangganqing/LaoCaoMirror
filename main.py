# -*- coding: utf-8 -*-
"""
全自动防老曹后视镜 v0.1 (MVP)
摄像头朝后 -> YOLO 人脸检测 -> ArcFace 识别是否老曹
  命中老曹  -> 切屏到护身窗口 (win32 置顶)
  有人但认不出 -> 提示音 + 黄框预警
  老曹离开 -> 冷却后自动切回

用法:
  python main.py                 # 默认摄像头 0
  python main.py --source 视频或图片路径   # 用文件模拟摄像头(测试)
  python main.py --test-switch   # 自测切屏/恢复后退出
"""
import argparse
import ctypes
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# PyInstaller 打包(--windowed)后 stdout 为 None, print 会崩; 资源锚定到 exe 所在目录
if getattr(sys, "frozen", False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    os.chdir(os.path.dirname(sys.executable))

# 必须在 numpy/cv2/torch 之前设置: 小推理下的线程自旋空转是 CPU 占用的最大来源
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2
import numpy as np
import onnxruntime as ort
import winsound
import torch
from ultralytics import YOLO

torch.set_num_threads(2)   # 不限制时 torch 会把全部核心拉去并行小推理, 风扇狂转

# ---------------- 配置 ----------------
DEFAULTS = {
    "camera_index": 0,
    "camera_resolution": [1920, 1080],  # 摄像头采集分辨率(设置里可选), 实际以摄像头支持为准
    "camera_autofocus": True,      # 关闭后可把焦点手动锁在后方通道
    "camera_focus": 160,           # 驱动常见范围 0~255；不支持时安全忽略
    "camera_auto_exposure": True,
    "camera_exposure": -6,         # Windows 摄像头常见范围 -13~-1
    "camera_backlight": 0,         # 逆光补偿，常见 0/1
    "detection_equalize": False,   # 仅增强检测副本，不改变原始预览
    "mirror": True,
    "fps": 10,                # 主循环帧率上限(检测+预览)。老曹是慢慢摸上来的, 降到 5 更省电(3帧确认≈0.6s)
    "max_frame_size": 1920,   # 进入检测的分辨率上限(长边); 摄像头请求 1080p, 一般不触发缩放
    "detect_imgsz": 960,      # YOLO 推理尺寸, 越大远处小脸越容易被检出, CPU 也越高
    "roi": None,              # 检测区域 [[x,y,w,h],...](相对画面比例)。null=全画面; 框外的脸完全忽略
    "min_face_ratio": 0.05,   # 脸框高/画面高 低于此值视为太远, 不提醒
    "alert_threshold": 0.35,  # 疑似老曹阈值: 超过则响提醒音(不切屏)
    "id_threshold": 0.5,      # 切屏阈值: 超过则确认是老曹, 立即切屏
    "confirm_frames": 2,      # 连续命中 N 帧才切屏
    "cooldown_seconds": 25,   # 老曹离开后保持防御的时长
    "review_rearm_seconds": 1.0,  # 连续未检出多久后，再出现时新建审核记录
    "beep_alert": True,       # 陌生人(认不出)提醒音
    "guard_window_keyword": "GuardWindow",  # 护身窗口标题关键字
    "gallery_dir": "laocao",  # 老曹照片文件夹(目录名=身份名)
    "max_learned_photos": 30,   # 自动确认学习的照片上限(手工照片不计入且永不自动删除)
    "learned_duplicate_threshold": 0.93,  # 与已有照片过于相似时不重复入库
    "models_dir": "models",
}


def load_config(path="config.json"):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    roi = cfg.get("roi")
    if roi and isinstance(roi[0], (int, float)):   # 兼容旧单框 [x,y,w,h] → 统一为框列表
        cfg["roi"] = [roi]
    elif not roi:
        cfg["roi"] = None
    # 其余情况: 已是框列表 [[x,y,w,h],...], 保持
    return cfg


def save_config(cfg, path="config.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------- 检测 + 识别 ----------------
# ArcFace 标准五点(112x112), 本模型无关键点输出, 用等比缩放对齐
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
FACE_CACHE_VERSION = 3


class FaceDetector:
    def __init__(self, weights):
        self.model = YOLO(weights)
        self.device = "cpu"

    def detect(self, frame, imgsz=960):
        """返回 [(x1,y1,x2,y2), ...]"""
        res = self.model(frame, verbose=False, imgsz=imgsz, device=self.device)[0]
        return res.boxes.xyxy.cpu().numpy().astype(int)


class FaceAligner:
    """SCRFD 检出五点关键点，并仿射对齐到 ArcFace 标准 112x112 人脸。"""

    def __init__(self, onnx_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.log_severity_level = 3  # 动态 320 输入与模型静态输出注记不一致的警告不影响结果
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self._anchor_cache = {}

    def _anchors(self, input_size, stride):
        key = (input_size, stride)
        if key not in self._anchor_cache:
            yy, xx = np.mgrid[:input_size // stride, :input_size // stride]
            centers = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self._anchor_cache[key] = np.repeat(centers, 2, axis=0)
        return self._anchor_cache[key]

    def align(self, image, input_size=320, threshold=0.35):
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        scale = min(input_size / w, input_size / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (input_size, input_size),
            (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.sess.run(None, {self.input_name: blob})

        candidates = []
        for i, stride in enumerate((8, 16, 32)):
            scores = outputs[i].reshape(-1)
            boxes = outputs[i + 3].reshape(-1, 4) * stride
            keypoints = outputs[i + 6].reshape(-1, 10) * stride
            centers = self._anchors(input_size, stride)
            for idx in np.where(scores >= threshold)[0]:
                x1 = centers[idx, 0] - boxes[idx, 0]
                y1 = centers[idx, 1] - boxes[idx, 1]
                x2 = centers[idx, 0] + boxes[idx, 2]
                y2 = centers[idx, 1] + boxes[idx, 3]
                points = np.array([
                    [centers[idx, 0] + keypoints[idx, j * 2],
                     centers[idx, 1] + keypoints[idx, j * 2 + 1]]
                    for j in range(5)
                ], dtype=np.float32)
                candidates.append(((x2 - x1) * (y2 - y1), points / scale))
        if not candidates:
            return None
        landmarks = max(candidates, key=lambda item: item[0])[1]
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARC_DST, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(image, matrix, (112, 112), borderValue=0)


class FaceRecognizer:
    def __init__(self, onnx_path, gallery_dir, detector=None, aligner=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2   # 112x112 小图, 2 线程足够
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.detector = detector   # 注册半身照时需先检测裁脸
        self.aligner = aligner
        self.gallery_dir = gallery_dir
        self.gallery = {}  # name -> 归一化平均特征
        self.gallery_embeddings = {}  # name -> {照片文件名: 归一化特征}
        self._load_gallery(gallery_dir)

    def _largest_face(self, img):
        if self.detector is None:
            return img
        boxes = self.detector.detect(img)
        if len(boxes) == 0:
            return None
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return img[y1:y2, x1:x2]

    def _load_gallery(self, gallery_dir):
        """单身份: gallery_dir 即身份文件夹(目录名=身份名), 内放 1~5 张照片。
        特征缓存到目录内 face_cache.json: 照片没变时启动直接读缓存, 不重算"""
        if not os.path.isdir(gallery_dir):
            return
        name = os.path.basename(os.path.normpath(gallery_dir))
        photos = [fn for fn in os.listdir(gallery_dir)
                  if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        # 指纹: 文件名 + [大小, 修改时间](用 list, json 读写后才能相等比较)
        fingerprint = {fn: [os.path.getsize(os.path.join(gallery_dir, fn)),
                            int(os.path.getmtime(os.path.join(gallery_dir, fn)))]
                       for fn in photos}
        cache_path = os.path.join(gallery_dir, "face_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if (cache.get("version") == FACE_CACHE_VERSION
                        and cache.get("name") == name
                        and cache.get("fingerprint") == fingerprint
                        and isinstance(cache.get("embeddings"), dict)):
                    vectors = {
                        fn: np.array(values, dtype=np.float32)
                        for fn, values in cache["embeddings"].items()
                    }
                    self.gallery_embeddings[name] = vectors
                    self._refresh_gallery_mean(name)
                    print(f"[人脸库] {name}: {len(photos)} 张照片 (读取缓存, 无需重算)")
                    return
            except Exception:
                pass  # 缓存损坏则走重算
        registered = []  # (文件名, 特征)
        for fn in photos:
            img = cv2.imread(os.path.join(gallery_dir, fn))
            if img is None:
                continue
            face = img if self.aligner is not None else self._largest_face(img)
            if face is None or face.size == 0:
                print(f"[人脸库] 跳过 {fn}: 未检出人脸(照片太模糊或侧脸太狠?)")
                continue
            embedding = self._embed(face, align_size=640)
            if embedding is None:
                print(f"[人脸库] 跳过 {fn}: 五点关键点对齐失败")
                continue
            registered.append((fn, embedding))
        if registered:
            self.gallery_embeddings[name] = dict(registered)
            self._refresh_gallery_mean(name)
            mean = self.gallery[name]
            worst = min(registered, key=lambda r: float(mean @ r[1]))
            print(f"[人脸库] {name}: {len(registered)} 张照片入库, "
                  f"最差一致度 {float(mean @ worst[1]):.2f} ({worst[0]})")
            try:
                self._write_gallery_cache(name)
            except Exception:
                pass

    def _refresh_gallery_mean(self, name):
        """用当前逐照片特征重算身份中心；确认学习后无需重启即可生效。"""
        vectors = list(self.gallery_embeddings.get(name, {}).values())
        if not vectors:
            self.gallery.pop(name, None)
            return
        mean = np.mean(vectors, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            self.gallery[name] = mean / norm

    def _write_gallery_cache(self, name):
        photos = [fn for fn in os.listdir(self.gallery_dir)
                  if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        fingerprint = {
            fn: [os.path.getsize(os.path.join(self.gallery_dir, fn)),
                 int(os.path.getmtime(os.path.join(self.gallery_dir, fn)))]
            for fn in photos
        }
        embeddings = {
            fn: vector.tolist()
            for fn, vector in self.gallery_embeddings.get(name, {}).items()
            if fn in fingerprint
        }
        cache_path = os.path.join(self.gallery_dir, "face_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"version": FACE_CACHE_VERSION, "name": name,
                       "fingerprint": fingerprint, "embeddings": embeddings,
                       "mean": self.gallery.get(name, np.array([])).tolist()},
                      f, ensure_ascii=False)

    def learn_face(self, face_img, max_learned=30, duplicate_threshold=0.93, now=None):
        """人工确认后把脸照加入图库，并立即更新内存特征。

        learned_ 前缀专门标记自动样本；达到上限时只删除其中最旧的，
        用户手工放入图库的照片永远不会被这里清理。
        """
        if face_img is None or face_img.size == 0:
            return {"status": "invalid", "message": "待学习照片为空"}
        embedding = self._embed(face_img)
        if embedding is None:
            return {"status": "invalid", "message": "照片中的五点关键点对齐失败"}
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            return {"status": "invalid", "message": "无法生成有效人脸特征"}
        embedding /= norm

        os.makedirs(self.gallery_dir, exist_ok=True)
        name = os.path.basename(os.path.normpath(self.gallery_dir))
        samples = self.gallery_embeddings.setdefault(name, {})
        similarities = [(fn, float(vector @ embedding)) for fn, vector in samples.items()]
        if similarities:
            closest_file, closest_score = max(similarities, key=lambda item: item[1])
            if closest_score >= duplicate_threshold:
                return {"status": "duplicate", "similarity": closest_score,
                        "duplicate_of": closest_file}

        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"learned_{stamp}.jpg"
        path = os.path.join(self.gallery_dir, filename)
        suffix = 1
        while os.path.exists(path):
            filename = f"learned_{stamp}_{suffix}.jpg"
            path = os.path.join(self.gallery_dir, filename)
            suffix += 1
        if not cv2.imwrite(path, face_img):
            return {"status": "error", "message": "学习照片写入磁盘失败"}

        samples[filename] = embedding
        learned = [fn for fn in samples if fn.startswith("learned_")]
        learned.sort(key=lambda fn: (os.path.getmtime(os.path.join(self.gallery_dir, fn)), fn))
        removed = []
        while len(learned) > max(0, int(max_learned)):
            oldest = learned.pop(0)
            oldest_path = os.path.join(self.gallery_dir, oldest)
            try:
                os.remove(oldest_path)
                removed.append(oldest)
                samples.pop(oldest, None)
            except OSError:
                break
        self._refresh_gallery_mean(name)
        try:
            self._write_gallery_cache(name)
        except Exception:
            pass
        return {"status": "learned", "path": path, "removed": removed,
                "sample_count": len(samples)}

    def _embed(self, face_img, align_size=320):
        if self.aligner is not None:
            a = self.aligner.align(face_img, input_size=align_size)
            if a is None:
                return None
        else:
            a = cv2.resize(face_img, (112, 112))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)
        a = (a - 127.5) / 127.5
        e = self.sess.run(None, {self.input_name: a.transpose(2, 0, 1)[None]})[0][0]
        return e / np.linalg.norm(e)

    def identify(self, face_img):
        """返回 (best_name, best_score)，库为空返回 (None, -1)"""
        if not self.gallery:
            return None, -1.0
        e = self._embed(face_img)
        if e is None:
            return None, -1.0
        name = max(self.gallery, key=lambda k: float(self.gallery[k] @ e))
        return name, float(self.gallery[name] @ e)


class ReviewQueue:
    """跨重启保存的识别审核队列；每个防御事件对应一张最佳脸照。"""

    def __init__(self, gallery_dir):
        self.review_dir = Path(gallery_dir) / "review_queue"

    def _paths(self, record_id):
        return (self.review_dir / f"{record_id}.jpg",
                self.review_dir / f"{record_id}.json")

    def _write_metadata(self, path, metadata):
        temp_path = path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

    def start_event(self, face_img, score, now=None):
        if face_img is None or face_img.size == 0:
            return None
        created = now or datetime.now()
        stamp = created.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        record_id = f"{stamp}_{uuid4().hex[:8]}"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        image_path, metadata_path = self._paths(record_id)
        if not cv2.imwrite(str(image_path), face_img):
            return None
        metadata = {
            "id": record_id,
            "created_at": created.isoformat(timespec="milliseconds"),
            "score": float(score),
            "image_name": image_path.name,
        }
        try:
            self._write_metadata(metadata_path, metadata)
        except Exception:
            image_path.unlink(missing_ok=True)
            return None
        return record_id

    def consider(self, record_id, face_img, score):
        if not record_id or face_img is None or face_img.size == 0:
            return False
        image_path, metadata_path = self._paths(record_id)
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if float(score) <= float(metadata["score"]):
                return False
            if not cv2.imwrite(str(image_path), face_img):
                return False
            metadata["score"] = float(score)
            self._write_metadata(metadata_path, metadata)
            return True
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def list_items(self):
        if not self.review_dir.is_dir():
            return []
        items = []
        for metadata_path in self.review_dir.glob("*.json"):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                record_id = str(metadata["id"])
                image_path, expected_metadata_path = self._paths(record_id)
                if expected_metadata_path != metadata_path or not image_path.is_file():
                    continue
                items.append({
                    "id": record_id,
                    "created_at": str(metadata["created_at"]),
                    "score": float(metadata["score"]),
                    "image_path": str(image_path),
                })
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def delete(self, record_ids):
        deleted = 0
        for record_id in dict.fromkeys(record_ids):
            image_path, metadata_path = self._paths(record_id)
            existed = image_path.exists() or metadata_path.exists()
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            if existed:
                deleted += 1
        return deleted

    def confirm_many(self, record_ids, recognizer,
                     max_learned=30, duplicate_threshold=0.93):
        """批量确认是目标身份；成功或重复项移出队列，失败项保留。"""
        result = {"learned": 0, "duplicate": 0, "invalid": 0, "error": 0}
        available = {item["id"]: item for item in self.list_items()}
        for record_id in dict.fromkeys(record_ids):
            item = available.get(record_id)
            if item is None:
                result["error"] += 1
                continue
            image = cv2.imread(item["image_path"])
            if image is None or image.size == 0:
                result["invalid"] += 1
                continue
            try:
                learned = recognizer.learn_face(
                    image, max_learned=max_learned,
                    duplicate_threshold=duplicate_threshold)
            except Exception:
                result["error"] += 1
                continue
            status = learned.get("status", "error")
            if status in ("learned", "duplicate"):
                result[status] += 1
                self.delete([record_id])
            elif status == "invalid":
                result["invalid"] += 1
            else:
                result["error"] += 1
        return result

    def reject_many(self, record_ids):
        return {"deleted": self.delete(record_ids)}


# ---------------- 切屏 (win32) ----------------
user32 = ctypes.windll.user32


def find_guard_window(keyword):
    """找标题含关键字且可见的顶层窗口; 空关键字 = 未启用切屏"""
    if not keyword:
        return None
    import win32con
    import win32gui
    found = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if keyword in title and "RearviewMirror" not in title:
            ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if not ex & win32con.WS_EX_TOOLWINDOW:
                found.append(hwnd)
    win32gui.EnumWindows(cb, None)
    return found[0] if found else None


def switch_to(hwnd):
    """把窗口带到前台。Windows 前台锁定会拒绝跨进程抢焦点, 组合三种手段:
    1) AttachThreadInput 借用前台线程的输入权限  2) ALT 键事件解锁
    3) SwitchToThisWindow 兜底"""
    import win32api
    import win32process
    cur = win32api.GetCurrentThreadId()
    try:
        fg = user32.GetForegroundWindow()
        if fg and fg != hwnd:
            fg_thread, _ = win32process.GetWindowThreadProcessId(fg)
            if fg_thread and fg_thread != cur:
                user32.AttachThreadInput(cur, fg_thread, True)
        user32.SetForegroundWindow(hwnd)
        if fg and fg != hwnd:
            fg_thread, _ = win32process.GetWindowThreadProcessId(fg)
            if fg_thread and fg_thread != cur:
                user32.AttachThreadInput(cur, fg_thread, False)
    except Exception:
        pass
    user32.keybd_event(0x12, 0, 0, 0)   # ALT down
    user32.keybd_event(0x12, 0, 2, 0)   # ALT up
    user32.SwitchToThisWindow(hwnd, True)


def switch_screen(guard_hwnd):
    """切到护身窗口并置顶, 返回切换前的前台窗口句柄"""
    orig = user32.GetForegroundWindow()
    import win32con
    user32.SetWindowPos(guard_hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    switch_to(guard_hwnd)
    return orig


def restore_screen(guard_hwnd, orig_hwnd):
    """取消护身窗口置顶, 切回原前台窗口"""
    import win32con
    if guard_hwnd:
        user32.SetWindowPos(guard_hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
    if orig_hwnd:
        switch_to(orig_hwnd)


def toggle_desktop():
    """发送 Win+D；用于未配置护身窗口时显示桌面，再次调用恢复窗口。"""
    user32.keybd_event(0x5B, 0, 0, 0)   # Left Windows down
    user32.keybd_event(0x44, 0, 0, 0)   # D down
    user32.keybd_event(0x44, 0, 2, 0)   # D up
    user32.keybd_event(0x5B, 0, 2, 0)   # Left Windows up


def beep(freq, ms):
    try:
        winsound.Beep(freq, ms)
    except Exception:
        pass


# ---------------- 引擎 ----------------
_model_cache = {}


def _iou(a, b):
    """两个框 (x1,y1,x2,y2) 的交并比"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def expand_face_box(box, frame_shape, ratio=0.25):
    """给紧贴人脸的检测框补少量上下文，供 SCRFD 稳定定位五点关键点。"""
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    pad_x = round((x2 - x1) * ratio)
    pad_y = round((y2 - y1) * ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))


def _pixel_rois(frame, rois):
    """把归一化 ROI 转为画面内的像素框 (x1,y1,x2,y2)。"""
    if not rois:
        return []
    h, w = frame.shape[:2]
    out = []
    for x, y, rw, rh in rois:
        x1 = max(0, min(w, int(x * w)))
        y1 = max(0, min(h, int(y * h)))
        x2 = max(0, min(w, int((x + rw) * w)))
        y2 = max(0, min(h, int((y + rh) * h)))
        if x2 - x1 > 10 and y2 - y1 > 10:
            out.append((x1, y1, x2, y2))
    return out


def detect_faces(detector, frame, rois, imgsz=960):
    """在所有 ROI 的最小包围区只推理一次，返回全画面坐标的人脸框。

    旧实现会给每个 ROI 各跑一次 YOLO；两个区域就几乎翻倍耗时。
    合并后再按框中心过滤，仍保证 ROI 外的人脸不会触发识别和告警。
    """
    pixel_rois = _pixel_rois(frame, rois)
    if pixel_rois:
        ox = min(r[0] for r in pixel_rois)
        oy = min(r[1] for r in pixel_rois)
        x2 = max(r[2] for r in pixel_rois)
        y2 = max(r[3] for r in pixel_rois)
        region = frame[oy:y2, ox:x2]
    else:
        ox = oy = 0
        region = frame

    boxes = []
    for x1, y1, x2, y2 in detector.detect(region, imgsz=imgsz):
        box = (int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy)
        if pixel_rois:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if not any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                       for rx1, ry1, rx2, ry2 in pixel_rois):
                continue
        boxes.append(box)
    return boxes


def build_models(cfg):
    """模型进程内缓存: 引擎重启(换摄像头等)时免重复加载; 有 NVIDIA 显卡自动用 GPU"""
    key = (cfg["models_dir"], cfg["gallery_dir"])
    if key not in _model_cache:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        detector = FaceDetector(os.path.join(cfg["models_dir"], "yolov8n-face.pt"))
        detector.device = device
        aligner = FaceAligner(os.path.join(cfg["models_dir"], "scrfd_2.5g.onnx"))
        recognizer = FaceRecognizer(os.path.join(cfg["models_dir"], "w600k_mbf.onnx"),
                                    cfg["gallery_dir"], detector=detector, aligner=aligner)
        _model_cache[key] = (detector, recognizer)
        if device != "cpu":
            print(f"[算力] 推理设备: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("[算力] 推理设备: CPU (无可用 NVIDIA 显卡)")
    return _model_cache[key]


def report_startup(cfg, recognizer):
    """启动检查: 人脸库与护身窗口。返回护身窗口句柄(可能为 None)"""
    if not recognizer.gallery:
        print(f"[警告] 人脸库为空! 请把老曹照片放进 {cfg['gallery_dir']}/ 目录(2~5张)")
        print("        空库时只做提醒, 不会切屏。")
    guard_hwnd = find_guard_window(cfg["guard_window_keyword"])
    if guard_hwnd:
        import win32gui
        print(f"[护身窗口] {win32gui.GetWindowText(guard_hwnd)}")
    else:
        print("[护身窗口] 未找到(切屏将不可用, 仅提醒)")
    return guard_hwnd


class CliCallbacks:
    """命令行模式回调: cv2 预览窗 + 控制台日志, q 退出 / r 手动确认老曹离开"""

    def __init__(self, restore_event=None):
        self.restore_event = restore_event
        self.t_prev = time.time()
        self.fps = 0.0

    def on_event(self, text):
        print(text)

    def on_state(self, state):
        pass

    def on_frame(self, frame):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6)
        self.t_prev = now
        cv2.putText(frame, f"{self.fps:.0f}fps", (frame.shape[1] - 80, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("RearviewMirror", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") and self.restore_event is not None:
            self.restore_event.set()
        return key != ord("q")


def run_loop(cfg, cap, detector, recognizer, guard_hwnd, cb,
             stop_event=None, max_seconds=None, image_source=False,
             restore_event=None):
    """检测/识别/状态机/切屏主循环, 命令行与 GUI 共用。
    cb 需实现: on_event(str), on_state(str), on_frame(ndarray) 返回 False 表示请求退出
    restore_event 只负责恢复界面；真假判断统一留到持久化审核队列处理。
    cfg 为共享引用: 每帧重读 fps/阈值/roi 等, 设置可热更新"""
    interval = 1.0 / max(cfg.get("fps", 10), 1)
    if cfg.get("roi"):
        cb.on_event(f"[检测区] {len(cfg['roi'])} 个区域, 框外的脸完全忽略")
    else:
        cb.on_event("[检测区] 全画面检测")
    state = "PATROL"       # PATROL 警戒巡逻 / ALERT 警戒 / DEFEND 防御(等待手动确认)
    hit_streak = 0         # 连续命中老曹帧数
    last_seen = 0.0        # 最后一次检出老曹的时刻
    orig_hwnd = None
    last_reassert = 0.0
    last_alert = 0.0
    t_start = time.time()
    t_frame = time.time()
    guard_cur = guard_hwnd
    keyword_cur = cfg.get("guard_window_keyword") or ""
    last_guard_check = 0.0
    prev_faces = []   # 上一帧脸的身份缓存 [(x1,y1,x2,y2,name,score)]: IoU 匹配沿用, 免重复识别
    cur_faces = []
    review_queue = ReviewQueue(cfg["gallery_dir"])
    review_record_id = None
    best_hit_crop = None
    best_hit_score = -1.0
    desktop_fallback = False
    camera_signature = None

    def start_review_event():
        nonlocal review_record_id
        review_record_id = review_queue.start_event(best_hit_crop, best_hit_score)
        if review_record_id:
            cb.on_event(
                f"[记录] 已加入待审核队列 (相似度{best_hit_score:.2f}，"
                f"共{len(review_queue.list_items())}条)")
        else:
            cb.on_event("[记录失败] 无法保存本次识别照片，仍可恢复界面")

    def finish():
        if state == "DEFEND":   # 退出时别把屏幕留在防御态
            if desktop_fallback:
                toggle_desktop()
            else:
                restore_screen(guard_cur, orig_hwnd)
        cap.release()
        cv2.destroyAllWindows()

    while stop_event is None or not stop_event.is_set():
        # 帧率节流: 用 stop_event.wait 等待, 停止请求可立即唤醒(主线程 stop 不被拖住)
        elapsed = time.time() - t_frame
        remaining = interval - elapsed
        if stop_event is not None:
            if stop_event.wait(remaining if remaining > 0 else 0):
                break
        elif remaining > 0:
            time.sleep(remaining)
        t_frame = time.time()
        next_camera_signature = tuple(cfg.get(key) for key in (
            "camera_autofocus", "camera_focus", "camera_auto_exposure",
            "camera_exposure", "camera_backlight"))
        if next_camera_signature != camera_signature:
            camera_report = apply_camera_controls(cap, cfg)
            camera_signature = next_camera_signature
            if camera_report["unsupported"]:
                cb.on_event("[摄像头] 驱动不支持: " + ", ".join(camera_report["unsupported"]))
        if max_seconds and time.time() - t_start > max_seconds:
            cb.on_event(f"[结束] 达到限时 {max_seconds}s, 正常退出")
            finish()
            return
        ok, frame = cap.read()
        if not ok:
            # 图片源: 循环播放; 视频源: 结束退出
            if image_source:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cb.on_event("[结束] 视频源播放完毕")
            finish()
            return
        frame = preprocess(frame, cfg)
        detection_frame = prepare_detection_frame(frame, cfg)

        rois = cfg.get("roi")   # 每帧重读, 框选确认后下一帧即生效, 无需重启引擎
        if rois:
            H, W = frame.shape[:2]
            for r in rois:
                rx, ry = int(r[0] * W), int(r[1] * H)
                rw, rh = int(r[2] * W), int(r[3] * H)
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 220, 220), 1)
                cv2.putText(frame, "ROI", (rx + 4, ry + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
        laocao_here = False
        suspected_here = False
        someone_here = False
        h_frame = frame.shape[0]
        cur_faces = []

        boxes = detect_faces(detector, detection_frame, rois,
                             imgsz=cfg.get("detect_imgsz", 960))
        for x1, y1, x2, y2 in boxes:
            ratio = (y2 - y1) / h_frame
            name, score = None, -1.0
            face_crop = None
            if x2 - x1 > 8 and y2 - y1 > 8:
                # 紧框适合 YOLO，但五点对齐和待确认照片需要少量额头/下巴上下文。
                fx1, fy1, fx2, fy2 = expand_face_box((x1, y1, x2, y2), frame.shape)
                face_crop = detection_frame[fy1:fy2, fx1:fx2].copy()
                # 检测框和缓存框均为全画面坐标，所有 ROI 都能正确复用身份。
                for pf in prev_faces:
                    if _iou((x1, y1, x2, y2), pf[:4]) > 0.4:
                        name, score = pf[4], pf[5]
                        break
                if name is None:
                    name, score = recognizer.identify(face_crop)
            hit = (score >= cfg["id_threshold"])
            suspected = (not hit and score >= cfg.get("alert_threshold", 0.35))
            if hit:
                laocao_here = True
                if face_crop is not None and score > best_hit_score:
                    best_hit_crop = face_crop
                    best_hit_score = float(score)
                color, label = (0, 0, 255), f"{name}! {score:.2f}"   # 红: 确认老曹
            elif suspected and ratio >= cfg["min_face_ratio"]:
                suspected_here = True
                now = time.time()
                if now - last_alert > 5:
                    beep(2000, 120); beep(2000, 120); beep(2000, 120)   # 急促三连: 疑似老曹
                    last_alert = now
                    cb.on_event(f"[疑似] 可能是老曹 (相似度{score:.2f}, 达 {cfg['id_threshold']} 才切屏)")
                color, label = (0, 120, 255), f"?? {score:.2f}"      # 橙: 疑似
            elif score >= 0 and ratio >= cfg["min_face_ratio"]:
                someone_here = True
                color, label = (0, 200, 255), "???"       # 黄: 普通陌生人, 静默
            else:
                color, label = (0, 200, 255), "???"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cur_faces.append((x1, y1, x2, y2, name, score))
        prev_faces = cur_faces

        # ---- 状态机 ----
        now = time.time()
        kw = cfg.get("guard_window_keyword") or ""
        guard_stale = guard_cur is not None and not user32.IsWindow(guard_cur)
        if (kw != keyword_cur or guard_stale
                or (kw and guard_cur is None and now - last_guard_check > 5)):
            found = find_guard_window(kw) if kw else None
            last_guard_check = now
            if kw != keyword_cur or found != guard_cur:
                guard_cur = found
                keyword_cur = kw
                if guard_cur:
                    import win32gui
                    cb.on_event(f"[护身窗口] 已切换: {win32gui.GetWindowText(guard_cur)}")
                else:
                    cb.on_event("[护身窗口] 未找到(仅提醒), 每5秒自动重试")
            else:
                keyword_cur = kw
        if laocao_here:
            last_seen = now
            hit_streak += 1
            if state != "DEFEND" and hit_streak >= cfg["confirm_frames"]:
                state = "DEFEND"
                start_review_event()
                if guard_cur:
                    orig_hwnd = switch_screen(guard_cur)
                    desktop_fallback = False
                    beep(1000, 600)
                    cb.on_event(f"[防御] 老曹出现! 已切屏 (老曹离开后点\"恢复界面\"/按 R 切回)")
                else:
                    toggle_desktop()
                    desktop_fallback = True
                    beep(1000, 600)
                    cb.on_event("[防御] 老曹出现! 未配置护身窗口，已显示桌面")
            elif state == "DEFEND":
                if review_record_id is None:
                    # 恢复界面与记录分段解耦：离开后再次出现即新建记录。
                    start_review_event()
                elif best_hit_crop is not None:
                    review_queue.consider(
                        review_record_id, best_hit_crop, best_hit_score)
                if guard_cur and not desktop_fallback:
                    # 防御期间保活: 焦点被意外抢走时拉回护身窗口
                    if now - last_reassert > 1.0 and user32.GetForegroundWindow() != guard_cur:
                        switch_to(guard_cur)
                        last_reassert = now
        else:
            hit_streak = 0
            if state == "DEFEND":
                rearm_seconds = max(float(cfg.get("review_rearm_seconds", 1.0)), 0.0)
                if (review_record_id is not None and last_seen
                        and now - last_seen >= rearm_seconds):
                    review_record_id = None
                    best_hit_crop = None
                    best_hit_score = -1.0
            else:
                best_hit_crop = None
                best_hit_score = -1.0

        # ---- 手动恢复：记录继续留在审核队列，不在这里判断真假 ----
        restore_requested = restore_event is not None and restore_event.is_set()
        if restore_requested:
            restore_event.clear()
        if restore_requested:
            if state == "DEFEND":
                if desktop_fallback:
                    toggle_desktop()
                else:
                    restore_screen(guard_cur, orig_hwnd)
                state = "PATROL"
                desktop_fallback = False
                review_record_id = None
                best_hit_crop = None
                best_hit_score = -1.0
                cb.on_event("[恢复] 已切回界面，识别照片保留在审核记录中")
            else:
                cb.on_event("[恢复] 当前不在防御态, 无需恢复")
        if state != "DEFEND":            # 非防御态: 疑似/警戒/巡逻自动切换
            if suspected_here:
                state = "SUSPECT"
            else:
                state = "ALERT" if someone_here else "PATROL"

        # ---- 预览帧渲染(状态条) ----
        bar = {"PATROL": ("PATROL - SAFE", (80, 220, 80)),
               "ALERT": ("ALERT - SOMEONE", (0, 200, 255)),
               "SUSPECT": ("SUSPECT - BE READY!", (0, 120, 255)),
               "DEFEND": ("DEFEND - SWITCHED!", (0, 0, 255))}.get(
            state, (state.upper(), (200, 200, 200)))
        if state == "DEFEND" and int(now * 4) % 2 == 0:   # 防御态红框闪烁
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 12)
        cv2.putText(frame, bar[0], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar[1], 2)
        cb.on_state(state)
        if cb.on_frame(frame) is False:
            finish()
            return

    finish()


def run(cfg, source, max_seconds=None, select_roi=False):
    """命令行入口"""
    import threading
    detector, recognizer = build_models(cfg)
    guard_hwnd = report_startup(cfg, recognizer)
    cap = open_source(source, cfg)
    setup_roi(cap, cfg, select_roi)
    restore_event = threading.Event()
    run_loop(cfg, cap, detector, recognizer, guard_hwnd,
             CliCallbacks(restore_event), restore_event=restore_event,
             max_seconds=max_seconds,
             image_source=isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))


def open_source(source, cfg):
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return ImageSource(source)
    if source is None or str(source).isdigit():
        idx = cfg["camera_index"] if source is None else int(source)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[错误] 摄像头 {idx} 打不开: 被占用或不存在。可用 --source 指定视频文件测试")
            sys.exit(1)
        rw, rh = cfg.get("camera_resolution", [1920, 1080])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(rw))    # 请求指定分辨率, 摄像头会协商到支持档位
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(rh))
        return cap
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法读取 {source}")
        sys.exit(1)
    return cap


def apply_camera_controls(cap, cfg):
    """尽力应用摄像头驱动属性，返回实际值和不支持项；不支持时不抛异常。"""
    report = {"actual": {}, "unsupported": [], "skipped": False}
    if not hasattr(cap, "set") or not hasattr(cap, "get"):
        report["skipped"] = True
        return report

    settings = [
        ("autofocus", cv2.CAP_PROP_AUTOFOCUS,
         1.0 if cfg.get("camera_autofocus", True) else 0.0),
        ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE,
         0.75 if cfg.get("camera_auto_exposure", True) else 0.25),
        ("backlight", cv2.CAP_PROP_BACKLIGHT,
         float(cfg.get("camera_backlight", 0))),
    ]
    if not cfg.get("camera_autofocus", True):
        settings.insert(1, ("focus", cv2.CAP_PROP_FOCUS,
                            float(cfg.get("camera_focus", 160))))
    if not cfg.get("camera_auto_exposure", True):
        settings.insert(-1, ("exposure", cv2.CAP_PROP_EXPOSURE,
                             float(cfg.get("camera_exposure", -6))))

    for name, prop, value in settings:
        try:
            if cap.set(prop, value):
                report["actual"][name] = float(cap.get(prop))
            else:
                report["unsupported"].append(name)
        except Exception:
            report["unsupported"].append(name)
    return report


def prepare_detection_frame(frame, cfg):
    """为检测生成可选的局部对比度增强副本，不修改预览原图。"""
    if not cfg.get("detection_equalize", False):
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


class ImageSource:
    """静态图片源(测试用)。Windows 的 MSMF 后端读部分 jpg 会一直失败, 故直接 imread"""

    def __init__(self, path):
        self.frame = cv2.imread(path)
        if self.frame is None:
            print(f"[错误] 无法读取图片 {path}")
            sys.exit(1)

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


def preprocess(frame, cfg):
    """主循环与框选共用的画面预处理: 镜像 + 缩到 max_frame_size 以内"""
    if cfg["mirror"]:
        frame = cv2.flip(frame, 1)
    cap_size = cfg.get("max_frame_size", 1280)
    if max(frame.shape[:2]) > cap_size:
        s = cap_size / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=s, fy=s)
    return frame


def setup_roi(cap, cfg, force):
    """--select-roi 时弹窗拖框(Enter 确认/c 取消), 结果以归一化比例存入 config;
    平时读已有 roi, 没有则全画面检测"""
    if not force:
        if cfg.get("roi"):
            print(f"[检测区] 使用已保存 ROI {cfg['roi']}(框外的脸完全忽略)")
            return cfg["roi"]
        print("[检测区] 未设置, 全画面检测。想只盯一侧: python main.py --select-roi")
        return None
    ok, frame = cap.read()
    if not ok:
        print("[检测区] 读不到画面, 全画面检测")
        return None
    frame = preprocess(frame, cfg)
    r = cv2.selectROI("Select ROI: drag + Enter (c = full frame)", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    if r and r[2] > 0 and r[3] > 0:
        H, W = frame.shape[:2]
        cfg["roi"] = [[round(r[0] / W, 4), round(r[1] / H, 4), round(r[2] / W, 4), round(r[3] / H, 4)]]
        save_config(cfg)
        print(f"[检测区] ROI 已保存并写入 config.json: {cfg['roi']}")
        return cfg["roi"]
    print("[检测区] 未框选(取消), 全画面检测")
    return None


def crop_rois(frame, rois):
    """rois: None(全画面) 或 归一化框列表 [[x,y,w,h],...]
    返回 [(region, (ox, oy)), ...]"""
    H, W = frame.shape[:2]
    if not rois:
        return [(frame, (0, 0))]
    out = []
    for r in rois:
        x, y = int(r[0] * W), int(r[1] * H)
        w, h = int(r[2] * W), int(r[3] * H)
        if w > 10 and h > 10:
            out.append((frame[y:y + h, x:x + w], (x, y)))
    return out if out else [(frame, (0, 0))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="摄像头索引或视频/图片路径")
    ap.add_argument("--max-seconds", type=int, default=None, help="限时自动退出(测试用)")
    ap.add_argument("--select-roi", action="store_true", help="弹窗框选检测区域(存入config)")
    ap.add_argument("--test-switch", action="store_true", help="自测切屏与恢复")
    args = ap.parse_args()
    cfg = load_config()

    if args.test_switch:
        hwnd = find_guard_window(cfg["guard_window_keyword"])
        if not hwnd:
            print(f"[失败] 没找到含 '{cfg['guard_window_keyword']}' 的窗口")
            sys.exit(1)
        print("3 秒后切屏...")
        time.sleep(3)
        orig = switch_screen(hwnd)
        print(f"已切屏, 5 秒后恢复 (原前台 hwnd={orig})")
        time.sleep(5)
        restore_screen(hwnd, orig)
        print("已恢复。自测通过 ✔")
    else:
        run(cfg, args.source, args.max_seconds, select_roi=args.select_roi)

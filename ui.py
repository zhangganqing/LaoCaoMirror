# -*- coding: utf-8 -*-
"""
全自动防老曹后视镜 - 桌面界面
预览 + 状态 + 事件 / 框选检测区 / 设置(含护身窗口下拉选择)

用法: python ui.py [--source 测试源]
"""
import json
import sys
import threading

import cv2
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QRubberBand, QSpinBox,
    QVBoxLayout, QWidget,
)

import main as core

WINDOW_TITLE = "防老曹后视镜"


def list_visible_windows():
    """枚举可作为护身窗口的可见窗口标题(排除自身与预览窗)"""
    import win32con
    import win32gui
    out = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        t = win32gui.GetWindowText(hwnd)
        if not t or "RearviewMirror" in t or WINDOW_TITLE in t:
            return
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if not ex & win32con.WS_EX_TOOLWINDOW:
            out.append(t)
    win32gui.EnumWindows(cb, None)
    return sorted(set(out))


class Engine(QThread):
    """后台引擎线程: 复用 main.run_loop, 通过信号回传帧/状态/事件"""
    frame_ready = pyqtSignal(object)
    state_ready = pyqtSignal(str)
    event_ready = pyqtSignal(str)
    engine_stopped = pyqtSignal()

    def __init__(self, cfg, source=None):
        super().__init__()
        self.cfg = cfg
        self.source = source
        self.last_frame = None
        self._stop = threading.Event()
        self._restore = threading.Event()
        self._confirm_learn = threading.Event()
        self._reject = threading.Event()

    def confirm_restore(self):
        """命令行/旧调用兼容：仅恢复，不学习。"""
        self._restore.set()

    def confirm_and_learn(self):
        """用户确认识别正确：学习本次照片并恢复。"""
        self._confirm_learn.set()

    def reject_and_restore(self):
        """用户确认是误报：删除本次照片并恢复。"""
        self._reject.set()

    def run(self):
        try:
            detector, recognizer = core.build_models(self.cfg)
        except Exception as e:
            self.event_ready.emit(f"[错误] 模型加载失败: {e}")
            self.engine_stopped.emit()
            return
        guard_hwnd = core.report_startup(self.cfg, recognizer)
        try:
            cap = core.open_source(self.source, self.cfg)
        except SystemExit:
            self.event_ready.emit("[错误] 摄像头打不开, 请检查 camera_index 或占用")
            self.engine_stopped.emit()
            return
        if hasattr(cap, "get"):
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.event_ready.emit(f"[摄像头] 实际分辨率 {w}x{h} (低于720p则画面必然模糊)")
        cb = _EngineCb(self)
        try:
            core.run_loop(self.cfg, cap, detector, recognizer, guard_hwnd,
                          cb, stop_event=self._stop, restore_event=self._restore,
                          confirm_learn_event=self._confirm_learn,
                          reject_event=self._reject)
        except Exception as e:
            self.event_ready.emit(f"[错误] 引擎异常: {e}")
        self.engine_stopped.emit()

    def stop(self):
        self._stop.set()


class _EngineCb:
    def __init__(self, engine):
        self.engine = engine

    def on_event(self, text):
        self.engine.event_ready.emit(text)

    def on_state(self, state):
        self.engine.state_ready.emit(state)

    def on_frame(self, frame):
        self.engine.last_frame = frame
        self.engine.frame_ready.emit(frame)
        return True


def cv2qt(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img)


STATE_TEXT = {
    "PATROL": ("🟢 巡逻中 · 安全", "#1f7a2f", "#e8f6e8"),
    "ALERT": ("🟡 有人靠近 · 警戒", "#9a6b00", "#fff5dd"),
    "SUSPECT": ("🟠 疑似老曹 · 提高警惕", "#c06000", "#ffedd9"),
    "DEFEND": ("🔴 已切屏 · 请确认识别结果", "#a11212", "#fde3e3"),
}


class RoiLabel(QLabel):
    """显示当前帧(自适应缩放到屏幕内); 拖出的框记录在原帧坐标上, 可撤销/清空"""

    def __init__(self, frame, current_rois=None):
        super().__init__()
        self.frame = frame
        H, W = frame.shape[:2]
        scr = QApplication.primaryScreen().availableGeometry()
        self.scale = min(1.0, (scr.width() - 120) / W, (scr.height() - 260) / H)
        disp = frame
        if self.scale < 1.0:
            disp = cv2.resize(frame, (int(W * self.scale), int(H * self.scale)))
        self.setPixmap(cv2qt(disp))
        self.setFixedSize(disp.shape[1], disp.shape[0])
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.origin = None       # 原帧坐标
        self.cur = None
        # 框全部记录为原帧坐标(QRect); 预加载当前已生效的检测区
        self.rects = [QRect(round(r[0] * W), round(r[1] * H),
                            round(r[2] * W), round(r[3] * H))
                      for r in (current_rois or [])]
        self.rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)

    def _to_frame(self, pos):
        """显示坐标 -> 原帧坐标"""
        return QPoint(round(pos.x() / self.scale), round(pos.y() / self.scale))

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setPen(QPen(QColor(0, 210, 210), 2))
        p.scale(self.scale, self.scale)
        for r in self.rects:
            p.drawRect(r)

    def mousePressEvent(self, e):
        self.origin = self._to_frame(e.position())
        self.rubber.setGeometry(QRect(self._to_disp(self.origin), QSize()))
        self.rubber.show()

    def mouseMoveEvent(self, e):
        if self.origin is not None:
            self.cur = QRect(self.origin, self._to_frame(e.position())).normalized()
            self.rubber.setGeometry(QRect(
                round(self.cur.x() * self.scale), round(self.cur.y() * self.scale),
                round(self.cur.width() * self.scale), round(self.cur.height() * self.scale)))
            self.update()

    def _to_disp(self, pt):
        return QPoint(round(pt.x() * self.scale), round(pt.y() * self.scale))

    def mouseReleaseEvent(self, e):
        if self.origin is not None and self.cur is not None:
            if self.cur.width() > 8 / self.scale and self.cur.height() > 8 / self.scale:
                self.rects.append(self.cur)
        self.origin = None
        self.cur = None
        self.rubber.hide()
        self.update()

    def undo(self):
        if self.rects:
            self.rects.pop()
            self.update()

    def clear_all(self):
        self.rects = []
        self.update()


class RoiDialog(QDialog):
    """纯 Qt 的多框选对话框(不用 cv2 窗口: OpenCV 高层窗口与 Qt 事件循环同线程会死锁)"""

    def __init__(self, frame, current_rois=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选检测区(可拖多个)")
        self.frame = frame
        lay = QVBoxLayout(self)
        self.label = RoiLabel(frame, current_rois)
        lay.addWidget(self.label)
        n = len(self.label.rects)
        hint = QLabel("青色框 = 当前已生效的检测区。按住左键拖出新框(可多个), "
                      "「撤销上一框」「清空全部」可删。注意: 框的是老曹的【脸】会出现的"
                      "区域(通常在画面中上部, 人走近时脸不会贴地)。确定后框外的脸完全忽略; "
                      "清空后确定 = 全画面检测。")
        hint.setStyleSheet("color:#666;")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        btns = QHBoxLayout()
        undo_btn = QPushButton("撤销上一框")
        undo_btn.clicked.connect(self.label.undo)
        btns.addWidget(undo_btn)
        clear_btn = QPushButton("清空全部")
        clear_btn.clicked.connect(self.label.clear_all)
        btns.addWidget(clear_btn)
        btns.addStretch(1)
        lay.addLayout(btns)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_roi(self):
        """返回归一化框列表; 一个框都没拖则返回 None(=全画面)"""
        if not self.label.rects:
            return None
        H, W = self.frame.shape[:2]
        return [[round(r.x() / W, 4), round(r.y() / H, 4),
                 round(r.width() / W, 4), round(r.height() / H, 4)]
                for r in self.label.rects]


class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.cfg = dict(cfg)
        self.setMinimumWidth(420)

        form = QFormLayout(self)

        self.win_box = QComboBox()
        wins = list_visible_windows()
        self.win_box.addItem("(不启用切屏, 仅提醒)", None)
        cur = cfg.get("guard_window_keyword")
        for t in wins:
            self.win_box.addItem(t, t)
            if cur and cur in t:
                self.win_box.setCurrentIndex(self.win_box.count() - 1)
        form.addRow("护身窗口:", self.win_box)

        self.fps = QSpinBox(); self.fps.setRange(1, 30); self.fps.setValue(cfg["fps"])
        form.addRow("检测帧率 (fps):", self.fps)

        self.imgsz = QComboBox()
        for label_, v in [("流畅 640 (低配 CPU 推荐)", 640), ("均衡 960 (普通 CPU 推荐)", 960),
                          ("高清 1280 (高性能 CPU / GPU)", 1280)]:
            self.imgsz.addItem(label_, v)
        cur = int(cfg.get("detect_imgsz", 960))
        self.imgsz.setCurrentIndex(max(0, [640, 960, 1280].index(cur) if cur in (640, 960, 1280) else 1))
        form.addRow("检测清晰度:", self.imgsz)

        self.alert_th = QDoubleSpinBox(); self.alert_th.setRange(0.1, 0.9); self.alert_th.setSingleStep(0.05)
        self.alert_th.setValue(cfg.get("alert_threshold", 0.35))
        form.addRow("提醒阈值 (疑似老曹):", self.alert_th)
        self.th = QDoubleSpinBox(); self.th.setRange(0.2, 0.95); self.th.setSingleStep(0.05)
        self.th.setValue(cfg["id_threshold"])
        form.addRow("切屏阈值 (确认老曹):", self.th)
        self.confirm = QSpinBox(); self.confirm.setRange(1, 10)
        self.confirm.setValue(cfg["confirm_frames"])
        form.addRow("确认帧数 (防误检):", self.confirm)
        self.minr = QDoubleSpinBox(); self.minr.setRange(0.01, 0.5); self.minr.setSingleStep(0.01)
        self.minr.setValue(cfg["min_face_ratio"])
        form.addRow("忽略小于画面占比:", self.minr)
        self.cam = QSpinBox(); self.cam.setRange(0, 9); self.cam.setValue(cfg["camera_index"])
        form.addRow("摄像头编号:", self.cam)

        self.res = QComboBox()
        for label_, v in [("640 x 480", [640, 480]), ("1280 x 720", [1280, 720]), ("1920 x 1080", [1920, 1080])]:
            self.res.addItem(label_, v)
        cur_res = list(cfg.get("camera_resolution", [1920, 1080]))
        idx_res = {tuple(v): i for i, (_, v) in enumerate(
            [("640 x 480", [640, 480]), ("1280 x 720", [1280, 720]), ("1920 x 1080", [1920, 1080])])}
        self.res.setCurrentIndex(idx_res.get(tuple(cur_res), 2))
        form.addRow("摄像头分辨率:", self.res)

        self.beep = QCheckBox("有人靠近时提示音"); self.beep.setChecked(cfg["beep_alert"])
        form.addRow("", self.beep)
        self.mirror = QCheckBox("镜像预览(后视镜观感)"); self.mirror.setChecked(cfg["mirror"])
        form.addRow("", self.mirror)

        tip = QLabel("保存后立即生效(引擎自动重启)。识别阈值建议先实跑,\n"
                     "看事件里老曹的相似度分数再微调。")
        tip.setStyleSheet("color:#666;")
        form.addRow(tip)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def result_cfg(self):
        win = self.win_box.currentData()
        c = dict(self.cfg)
        c["guard_window_keyword"] = win if win else ""
        c["fps"] = self.fps.value()
        c["detect_imgsz"] = self.imgsz.currentData()
        c["alert_threshold"] = round(self.alert_th.value(), 2)
        c["id_threshold"] = round(self.th.value(), 2)
        c["confirm_frames"] = self.confirm.value()
        c["min_face_ratio"] = round(self.minr.value(), 2)
        c["camera_index"] = self.cam.value()
        c["camera_resolution"] = self.res.currentData()
        c["beep_alert"] = self.beep.isChecked()
        c["mirror"] = self.mirror.isChecked()
        return c


class MainWindow(QMainWindow):
    def __init__(self, cfg, source=None, screenshot_path=None):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.cfg = dict(cfg)
        self.source = source
        self.engine = None
        self.defend_count = 0
        self.screenshot_path = screenshot_path

        central = QWidget()
        root = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.preview = QLabel("点击\"启动巡逻\"开始")
        self.preview.setMinimumSize(640, 480)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#888; font-size:14px;")
        root.addWidget(self.preview, 4)

        side = QVBoxLayout()
        root.addLayout(side, 2)

        self.status = QLabel("⚪ 未启动")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setMinimumHeight(44)
        self.status.setStyleSheet("font-size:17px; font-weight:bold; padding:6px;")
        side.addWidget(self.status)

        self.counter = QLabel("今日防御次数: 0")
        side.addWidget(self.counter)

        self.events = QListWidget()
        self.events.setWordWrap(True)   # 长日志自动换行, 不截断
        side.addWidget(self.events, 1)

        btns = QVBoxLayout()
        self.btn_toggle = QPushButton("启动巡逻")
        self.btn_toggle.clicked.connect(self.toggle_engine)
        btns.addWidget(self.btn_toggle)
        self.btn_confirm_learn = QPushButton("确认是老曹 · 学习并恢复")
        self.btn_confirm_learn.setStyleSheet("font-weight:bold; color:#176b2c;")
        self.btn_confirm_learn.setEnabled(False)
        self.btn_confirm_learn.clicked.connect(self.confirm_and_learn)
        btns.addWidget(self.btn_confirm_learn)
        self.btn_reject = QPushButton("误报 · 删除并恢复")
        self.btn_reject.setStyleSheet("font-weight:bold; color:#9a3b00;")
        self.btn_reject.setEnabled(False)
        self.btn_reject.clicked.connect(self.reject_and_restore)
        btns.addWidget(self.btn_reject)
        self.btn_roi = QPushButton("框选检测区")
        self.btn_roi.clicked.connect(self.pick_roi)
        btns.addWidget(self.btn_roi)
        self.btn_set = QPushButton("设置")
        self.btn_set.clicked.connect(self.open_settings)
        btns.addWidget(self.btn_set)
        self.btn_quit = QPushButton("退出")
        self.btn_quit.clicked.connect(self.close)
        btns.addWidget(self.btn_quit)
        side.addLayout(btns)

        self.log_event("提示: 先在\"设置\"里选择护身窗口, 再启动巡逻")
        if not cfg.get("guard_window_keyword"):
            self.log_event("⚠ 当前未配置护身窗口: 检测到老曹只会提醒, 不会切屏!")
            self.log_event("  (设置→护身窗口→选择打开着的课件/文档)")

        if self.screenshot_path:
            self.start_engine()
            QTimer.singleShot(30000, self.take_screenshot)   # exe 冷启动加载模型较慢

    # ---- 引擎控制 ----
    def toggle_engine(self):
        if self.engine:
            self.stop_engine()
        else:
            self.start_engine()

    def start_engine(self):
        self.engine = Engine(self.cfg, self.source)
        self.engine.frame_ready.connect(self.show_frame)
        self.engine.state_ready.connect(self.show_state)
        self.engine.event_ready.connect(self.log_event)
        self.engine.engine_stopped.connect(self.on_engine_stopped)
        self.engine.start()
        self.btn_toggle.setText("停止巡逻")
        self.status.setText("⚪ 启动中…")
        self.log_event("[引擎] 启动中, 预览几秒后恢复")

    def stop_engine(self):
        if self.engine:
            self.engine.stop()
            if not self.engine.wait(3000):
                self.log_event("[警告] 引擎停止超时, 强制继续")
            self.engine = None

    def on_engine_stopped(self):
        if self.engine is not None:   # 引擎自己报错退出时同步按钮
            self.engine = None
            self.btn_toggle.setText("启动巡逻")
            self.status.setText("⚪ 未启动")

    # ---- 信号槽 ----
    def show_frame(self, frame):
        pm = cv2qt(frame)
        self.preview.setPixmap(pm.scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def show_state(self, state):
        text, fg, bg = STATE_TEXT.get(state, (state, "#333", "#eee"))
        self.status.setText(text)
        self.status.setStyleSheet(
            f"font-size:17px; font-weight:bold; color:{fg}; background:{bg};"
            " border-radius:6px; padding:6px;")
        waiting_review = state == "DEFEND"
        self.btn_confirm_learn.setEnabled(waiting_review)
        self.btn_reject.setEnabled(waiting_review)

    def confirm_and_learn(self):
        if self.engine:
            self.engine.confirm_and_learn()
            self.btn_confirm_learn.setEnabled(False)
            self.btn_reject.setEnabled(False)

    def reject_and_restore(self):
        if self.engine:
            self.engine.reject_and_restore()
            self.btn_confirm_learn.setEnabled(False)
            self.btn_reject.setEnabled(False)

    def manual_restore(self):
        """兼容旧测试和外部调用：只恢复，不把未确认照片加入图库。"""
        if self.engine:
            self.engine.confirm_restore()

    def log_event(self, text):
        from datetime import datetime
        full = f"{datetime.now().strftime('%H:%M:%S')}  {text}"
        item = QListWidgetItem(full)
        # 按换行后的实际高度设置行高, 保证文字完整显示
        fm = self.events.fontMetrics()
        w = max(self.events.viewport().width() - 24, 160)
        rect = fm.boundingRect(QRect(0, 0, w, 2000), Qt.TextFlag.TextWordWrap, full)
        item.setSizeHint(QSize(0, rect.height() + 8))
        self.events.addItem(item)
        self.events.scrollToBottom()
        if self.events.count() > 200:
            self.events.takeItem(0)
        if text.startswith("[防御]") and "已切屏" in text:
            self.defend_count += 1
            self.counter.setText(f"今日防御次数: {self.defend_count}")

    # ---- ROI 框选 ----
    def pick_roi(self):
        """框选期间引擎保持运行(摄像头不重开), 确认后 cfg 原地更新, 下一帧即生效"""
        frame = self.engine.last_frame if self.engine else None
        if frame is None:
            try:
                cap = core.open_source(self.source, self.cfg)
                ok, f = cap.read()
                cap.release()
                frame = core.preprocess(f, self.cfg) if ok else None
            except SystemExit:
                frame = None
        if frame is None:
            self.log_event("[错误] 拿不到画面, 无法框选")
            return
        dlg = RoiDialog(frame, current_rois=self.cfg.get("roi"), parent=self)
        if dlg.exec():
            rois = dlg.result_roi()
            self.cfg["roi"] = rois          # 原地更新(引擎持同一 dict 引用), 热生效
            core.save_config(self.cfg)
            if rois:
                self.log_event(f"[检测区] 已应用 {len(rois)} 个检测区, 框外完全忽略")
            else:
                self.log_event("[检测区] 未拖框, 已设为全画面检测")
        else:
            self.log_event("[检测区] 已取消, 保持原设置")

    # ---- 设置 ----
    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            new_cfg = dlg.result_cfg()
            cam_changed = (self.engine is not None and (
                new_cfg.get("camera_index") != self.cfg.get("camera_index") or
                list(new_cfg.get("camera_resolution", [])) !=
                list(self.cfg.get("camera_resolution", []))))
            self.cfg.clear()
            self.cfg.update(new_cfg)   # 原地更新(引擎共享同一引用), 除摄像头外全部热生效
            core.save_config(self.cfg)
            if cam_changed:
                self.log_event("[设置] 已保存, 摄像头变更, 重启引擎…")
                self.stop_engine()
                self.start_engine()
            else:
                self.log_event("[设置] 已保存并即时生效")

    def take_screenshot(self):
        pm = self.grab()
        pm.save(self.screenshot_path)
        self.stop_engine()
        QApplication.quit()

    def closeEvent(self, event):
        self.stop_engine()
        event.accept()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="测试源(摄像头索引/视频/图片)")
    ap.add_argument("--screenshot", default=None, help="启动9秒后截图保存并退出(自检用)")
    args = ap.parse_args()

    cfg = core.load_config()
    app = QApplication(sys.argv)
    win = MainWindow(cfg, source=args.source, screenshot_path=args.screenshot)
    win.resize(980, 560)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

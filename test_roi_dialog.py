# -*- coding: utf-8 -*-
"""RoiDialog 多框拖选自动化测试: 覆盖 1:1 显示与高分辨率帧缩放显示两种场景"""
import faulthandler
import sys

faulthandler.enable()

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from ui import RoiDialog

app = QApplication(sys.argv)


def drag(dlg, x1, y1, x2, y2):
    label = dlg.label
    QTest.mousePress(label, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x1, y1))
    s, e = QPoint(x1, y1), QPoint(x2, y2)
    for i in range(1, 6):
        QTest.mouseMove(label, s + (e - s) * i / 5)
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, e)


def check_roi(got, exp_disps, scale, size, tag):
    """exp_disps: 每框的显示坐标拖拽点 [x,y,w,h]; 换算到原帧后应与 got 一致(±0.01)"""
    W, H = size
    assert got and len(got) == len(exp_disps), f"{tag}: 框数量不符 {len(got) if got else 0} vs {len(exp_disps)}"
    for g, ed in zip(got, exp_disps):
        exp = [ed[0] / (scale * W), ed[1] / (scale * H),
               ed[2] / (scale * W), ed[3] / (scale * H)]
        assert all(abs(a - b) < 0.01 for a, b in zip(g, exp)), f"{tag}: {g} vs {exp}"


# 场景1: 小帧(640x480), 通常 1:1 显示
frame = np.full((480, 640, 3), 90, dtype=np.uint8)
dlg = RoiDialog(frame)
dlg.show()
QTest.qWaitForWindowExposed(dlg)
print(f"场景1 scale={dlg.label.scale:.2f}")
drag(dlg, 100, 80, 300, 240)
drag(dlg, 400, 300, 550, 420)
rois = dlg.result_roi()
check_roi(rois, [[100, 80, 200, 160], [400, 300, 150, 120]], dlg.label.scale, (640, 480), "场景1")
dlg.label.undo()
assert len(dlg.result_roi()) == 1, "撤销失败"
dlg.label.clear_all()
assert dlg.result_roi() is None, "清空失败"
print("场景1(1:1) 多框/撤销/清空 PASS")

# 场景2: 1080p 帧, 应自适应缩小显示且坐标换算正确
frame2 = np.full((1080, 1920, 3), 60, dtype=np.uint8)
dlg2 = RoiDialog(frame2)
dlg2.show()
QTest.qWaitForWindowExposed(dlg2)
lw, lh = dlg2.label.width(), dlg2.label.height()
print(f"场景2 scale={dlg2.label.scale:.2f}, 显示 {lw}x{lh} (帧 1920x1080)")
assert max(lw, lh) <= 1920 and dlg2.label.scale <= 1.0, "显示未自适应缩小"
drag(dlg2, 100, 80, 300, 240)
check_roi(dlg2.result_roi(), [[100, 80, 200, 160]], dlg2.label.scale, (1920, 1080), "场景2")
print("场景2(1080p 缩放显示) 坐标换算 PASS")
print("RoiDialog 全部自测 PASS")

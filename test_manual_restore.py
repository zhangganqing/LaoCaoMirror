# -*- coding: utf-8 -*-
"""手动恢复闭环测试: 图片源恒有老曹脸 -> 进入防御 -> 手动确认 -> 回到巡逻"""
import json
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from ui import MainWindow

app = QApplication(sys.argv)
cfg = json.load(open("config.json", encoding="utf-8"))
cfg["roi"] = None   # 测试不依赖用户框选的检测区
win = MainWindow(cfg, source="laocao/0f18b067c807a1bbbcdb0e2450cfd880.jpg")
win.resize(900, 560)
win.show()
win.start_engine()

phase = ["wait_defend", "wait_patrol"]
deadline = 40   # 秒
elapsed = [0]


def poll():
    elapsed[0] += 1
    text = win.status.text()
    if phase[0] == "wait_defend":
        if text.startswith("🔴"):
            print(f"{elapsed[0]}s 进入防御态 ✔, 发出手动恢复")
            win.manual_restore()
            phase[0] = "wait_restore"
    elif phase[0] == "wait_restore":
        # 图片源恒有脸: 恢复后 3 帧内会重新进入防御(逻辑正确), 故以恢复事件日志为准
        logs = [win.events.item(i).text() for i in range(win.events.count())]
        if any("[恢复] 已手动确认老曹离开" in t for t in logs):
            print(f"{elapsed[0]}s 收到手动恢复事件 ✔")
            print("手动恢复闭环 PASS")
            win.stop_engine()
            app.quit()
            return
    if elapsed[0] >= deadline:
        print(f"FAIL: {deadline}s 内停留在 [{phase[0]}], 当前状态: {text}")
        win.stop_engine()
        app.quit()


timer = QTimer()
timer.timeout.connect(poll)
timer.start(1000)
sys.exit(app.exec())

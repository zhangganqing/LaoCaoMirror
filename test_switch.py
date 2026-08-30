# -*- coding: utf-8 -*-
"""切屏/恢复闭环回归测试: 自建两个 tkinter 窗口模拟真实场景
A(UserWindow) = 用户正在用的界面; B(GuardWindow) = 护身窗口
验证: A -> 切到 B -> 自动恢复回 A
"""
import subprocess
import sys
import time

import win32gui

from main import find_guard_window, restore_screen, switch_screen, switch_to

WINDOW_SCRIPT = r"""
import tkinter as tk, sys
root = tk.Tk(); root.title(sys.argv[1]); root.geometry('420x300')
tk.Label(root, text=sys.argv[1], font=('Arial', 22)).pack(expand=True)
root.mainloop()
"""

fg = win32gui.GetForegroundWindow
results = []

def check(name, cond):
    results.append((name, cond))
    extra = f"  (实际前台: {win32gui.GetWindowText(fg())!r})" if not cond else ""
    print(f"{'PASS' if cond else 'FAIL'}  {name}{extra}")

g = subprocess.Popen([sys.executable, "-c", WINDOW_SCRIPT, "GuardWindow-护身界面"])
u = subprocess.Popen([sys.executable, "-c", WINDOW_SCRIPT, "UserWindow-正在学习"])
time.sleep(3)
try:
    hwnd_g = find_guard_window("GuardWindow")
    hwnd_u = find_guard_window("UserWindow")
    check("找到护身窗口", bool(hwnd_g))
    check("找到用户窗口", bool(hwnd_u))

    switch_to(hwnd_u); time.sleep(0.8)
    orig = fg()
    check("起点: 前台不是护身窗口", orig != hwnd_g and orig != 0)

    switch_screen(hwnd_g); time.sleep(1)
    check("切屏: 护身窗口接管前台", fg() == hwnd_g)

    restore_screen(hwnd_g, orig); time.sleep(1)
    check("恢复: 切回用户窗口", fg() == orig)
finally:
    g.kill(); u.kill()

print("-" * 40)
print("闭环自测:", "全部通过" if all(c for _, c in results) else "存在失败项")
sys.exit(0 if all(c for _, c in results) else 1)

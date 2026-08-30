# -*- coding: utf-8 -*-
"""测试辅助: 起一个标题含 GuardWindow 的窗口充当护身窗口(测试期间后台挂着)"""
import tkinter as tk

root = tk.Tk()
root.title("GuardWindow-护身界面")
root.geometry("420x300")
tk.Label(root, text="高等数学 第三章\n(护身界面)", font=("Microsoft YaHei", 18)).pack(expand=True)
root.mainloop()

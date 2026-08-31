# -*- coding: utf-8 -*-
"""CPU 模式资源实测: 跑 main.py 15 秒, 逐秒采样 CPU/内存。"""
import os
import subprocess
import sys
import time

import psutil

child_env = dict(os.environ)
child_env["LCSB_COMPUTE_BACKEND"] = "cpu"  # 有独显的开发机也必须测到朋友电脑的 CPU 发行版
child_env["LCSB_DISTRIBUTION_LABEL"] = "CPU"
p = subprocess.Popen(
    [sys.executable, "-u", "main.py", "--source", "test_face.jpg", "--max-seconds", "15"],
    stdout=sys.stdout, stderr=sys.stderr, env=child_env)
time.sleep(2)  # 等模型加载完
proc = psutil.Process(p.pid)
n_cpu = psutil.cpu_count()
samples = []
try:
    for i in range(12):
        if not p.poll() is None:
            print(f"子进程在第 {i+3}s 前退出, 退出码 {p.returncode}")
            break
        cpu = proc.cpu_percent(interval=1)   # 100% = 占满一个核
        mem = proc.memory_info().rss / 1024**2
        samples.append((cpu, mem))
        print(f"t={i+3:>2}s  CPU={cpu:6.1f}%核  MEM={mem:6.0f}MB")
finally:
    if p.poll() is None:
        p.kill()

if samples:
    avg_cpu = sum(c for c, _ in samples) / len(samples)
    peak_mem = max(m for _, m in samples)
    print("-" * 42)
    print(f"平均: CPU {avg_cpu:.0f}%核 (≈全机 {avg_cpu/n_cpu:.0f}%, {n_cpu} 核)  内存峰值 {peak_mem:.0f} MB")

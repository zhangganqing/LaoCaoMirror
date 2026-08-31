# -*- coding: utf-8 -*-
"""NVIDIA GPU 发行版固定入口。"""
import os

os.environ["LCSB_COMPUTE_BACKEND"] = "gpu"
os.environ["LCSB_DISTRIBUTION_LABEL"] = "GPU"

from ui import main


if __name__ == "__main__":
    main()

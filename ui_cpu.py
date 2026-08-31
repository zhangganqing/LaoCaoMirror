# -*- coding: utf-8 -*-
"""CPU 发行版固定入口。"""
import os

os.environ["LCSB_COMPUTE_BACKEND"] = "cpu"
os.environ["LCSB_DISTRIBUTION_LABEL"] = "CPU"

from ui import main


if __name__ == "__main__":
    main()

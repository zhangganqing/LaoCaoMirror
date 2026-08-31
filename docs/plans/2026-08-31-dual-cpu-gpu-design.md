# CPU / GPU 双版本设计

## 目标

从同一套源码生成两个明确区分的版本。CPU 版不携带 PyTorch、Ultralytics 或 CUDA 运行时，三个模型全部通过 ONNX Runtime CPU 执行；GPU 版使用 ONNX Runtime CUDA 执行三个模型，缺少 NVIDIA 驱动、CUDA、cuDNN 或 GPU 版运行时时明确报错，不静默伪装成 GPU 后退到 CPU。两版共用 `config.json`、`models/`、`laocao/` 和审核记录格式。

## 架构

把 `yolov8n-face.pt` 一次性导出为动态输入的 FP32 ONNX，并用项目自己的轻量预处理、输出解析和 NMS 替代运行时的 Ultralytics/PyTorch。`FaceDetector`、`FaceAligner`、`FaceRecognizer` 统一由一个 ONNX 会话工厂创建；CPU 启动器强制选择 `CPUExecutionProvider`，GPU 启动器强制要求 `CUDAExecutionProvider`。开发入口 `ui.py` 保持自动选择，便于本机调试，但两个发布入口不会因用户配置互相串版。

## 分发与错误处理

新增 CPU/GPU 独立启动器、依赖清单和 PyInstaller spec，输出名分别为 `LaoCaoMirror-CPU` 与 `LaoCaoMirror-GPU`。GPU 版启动时先验证 CUDA provider，失败信息说明需要安装匹配版本的 NVIDIA 驱动、CUDA、cuDNN 与 `onnxruntime-gpu`；CPU 版即使机器有显卡也只走 CPU。启动日志逐项报告检测、对齐、识别模型的实际 provider。暂时只准备构建配置并完成脚本实测，不创建压缩包、不推送、不发布 Release。

## 验证

测试覆盖 provider 选择、CPU 强制模式、GPU 缺失时报错、模型缓存按后端隔离、YOLO ONNX 输出解析、重复框 NMS/碎片框合并，以及两个启动器的固定模式。使用现有真实照片对比旧 `.pt` 与新 ONNX 的检测数量和坐标，并运行全部 38 项现有回归、ROI 专项、语法检查和 CPU/GPU 启动检查。

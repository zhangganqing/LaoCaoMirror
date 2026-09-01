# GPU 安装帮助实现计划

> **注意：** 使用 executing-plans skill 逐任务实现此计划。

**目标：** 让 GPU 版缺少运行库时显示可操作的安装帮助，并随程序提供完整中文教程。

**架构：** 核心层生成统一诊断文本，Qt 层负责弹窗和打开教程，打包层将教程复制到 GPU 版本目录。教程依据 ONNX Runtime、NVIDIA、Microsoft 官方要求编写。

**技术栈：** Python、PyQt6、ONNX Runtime GPU、PowerShell、PyInstaller、unittest。

---

### 任务 1：统一 GPU 诊断文本

**文件：**
- 修改：`main.py`
- 修改：`test_inference_backend.py`

1. 先写失败测试，要求诊断包含 CUDA 12.x、cuDNN 9.x、VC++、onnxruntime-gpu、教程路径和 `where.exe` 命令。
2. 运行 `python -m unittest test_inference_backend -v`，确认因函数缺失失败。
3. 实现 `gpu_setup_guide_path()` 与 `gpu_setup_message()`，并让 GPU provider 缺失/加载失败共用该消息。
4. 重跑测试，确认通过。

### 任务 2：界面安装帮助

**文件：**
- 修改：`ui.py`
- 修改：`test_inference_backend.py`

1. 先写失败测试，要求 GPU 模型错误发出专用信号且 GPU 界面存在帮助按钮。
2. 引擎捕获 GPU 环境错误后发信号；主线程显示对话框并可打开教程。
3. 使用 `os.startfile` 打开教程，失败时记录完整路径。
4. 运行后端和 UI 生命周期测试。

### 任务 3：中文教程与打包

**文件：**
- 创建：`GPU安装教程.md`
- 修改：`README.md`
- 修改：`LaoCaoMirrorGPU.spec`
- 修改：`test_inference_backend.py`

1. 先写静态失败测试，要求教程含两种安装路线、官方链接和验证命令，spec 包含教程。
2. 根据官方文档编写教程并加入 GPU 包数据文件。
3. 运行静态测试和 spec 语法检查。

### 任务 4：回归与提交

1. 运行全部 unittest、ROI 专项和 `py_compile`。
2. 运行 GPU 缺失诊断命令，检查用户可见文案。
3. 检查未生成打包产物、未修改照片或个人配置。
4. 本地提交，不推送、不发布。

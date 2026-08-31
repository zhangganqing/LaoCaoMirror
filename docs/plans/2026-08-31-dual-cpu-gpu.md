# CPU / GPU 双版本实现计划

> **注意：** 使用 executing-plans skill 逐任务实现此计划。

**目标：** 从同一套源码提供轻量 CPU 版与全模型 CUDA GPU 版，并让两版配置和照片库完全兼容。

**架构：** 将 YOLO 人脸检测器导出为 FP32 动态 ONNX，三个模型统一通过 ONNX Runtime 会话工厂运行。两个固定后端启动器与两套依赖/构建配置负责分发，GPU 缺失时明确失败而不静默回退。

**技术栈：** Python、ONNX Runtime / ONNX Runtime GPU、OpenCV、NumPy、PyQt6、PyInstaller、unittest。

---

### 任务 1：后端选择与会话工厂

**文件：**
- 创建：`test_inference_backend.py`
- 修改：`main.py`

**步骤 1：编写失败测试**

覆盖 CPU 只返回 `CPUExecutionProvider`、GPU 缺少 CUDA provider 时抛出可理解错误、GPU 可用时优先 CUDA，以及模型缓存键包含后端。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_inference_backend -v`
预期：因后端解析函数不存在而 FAIL。

**步骤 3：编写最小实现**

增加 `resolve_compute_backend()`、`execution_providers()` 与 `create_onnx_session()`，将对齐和识别模型接入会话工厂，并在启动日志报告实际 provider。

**步骤 4：运行测试验证通过**

运行：`python -m unittest test_inference_backend -v`
预期：PASS。

### 任务 2：YOLO ONNX 检测器

**文件：**
- 创建：`models/yolov8n-face.onnx`
- 修改：`main.py`
- 修改：`test_detection_pipeline.py`

**步骤 1：编写失败测试**

用合成 ONNX 输出验证 letterbox 坐标还原、置信度过滤和 NMS；确保接口仍返回 `(x1,y1,x2,y2)`。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_detection_pipeline -v`
预期：因当前检测器仍依赖 Ultralytics 或缺少解析函数而 FAIL。

**步骤 3：导出模型并实现最小检测器**

运行一次：`python -c "from ultralytics import YOLO; YOLO('models/yolov8n-face.pt').export(format='onnx', dynamic=True, simplify=False, opset=17)"`

实现 NumPy/OpenCV 预处理、ONNX 推理、输出解析和 NMS；移除运行时的 `torch`、`ultralytics` 导入。

**步骤 4：验证检测器**

运行：`python -m unittest test_detection_pipeline -v`
预期：PASS。再用现有两张重复框照片比较旧/新检测结果。

### 任务 3：固定 CPU/GPU 启动入口

**文件：**
- 创建：`ui_cpu.py`
- 创建：`ui_gpu.py`
- 修改：`ui.py`
- 测试：`test_inference_backend.py`

**步骤 1：编写失败测试**

验证 CPU 启动器在导入 UI 前固定 `cpu`，GPU 启动器固定 `gpu`，界面日志能展示版本名称和 provider。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_inference_backend -v`
预期：启动器不存在而 FAIL。

**步骤 3：实现启动器**

两个入口只设置 `LCSB_COMPUTE_BACKEND` 和版本标签后调用共享 `ui.main()`；不在设置中暴露可串版的切换项。

**步骤 4：运行测试验证通过**

运行：`python -m unittest test_inference_backend -v`
预期：PASS。

### 任务 4：依赖与双构建配置

**文件：**
- 创建：`requirements-common.txt`
- 创建：`requirements-cpu.txt`
- 创建：`requirements-gpu.txt`
- 创建：`LaoCaoMirrorCPU.spec`
- 创建：`LaoCaoMirrorGPU.spec`
- 创建：`build_cpu.ps1`
- 创建：`build_gpu.ps1`
- 修改：`.gitignore`

**步骤 1：编写静态失败测试**

在 `test_inference_backend.py` 验证 CPU 依赖不包含 torch/ultralytics/CUDA，GPU 依赖使用 `onnxruntime-gpu`，两套 spec 使用正确入口和不同输出名。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_inference_backend -v`
预期：文件缺失而 FAIL。

**步骤 3：添加最小构建配置**

公共依赖只保留界面、OpenCV、NumPy、pywin32；CPU/GPU 文件分别选择互斥的 ONNX Runtime 包。spec 收集 Qt/OpenCV/ONNX Runtime 必需资源并排除 torch/ultralytics。

**步骤 4：静态验证**

运行：`python -m unittest test_inference_backend -v`
预期：PASS。不在此阶段执行正式打包。

### 任务 5：回归、真实模型与文档

**文件：**
- 修改：`README.md`
- 修改：`config.example.json`（仅在需要新增兼容字段时）

**步骤 1：运行全部自动测试**

运行：`python -m unittest test_batch_review test_camera_controls test_desktop_fallback test_detection_pipeline test_engine_lifecycle test_face_alignment test_face_learning test_inference_backend test_review_controls test_review_queue -v`
预期：全部 PASS。

**步骤 2：运行专项检查**

运行：`python test_roi_dialog.py`

运行：`python -m py_compile main.py ui.py ui_cpu.py ui_gpu.py`

预期：全部 PASS。

**步骤 3：实际后端验证**

CPU：`$env:LCSB_COMPUTE_BACKEND='cpu'; python main.py --source test_face.jpg --max-seconds 2`

GPU：`$env:LCSB_COMPUTE_BACKEND='gpu'; python main.py --source test_face.jpg --max-seconds 2`

预期：日志分别显示三个模型使用 CPU / CUDA，检测结果与旧模型在允许的坐标误差内一致。

**步骤 4：更新文档并检查差异**

说明两个版本的依赖、CUDA/cuDNN 前置条件、运行入口和不静默回退策略；确认没有修改用户照片、审核队列、打包产物或 GitHub Release。

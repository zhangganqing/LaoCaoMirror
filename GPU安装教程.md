# GPU 版安装教程（Windows / NVIDIA）

GPU 版只支持 NVIDIA 显卡。普通电脑、AMD/Intel 显卡或不想安装 CUDA 的用户，请直接使用 CPU 版。

## 为什么有显卡，程序仍显示“GPU 环境未安装完整”

任务管理器能看到显卡、`nvidia-smi` 能运行，只能证明 **NVIDIA 驱动**可用；PyTorch 能使用 CUDA，也可能只是因为 PyTorch 自带了一套私有 CUDA DLL。GPU 版使用的是 ONNX Runtime 1.26，它还必须能找到自己的 CUDA 12.x 和 cuDNN 9.x DLL。

常见缺失文件：

- `cublasLt64_12.dll`：CUDA 12.x 未安装，或 CUDA `bin` 不在 PATH。
- `cudnn64_9.dll`：cuDNN 9.x 未安装，或 cuDNN `bin` 不在 PATH。
- 没有 `CUDAExecutionProvider`：装成了 CPU 版 `onnxruntime`，或 GPU 运行库加载失败。

ONNX Runtime 1.26 官方对应 CUDA 12.8（兼容 CUDA 12.x）和 cuDNN 9.x。

## 路线 A：运行打包版 GPU 程序

这是拿到 `LaoCaoMirror-GPU` 文件夹后的安装方法。

### 1. 确认 NVIDIA 驱动

打开 PowerShell：

```powershell
nvidia-smi
```

能看到显卡名称和驱动版本即可。命令不存在或报错时，先从 [NVIDIA 驱动下载页](https://www.nvidia.com/Download/index.aspx) 安装驱动。

### 2. 安装 Microsoft Visual C++ x64 运行库

下载并运行微软官方安装包：

- [Visual C++ 2015–2022 Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)

已经安装也可以再次运行并选择“修复”。

### 3. 安装 CUDA 12.x

建议安装 CUDA Toolkit 12.8：

- [CUDA Toolkit 官方下载](https://developer.nvidia.com/cuda-downloads)
- 如果官网默认显示 CUDA 13，请从 [CUDA 12.8 下载归档](https://developer.nvidia.com/cuda-12-8-0-download-archive) 选择 Windows → x86_64 → 你的系统版本 → exe (local)。

按默认选项安装。完成后，新开 PowerShell 检查：

```powershell
where.exe cublasLt64_12.dll
```

正常应显示类似：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cublasLt64_12.dll
```

### 4. 安装 cuDNN 9.x for CUDA 12

- [cuDNN 官方下载页](https://developer.nvidia.com/cudnn)
- [NVIDIA Windows 安装说明](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)

选择 cuDNN 9.x、CUDA 12、Windows x86_64。使用图形安装器时按默认选项安装；使用 zip 包时，按 NVIDIA 文档放置 `bin/include/lib`，并将 cuDNN 的 `bin` 目录加入系统 PATH。

新开 PowerShell 检查：

```powershell
where.exe cudnn64_9.dll
```

### 5. 重启电脑并验证

三个命令都应成功：

```powershell
nvidia-smi
where.exe cublasLt64_12.dll
where.exe cudnn64_9.dll
```

然后双击 `LaoCaoMirror-GPU.exe`。事件日志必须显示：

```text
[算力] GPU版 | 检测=CUDAExecutionProvider | 对齐=CUDAExecutionProvider | 识别=CUDAExecutionProvider
```

任何一项显示 CPU 都不算安装成功。

## 路线 B：从源码运行 GPU 版

源码版可以让 pip 安装 ONNX Runtime 需要的 CUDA/cuDNN 运行 DLL，不必安装完整 CUDA Toolkit，但仍需可用的 NVIDIA 驱动。

在项目目录打开 PowerShell：

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-source.txt
.\.venv-gpu\Scripts\python.exe ui_gpu.py
```

`requirements-gpu-source.txt` 使用 ONNX Runtime 官方提供的 extras：

```text
onnxruntime-gpu[cuda,cudnn]==1.26.0
```

程序启动时会预加载这些 pip 安装的 NVIDIA DLL。不要在同一虚拟环境同时安装 `onnxruntime` 和 `onnxruntime-gpu`。
首次安装会下载数百 MB 到数 GB 的 NVIDIA 运行库，耗时取决于网络速度，请等待命令完整结束。

## 仍然失败时怎么检查

在使用的 Python 环境运行：

```powershell
python -c "import onnxruntime as ort; ort.preload_dlls(); print(ort.get_available_providers()); ort.print_debug_info()"
```

输出必须包含：

```text
CUDAExecutionProvider
```

如果只看到 `CPUExecutionProvider`，请确认没有混装 CPU 包：

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
```

安装后关闭所有旧的程序窗口和 PowerShell，重新打开再测试。

## 官方依据

- [ONNX Runtime CUDA Execution Provider 版本要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime 安装说明](https://onnxruntime.ai/docs/install/)
- [NVIDIA CUDA Windows 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA cuDNN 安装指南](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/)

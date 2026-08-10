# 智能交通目标检测系统（核心版）

这是从原始工作目录提取出的可部署、可演示核心项目。该目录不包含依赖缓存、训练过程缓存、重复视频、历史任务文件和完整 COCO 原始数据集。

## 保留内容

- `backend/app/`：FastAPI 后端、检测、视频任务、训练、模型对比和历史删除实现。
- `frontend/src/`：React 前端界面与交互实现。
- `backend/models/`：YOLOv8n、YOLOv8s、YOLOv8m 与一个交通车辆自训练样例，共 4 个模型（首次启动自动下载官方预训练权重）。
- `deliverables/`：毕业设计报告和答辩演示文稿。
- `docs/`：部署、接口、演示和论文写作说明。

## 启动

前置条件：Python 3.11+、Node.js 18+。首次启动时脚本会自动创建后端虚拟环境并安装 Python 与前端依赖；依赖不会包含在本交付目录内。

请在已有的 PowerShell 窗口中运行脚本，不要双击 `start-dev.ps1`。双击会启动一个临时窗口，服务转入后台后窗口会自动关闭。
下面代码块中每行只输入命令文字；不要输入前面的 `PS D:\路径>` 提示符，也不要把之前的安装日志粘贴回 PowerShell。

```powershell
cd D:\路径\traffic_detection_core
Set-ExecutionPolicy -Scope Process Bypass
.\start-dev.ps1
```

首次启动需要联网下载依赖，之后只有 `backend/requirements.txt` 或 `frontend/package-lock.json` 变更时才会重新安装。运行产生的 `.venv/` 和 `node_modules/` 是本机缓存，可删除后由脚本自动恢复。

若需要从新窗口启动并保留终端输出，请使用：

```powershell
pwsh -NoExit -ExecutionPolicy Bypass -File D:\路径\traffic_detection_core\start-dev.ps1
```

启动完成后访问 `http://127.0.0.1:5173`，接口文档位于 `http://127.0.0.1:8000/docs`。

## 推理设备

默认启动方式会检测 `nvidia-smi`，读取 NVIDIA 驱动支持的 CUDA 版本，并自动选择与 PyTorch 2.5.1 匹配的 `cu118`、`cu121` 或 `cu124` 运行时。没有 NVIDIA GPU、驱动不可用或版本不足时，会自动安装 CPU 运行时。无需单独安装 CUDA Toolkit，但 NVIDIA GPU 必须已安装可用驱动。

CUDA 运行时首次下载约 2.5 GB，安装时建议预留至少 8 GB 磁盘空间。脚本会使用支持断点续传的下载方式；网络中断后，重新执行同一条启动命令即可继续，不会重新下载已完成的部分。下载和安装的运行时只保存在目标电脑的 `backend/.venv/`，不会包含在本项目交付文件中。磁盘空间不足或不需要 GPU 时，使用 CPU 参数启动。

需要强制使用 CPU 或 CUDA 时：

```powershell
.\start-dev.ps1 -InferenceRuntime Cpu
.\start-dev.ps1 -InferenceRuntime Cuda
```

## 运行后生成的内容

首次运行会在 `backend/storage/`、`backend/uploads/`、`backend/output_images/`、`backend/output_videos/` 和 `backend/logs/` 中生成数据。这些均属于本地运行记录，不是源码的一部分，也不应作为干净交付包内容。

完整部署说明见 `docs/DEPLOYMENT.md`。

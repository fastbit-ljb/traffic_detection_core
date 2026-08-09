# 部署说明

## 1. 运行形态

系统由 React/Vite 前端和 FastAPI 后端构成。后端启动后会提供 REST API、WebSocket、OpenAPI 文档以及标注图片/视频静态资源；前端默认请求 `http://127.0.0.1:8000`。

| 服务 | 默认地址 | 用途 |
|---|---|---|
| 前端 | `http://127.0.0.1:5173` | 操作与可视化界面 |
| 后端 API | `http://127.0.0.1:8000` | 检测、视频、训练、历史管理 |
| OpenAPI | `http://127.0.0.1:8000/docs` | 自动接口文档与调试页面 |
| 健康检查 | `http://127.0.0.1:8000/health` | 服务与资源状态 |

## 2. 本地部署（Windows PowerShell）

### 2.1 前置条件

- Python 3.11+，本项目实际实验使用 Python 3.12.10。
- Node.js 18+ 与 npm。
- 可选：NVIDIA GPU 与正确安装的 NVIDIA 驱动。无需另行安装 CUDA Toolkit；启动脚本会根据驱动版本下载匹配的 PyTorch CUDA 运行时。CUDA 首次下载约 2.5 GB，建议预留至少 8 GB 磁盘空间；下载中断后重新运行脚本即可断点续传。
- 模型文件：干净部署包预置 4 个可切换模型：`backend/yolov8n.pt`、`backend/models/yolov8s.pt`、`backend/models/yolov8m.pt` 三个官方预训练权重，以及 `backend/models/trained_28068449.pt` 交通车辆自训练样例。首次启动会自动注册它们，默认激活 YOLOv8n；也可通过界面或模型 API 上传并激活其他 `.pt` 文件。

### 2.2 配置环境变量

在项目根目录复制配置文件，并至少修改生产环境的 CORS 与 JWT 密钥：

```powershell
Copy-Item .env.example .env
```

开发环境的关键项示例：

```dotenv
ENVIRONMENT=development
TRAFFIC_DEBUG_MODE=true
TRAFFIC_ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
TRAFFIC_MODEL_NAME=yolov8n.pt
TRAFFIC_DETECTION_CONFIDENCE_THRESHOLD=0.40
TRAFFIC_ENABLE_GPU_ACCELERATION=true
VITE_API_BASE_URL=http://127.0.0.1:8000
```

生产环境必须设置长度至少 32 位的 `TRAFFIC_JWT_SECRET_KEY`，并将 `TRAFFIC_ALLOWED_ORIGINS` 替换为真实前端域名；不要保留示例中的占位值。

### 2.3 启动后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

若检测到 CUDA，`GET /api/inference-device` 会返回 GPU 名称。若显存不足，可在前端切换至 CPU，或调用 `PUT /api/inference-device` 将 `device` 设为 `cpu`。

### 2.4 启动前端

打开第二个 PowerShell：

```powershell
cd frontend
npm ci
$env:VITE_API_BASE_URL='http://127.0.0.1:8000'
npm run dev -- --host 0.0.0.0 --port 5173
```

浏览器访问 `http://127.0.0.1:5173`。前端发布构建命令为：

```powershell
npm run build
```

## 3. Docker 开发部署

Docker 方式用于开发联调：

```bash
docker compose --profile dev up --build
```

访问前端 `http://localhost:3000`，后端 `http://localhost:8000/docs`。该模式会把 `backend/`、`uploads/`、`output_images/` 和日志挂载到宿主机，便于检查任务结果。Docker 默认使用 CPU 推理；GPU 容器部署需额外配置 NVIDIA Container Toolkit 及 `--gpus all`，并使用与宿主驱动兼容的 CUDA PyTorch 镜像。

## 4. 持久化目录与备份

| 目录 | 内容 | 备份建议 |
|---|---|---|
| `backend/storage/traffic_detection.sqlite3` | 数据集、模型、任务、历史记录元数据 | 定期复制或使用 SQLite 在线备份 |
| `backend/output_images/` | 原图副本与标注图片 | 按保留期备份；历史删除会清理关联文件 |
| `backend/output_videos/` | 视频任务输出 | 按容量和保留期轮换 |
| `backend/models/` | 上传或训练得到的权重 | 单独备份并记录训练配置 |
| `backend/logs/` | 应用日志 | 设置日志轮换与告警 |

在 Linux 生产部署时，应将上述目录映射到持久卷，确保容器重建后任务和历史记录不丢失。

## 5. 生产上线检查

1. 使用反向代理终止 HTTPS，前端通过同域或固定 API 域名访问后端。
2. 关闭调试模式，限制 `TRAFFIC_ALLOWED_ORIGINS` 和可信主机。
3. 设置强 JWT 密钥，不将 `.env`、训练数据或模型上传到公开仓库。
4. 在目标设备上重新测量模型的速度、CPU/RSS、GPU 显存和视频端到端吞吐。
5. 配置磁盘配额、备份与日志轮换，防止视频输出和历史媒体耗尽磁盘。
6. 通过 `/health`、`/metrics` 与 `/api/system/info` 接入服务监控。

## 6. 常见问题

| 现象 | 排查方式 |
|---|---|
| 前端显示网络错误 | 确认后端在 8000 端口运行、`VITE_API_BASE_URL` 正确且 CORS 包含前端地址 |
| 任务状态一直失败 | 打开后端日志，检查模型文件、视频编解码器和可用磁盘空间 |
| 没有 GPU | 使用 `GET /api/inference-device`；CUDA 不可用时切换 CPU 或安装匹配的 PyTorch/CUDA |
| 视频无法在浏览器播放 | 检查 FFmpeg 是否可用；后端会将输出转换为浏览器兼容 MP4 |
| 历史删除后仍有图片 | 确认使用最新后端版本；删除逻辑只会清理应用生成目录下与历史记录关联的文件 |

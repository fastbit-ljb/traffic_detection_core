# API 文档

## 1. 约定

- 基础地址：`http://127.0.0.1:8000`。
- JSON 接口使用 `application/json`；图片、视频、模型和数据集上传使用 `multipart/form-data`。
- 成功响应为 200/202；参数或格式错误为 400/413；资源不存在为 404；依赖服务不可用为 503。
- 在线交互式文档：`GET /docs`；OpenAPI JSON：`GET /openapi.json`。

## 2. 接口总览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务、模型、统计服务和主机资源健康状态 |
| GET | `/api/system/info` | 应用版本、功能开关和默认模型信息 |
| GET | `/api/inference-device` | 当前推理设备状态 |
| PUT | `/api/inference-device` | 切换 `auto` / `cpu` / `cuda` |
| POST | `/api/detect-vehicles` | 图片检测及可选历史保存、基准线统计 |
| POST | `/api/detect-video` | 创建视频后台检测任务 |
| GET | `/api/video-jobs/{job_id}` | 查询单个视频任务 |
| GET | `/api/history` | 查询检测历史 |
| DELETE | `/api/history/{history_id}` | 删除单条历史记录及关联应用媒体 |
| DELETE | `/api/history` | 批量删除选中历史记录 |
| GET | `/api/datasets` | 查询已导入数据集 |
| POST | `/api/datasets/import` | 导入 YOLO 数据集 ZIP |
| GET | `/api/models` | 查询模型库 |
| POST | `/api/models/upload` | 上传 `.pt` 模型 |
| POST | `/api/models/{model_id}/activate` | 激活模型 |
| GET / POST | `/api/training/jobs` | 查询活动训练任务 / 创建训练任务 |
| GET | `/api/training/runs` | 查询最近已完成或失败的训练运行 |
| GET | `/api/experiments` | 查询持久化模型验证实验 |
| POST | `/api/experiments/compare` | 用同一验证集创建模型对比任务 |
| GET | `/api/intersection-status` | 当前信号与路口状态 |
| POST | `/api/emergency-override` | 触发紧急车辆优先控制 |
| GET | `/api/analytics/summary` | 交通分析摘要 |
| GET | `/metrics` | Prometheus 指标 |
| WS | `/ws/traffic-updates` | 实时状态、检测结果和视频任务推送 |

## 3. 主要接口

### 3.1 图片检测

`POST /api/detect-vehicles`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `image` | File | 是 | JPG/JPEG/PNG/BMP，最大 10 MB |
| `source` | string | 否 | `image` 或 `camera`，默认 `image` |
| `record_history` | boolean | 否 | 是否保存历史，默认 `true` |
| `baseline_enabled` | boolean | 否 | 是否启用基准线，默认 `false` |
| `baseline_orientation` | string | 否 | `horizontal` 或 `vertical` |
| `baseline_direction` | string | 否 | 水平线使用 `up/down`，垂直线使用 `left/right` |
| `baseline_position` | number | 否 | 归一化位置，范围 0.05-0.95 |
| `baseline_session` | string | 否 | 实时基准线统计会话标识 |

```bash
curl -X POST http://127.0.0.1:8000/api/detect-vehicles \
  -F "image=@test_images/1.jpg" \
  -F "record_history=true" \
  -F "baseline_enabled=true" \
  -F "baseline_orientation=horizontal" \
  -F "baseline_direction=down" \
  -F "baseline_position=0.50"
```

响应核心字段：

```json
{
  "total_vehicles": 4,
  "class_counts": {"person": 1, "car": 2, "bus": 0, "truck": 1},
  "flow_counts": {
    "entry": {"person": 0, "car": 0, "bus": 0, "truck": 0},
    "exit": {"person": 0, "car": 0, "bus": 0, "truck": 0}
  },
  "lane_counts": {"north": 1, "south": 1, "east": 0, "west": 1},
  "detected_vehicles": [],
  "processing_time": 0.02,
  "annotated_image_path": "output_images/traffic_xxx_annotated.jpg"
}
```

### 3.2 视频检测任务

`POST /api/detect-video` 返回 202 和任务对象。视频可使用与图片相同的基准线参数。

```bash
curl -X POST http://127.0.0.1:8000/api/detect-video \
  -F "video=@test_samples/ny_traffic.mp4" \
  -F "baseline_enabled=true" \
  -F "baseline_orientation=horizontal" \
  -F "baseline_direction=down" \
  -F "baseline_position=0.50"
```

随后查询：

```bash
curl http://127.0.0.1:8000/api/video-jobs/{job_id}
```

任务 `status` 依次为 `queued`、`running`、`completed` 或 `failed`。完成结果包含 `frames_processed`、`class_counts`、`flow_counts`、`processing_time` 和 `output_path`。

### 3.3 历史记录与批量删除

```bash
# 获取最近 50 条
curl "http://127.0.0.1:8000/api/history?limit=50"

# 删除一条
curl -X DELETE http://127.0.0.1:8000/api/history/{history_id}

# 批量删除
curl -X DELETE http://127.0.0.1:8000/api/history \
  -H "Content-Type: application/json" \
  -d '{"ids":["id-1","id-2"]}'
```

批量删除响应为 `{"deleted_ids":["id-1","id-2"]}`。后端只删除应用 `output_images/` 或 `output_videos/` 下与这些记录关联的媒体文件。

### 3.4 模型和推理设备

```bash
# 查看模型
curl http://127.0.0.1:8000/api/models

# 上传模型
curl -X POST http://127.0.0.1:8000/api/models/upload \
  -F "model_file=@experiments/results/coco2017_yolov8/runs/yolov8n/weights/best.pt" \
  -F "model_name=COCO交通子集-YOLOv8n"

# 激活模型
curl -X POST http://127.0.0.1:8000/api/models/{model_id}/activate

# 切换设备
curl -X PUT http://127.0.0.1:8000/api/inference-device \
  -H "Content-Type: application/json" \
  -d '{"device":"cuda"}'
```

### 3.5 训练和模型对比

```json
POST /api/training/jobs
{
  "dataset_id": "dataset-id",
  "base_model_id": "model-id",
  "epochs": 30,
  "batch": 8,
  "imgsz": 640
}
```

```json
POST /api/experiments/compare
{
  "dataset_id": "dataset-id",
  "model_ids": ["model-a", "model-b", "model-c"],
  "batch": 8,
  "imgsz": 640
}
```

两个接口都立即返回 202 和任务 ID；训练和验证结果可通过任务查询接口或前端数据管理页查看。

### 3.6 WebSocket 事件

连接地址：`ws://127.0.0.1:8000/ws/traffic-updates`。

| 事件类型 | 含义 |
|---|---|
| `intersection_status` | 每 2 秒推送的路口信号状态 |
| `vehicle_detection` | 图片/实时检测完成后的结果 |
| `video_progress` | 视频已处理帧数、分类统计和进出统计 |
| `video_completed` | 视频任务完成，包含最终任务结果 |
| `video_failed` | 视频任务失败，包含错误信息 |
| `model_changed` | 当前激活模型已切换 |
| `inference_device_changed` | 推理设备已切换 |
| `emergency_alert` | 紧急车辆优先事件 |

## 4. 典型错误

| 状态码 | 场景 | 处理建议 |
|---:|---|---|
| 400 | 文件类型、模型后缀、基准线配置错误 | 按接口字段和格式重新提交 |
| 413 | 图片、视频或数据集超过限制 | 压缩、裁剪或分片处理输入 |
| 404 | 模型、视频任务或历史记录不存在 | 刷新列表后使用有效 ID |
| 503 | 检测、视频、训练或数据服务未就绪 | 检查模型、后端启动日志及 GPU/CPU 资源 |
| 500 | 推理、转码或内部任务失败 | 查看后端日志，并保留任务 ID 以便定位 |

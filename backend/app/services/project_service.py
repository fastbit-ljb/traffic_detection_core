"""Persistent project services for datasets, models, training, video jobs and history."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ultralytics import YOLO

from .intelligent_vehicle_detector import IntelligentVehicleDetector


TARGET_NAMES = ["person", "car", "bus", "truck"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OFFICIAL_DETECTION_MODELS = (
    ("YOLOv8n 官方预训练（速度优先）", "yolov8n.pt"),
    ("YOLOv8s 官方预训练（均衡）", "yolov8s.pt"),
    ("YOLOv8m 官方预训练（精度优先）", "yolov8m.pt"),
)
DEPLOYMENT_CUSTOM_MODELS = (
    ("交通车辆自训练样例（1 Epoch）", "trained_28068449.pt"),
)


def _as_float(value: Any) -> Optional[float]:
    """Return a finite metric value without leaking library-specific number types."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _normalize_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Normalize Ultralytics metric keys from a validation result or results.csv."""
    aliases = {
        "precision": ("metrics/precision(B)", "precision", "metrics/precision"),
        "recall": ("metrics/recall(B)", "recall", "metrics/recall"),
        "map50": ("metrics/mAP50(B)", "map50", "metrics/mAP50"),
        "map50_95": ("metrics/mAP50-95(B)", "map50_95", "metrics/mAP50-95"),
    }
    normalized: Dict[str, Optional[float]] = {}
    for target_key, candidates in aliases.items():
        normalized[target_key] = next(
            (number for candidate in candidates if (number := _as_float(raw_metrics.get(candidate))) is not None),
            None,
        )

    precision = normalized["precision"]
    recall = normalized["recall"]
    normalized["f1"] = (
        (2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return normalized


def _read_training_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    """Read the final validation row emitted by Ultralytics training."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return _normalize_metrics({})
    with results_csv.open("r", encoding="utf-8-sig", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        return _normalize_metrics({})
    return _normalize_metrics({key.strip(): value for key, value in rows[-1].items() if key})


def _extract_validation_metrics(validation_result: Any, elapsed_seconds: float) -> Dict[str, Optional[float]]:
    """Extract comparable detection metrics from an Ultralytics ``model.val`` result."""
    raw_metrics = getattr(validation_result, "results_dict", {})
    metrics = _normalize_metrics(raw_metrics if isinstance(raw_metrics, dict) else {})
    speed = getattr(validation_result, "speed", {})
    metrics["inference_ms"] = _as_float(speed.get("inference")) if isinstance(speed, dict) else None
    metrics["evaluation_seconds"] = round(elapsed_seconds, 3)
    return metrics


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectRepository:
    """SQLite-backed records that make results survive an API restart."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.database_path = storage_dir / "traffic_detection.sqlite3"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT 'base',
                    parent_model_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS detection_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    media_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    class_counts_json TEXT NOT NULL,
                    total_objects INTEGER NOT NULL,
                    processing_time REAL NOT NULL,
                    output_path TEXT,
                    original_path TEXT,
                    model_name TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            history_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(detection_history)").fetchall()
            }
            if "original_path" not in history_columns:
                connection.execute("ALTER TABLE detection_history ADD COLUMN original_path TEXT")
            if "user_id" not in history_columns:
                connection.execute("ALTER TABLE detection_history ADD COLUMN user_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_history_user_created "
                "ON detection_history(user_id, created_at DESC)"
            )
            model_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(models)").fetchall()
            }
            if "version" not in model_columns:
                connection.execute("ALTER TABLE models ADD COLUMN version TEXT NOT NULL DEFAULT 'base'")
            if "parent_model_id" not in model_columns:
                connection.execute("ALTER TABLE models ADD COLUMN parent_model_id TEXT")
            if "metadata_json" not in model_columns:
                connection.execute("ALTER TABLE models ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
            connection.execute(
                "UPDATE jobs SET status = 'failed', message = '服务重启前任务已中断', updated_at = ? "
                "WHERE status IN ('queued', 'running')",
                (utc_now(),),
            )

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _as_dict(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for key in ("summary_json", "class_counts_json", "payload_json", "result_json"):
            if key in item:
                raw_value = item.pop(key)
                item[key.removesuffix("_json")] = json.loads(raw_value) if raw_value is not None else None
        if "class_counts" in item:
            item["class_counts"] = {
                key.removeprefix("VehicleType.").lower(): value
                for key, value in item["class_counts"].items()
            }
        if "is_active" in item:
            item["is_active"] = bool(item["is_active"])
        return item

    def create_user(self, username: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """Create one account, returning None when the username already exists."""
        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "password_hash": password_hash,
            "created_at": utc_now(),
            "is_active": True,
        }
        try:
            with self._lock, self._connection() as connection:
                connection.execute(
                    "INSERT INTO users (id, username, password_hash, created_at, is_active) "
                    "VALUES (?, ?, ?, ?, 1)",
                    (user["id"], user["username"], user["password_hash"], user["created_at"]),
                )
        except sqlite3.IntegrityError:
            return None
        return user

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def add_dataset(self, name: str, path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
        dataset = {
            "id": str(uuid.uuid4()),
            "name": name,
            "path": str(path.resolve()),
            "summary": summary,
            "created_at": utc_now(),
        }
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?)",
                (dataset["id"], dataset["name"], dataset["path"], json.dumps(summary), dataset["created_at"]),
            )
        return dataset

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._as_dict(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return self._as_dict(row) if row else None

    def register_model(self, name: str, path: Path, source: str, activate: bool = False) -> Dict[str, Any]:
        resolved_path = str(path.resolve())
        with self._lock, self._connection() as connection:
            existing = connection.execute("SELECT * FROM models WHERE path = ?", (resolved_path,)).fetchone()
            if existing:
                model = self._as_dict(existing)
                if activate:
                    connection.execute("UPDATE models SET is_active = 0")
                    connection.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model["id"],))
                    model["is_active"] = True
                return model

            model = {
                "id": str(uuid.uuid4()),
                "name": name,
                "path": resolved_path,
                "source": source,
                "is_active": activate,
                "created_at": utc_now(),
            }
            if activate:
                connection.execute("UPDATE models SET is_active = 0")
            connection.execute(
                "INSERT INTO models (id, name, path, source, is_active, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (model["id"], model["name"], model["path"], model["source"], int(activate), model["created_at"]),
            )
        return model

    def register_available_official_models(self, *directories: Path) -> List[Dict[str, Any]]:
        """Register downloaded official YOLOv8 detection weights for model switching."""
        registered_models = []
        for name, filename in OFFICIAL_DETECTION_MODELS:
            model_path = next(
                (directory / filename for directory in directories if (directory / filename).is_file()),
                None,
            )
            if model_path and model_path.stat().st_size > 0:
                registered_models.append(self.register_model(name, model_path, "官方预训练"))
        return registered_models

    def register_available_custom_models(self, *directories: Path) -> List[Dict[str, Any]]:
        """Register the representative traffic model packaged with a clean deployment."""
        registered_models = []
        for name, filename in DEPLOYMENT_CUSTOM_MODELS:
            model_path = next(
                (directory / filename for directory in directories if (directory / filename).is_file()),
                None,
            )
            if model_path and model_path.stat().st_size > 0:
                registered_models.append(self.register_model(name, model_path, "自训练样例"))
        return registered_models

    def list_models(self) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT * FROM models ORDER BY is_active DESC, created_at DESC").fetchall()
        return [self._as_dict(row) for row in rows]

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._as_dict(row) if row else None

    def activate_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not row:
                return None
            connection.execute("UPDATE models SET is_active = 0")
            connection.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        model = self._as_dict(row)
        model["is_active"] = True
        return model

    def add_history(
        self,
        media_type: str,
        source_name: str,
        class_counts: Dict[str, int],
        total_objects: int,
        processing_time: float,
        output_path: Optional[str],
        model_name: Optional[str],
        original_path: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "media_type": media_type,
            "source_name": source_name,
            "class_counts": class_counts,
            "total_objects": total_objects,
            "processing_time": processing_time,
            "output_path": output_path,
            "original_path": original_path,
            "model_name": model_name,
            "created_at": utc_now(),
        }
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO detection_history (
                    id, user_id, media_type, source_name, class_counts_json, total_objects,
                    processing_time, output_path, original_path, model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item["id"], item["user_id"], item["media_type"], item["source_name"], json.dumps(item["class_counts"]),
                    item["total_objects"], item["processing_time"], item["output_path"], item["original_path"], item["model_name"], item["created_at"],
                ),
            )
        return item

    def list_history(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            if user_id is None:
                rows = connection.execute(
                    "SELECT * FROM detection_history ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM detection_history WHERE user_id = ? "
                    "ORDER BY created_at DESC LIMIT ?", (user_id, limit)
                ).fetchall()
        return [self._as_dict(row) for row in rows]

    @staticmethod
    def _remove_history_assets(entry: Dict[str, Any]) -> None:
        """Remove only media generated by this application for a deleted record."""
        managed_roots = [Path("./output_images").resolve(), Path("./output_videos").resolve()]
        for key in ("original_path", "output_path"):
            raw_path = entry.get(key)
            if not raw_path:
                continue
            try:
                candidate = Path(raw_path).resolve()
                if any(candidate.is_relative_to(root) for root in managed_roots):
                    candidate.unlink(missing_ok=True)
            except OSError:
                # A missing, locked, or externally removed artifact must not make
                # the database record undeletable.
                continue

    def delete_history(self, history_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Delete one persisted detection record and return the removed entry."""
        with self._lock, self._connection() as connection:
            if user_id is None:
                row = connection.execute(
                    "SELECT * FROM detection_history WHERE id = ?", (history_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM detection_history WHERE id = ? AND user_id = ?", (history_id, user_id)
                ).fetchone()
            if not row:
                return None
            if user_id is None:
                connection.execute("DELETE FROM detection_history WHERE id = ?", (history_id,))
            else:
                connection.execute(
                    "DELETE FROM detection_history WHERE id = ? AND user_id = ?", (history_id, user_id)
                )
        entry = self._as_dict(row)
        self._remove_history_assets(entry)
        return entry

    def delete_history_entries(self, history_ids: List[str], user_id: Optional[str] = None) -> List[str]:
        """Delete persisted detection records in one transaction."""
        unique_history_ids = list(dict.fromkeys(history_id for history_id in history_ids if history_id))
        if not unique_history_ids:
            return []

        placeholders = ", ".join("?" for _ in unique_history_ids)
        with self._lock, self._connection() as connection:
            query = f"SELECT * FROM detection_history WHERE id IN ({placeholders})"
            query_params: List[Any] = unique_history_ids
            if user_id is not None:
                query += " AND user_id = ?"
                query_params.append(user_id)
            rows = connection.execute(query, query_params).fetchall()
            entries_by_id = {row["id"]: self._as_dict(row) for row in rows}
            deleted_ids = set(entries_by_id)
            if deleted_ids:
                delete_query = f"DELETE FROM detection_history WHERE id IN ({placeholders})"
                delete_params: List[Any] = unique_history_ids
                if user_id is not None:
                    delete_query += " AND user_id = ?"
                    delete_params.append(user_id)
                connection.execute(delete_query, delete_params)
        for history_id in unique_history_ids:
            entry = entries_by_id.get(history_id)
            if entry:
                self._remove_history_assets(entry)
        return [history_id for history_id in unique_history_ids if history_id in deleted_ids]

    def create_job(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "任务已排队",
            "payload": payload,
            "result": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job["id"], job["kind"], job["status"], job["progress"], job["message"],
                    json.dumps(job["payload"]), None, job["created_at"], job["updated_at"],
                ),
            )
        return job

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        current = self.get_job(job_id)
        if not current:
            return None
        next_status = status or current["status"]
        next_progress = current["progress"] if progress is None else max(0.0, min(100.0, progress))
        next_message = message or current["message"]
        next_result = current["result"] if result is None else result
        updated_at = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, progress = ?, message = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (next_status, next_progress, next_message, json.dumps(next_result) if next_result else None, updated_at, job_id),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._as_dict(row) if row else None

    def list_jobs(
        self,
        kind: Optional[str] = None,
        limit: int = 30,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM jobs"
        conditions: List[str] = []
        params: List[Any] = []
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if active_only:
            conditions.append("status IN ('queued', 'running')")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._as_dict(row) for row in rows]


class DatasetService:
    """Imports a ZIP archive and normalizes it into a four-class YOLO dataset."""

    def __init__(self, repository: ProjectRepository):
        self.repository = repository
        self.datasets_dir = repository.storage_dir / "datasets"

    async def import_archive(self, content: bytes, original_name: str, dataset_name: str) -> Dict[str, Any]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._import_archive_sync, content, original_name, dataset_name
        )

    def _import_archive_sync(self, content: bytes, original_name: str, dataset_name: str) -> Dict[str, Any]:
        if not original_name.lower().endswith(".zip"):
            raise ValueError("数据集必须是 ZIP 压缩包")

        upload_id = str(uuid.uuid4())
        workspace = self.datasets_dir / upload_id
        archive_path = workspace / "upload.zip"
        extracted_dir = workspace / "extracted"
        prepared_dir = workspace / "prepared"
        workspace.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(content)

        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (extracted_dir / member.filename).resolve()
                    if not target.is_relative_to(extracted_dir.resolve()):
                        raise ValueError("压缩包包含非法路径")
                archive.extractall(extracted_dir)
        except zipfile.BadZipFile as error:
            raise ValueError("无法读取数据集压缩包") from error

        source_root = self._find_source_root(extracted_dir)
        self._normalize_dataset(source_root, prepared_dir)
        summary = self._validate_and_write_yaml(prepared_dir)
        return self.repository.add_dataset(dataset_name.strip() or Path(original_name).stem, prepared_dir, summary)

    @staticmethod
    def _find_source_root(extracted_dir: Path) -> Path:
        for candidate in [extracted_dir, *[path for path in extracted_dir.iterdir() if path.is_dir()]]:
            if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
                return candidate
        raise ValueError("压缩包必须包含 images/ 与 labels/ 目录")

    def _normalize_dataset(self, source_root: Path, prepared_dir: Path) -> None:
        images_root = source_root / "images"
        labels_root = source_root / "labels"
        prepared_dir.mkdir(parents=True, exist_ok=True)

        if (images_root / "train").is_dir():
            for split in ("train", "val"):
                source_images = images_root / split
                source_labels = labels_root / split
                if not source_images.is_dir() or not source_labels.is_dir():
                    raise ValueError("标准 YOLO 数据集必须同时包含 train/val 的 images 与 labels")
                shutil.copytree(source_images, prepared_dir / "images" / split, dirs_exist_ok=True)
                shutil.copytree(source_labels, prepared_dir / "labels" / split, dirs_exist_ok=True)
            return

        image_files = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(image_files) < 2:
            raise ValueError("至少需要两张带标注图片以划分训练集和验证集")

        validated_pairs = []
        for image_path in image_files:
            relative_path = image_path.relative_to(images_root)
            label_path = (labels_root / relative_path).with_suffix(".txt")
            if label_path.exists():
                validated_pairs.append((image_path, label_path, relative_path.name))
        if len(validated_pairs) < 2:
            raise ValueError("未找到与图片对应的 YOLO 标签文件")

        split_index = max(1, int(len(validated_pairs) * 0.8))
        for index, (image_path, label_path, file_name) in enumerate(validated_pairs):
            split = "train" if index < split_index else "val"
            image_destination = prepared_dir / "images" / split / file_name
            label_destination = prepared_dir / "labels" / split / f"{Path(file_name).stem}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_destination)
            shutil.copy2(label_path, label_destination)

    @staticmethod
    def _validate_and_write_yaml(prepared_dir: Path) -> Dict[str, Any]:
        class_counts = {name: 0 for name in TARGET_NAMES}
        split_summary: Dict[str, int] = {}
        for split in ("train", "val"):
            image_dir = prepared_dir / "images" / split
            label_dir = prepared_dir / "labels" / split
            images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                raise ValueError(f"{split} 集没有可用图片")
            split_summary[split] = len(images)
            for label_path in label_dir.rglob("*.txt"):
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    # YOLO permits empty label files for images with no target objects.
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"标签格式错误: {label_path.name}:{line_number}")
                    try:
                        class_id = int(fields[0])
                        [float(value) for value in fields[1:]]
                    except ValueError as error:
                        raise ValueError(f"标签数值错误: {label_path.name}:{line_number}") from error
                    if class_id not in range(len(TARGET_NAMES)):
                        raise ValueError("标签类别必须是 0:person、1:car、2:bus、3:truck")
                    class_counts[TARGET_NAMES[class_id]] += 1

        yaml_path = prepared_dir / "data.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {prepared_dir.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "names:",
                    "  0: person",
                    "  1: car",
                    "  2: bus",
                    "  3: truck",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "images": split_summary,
            "labels_by_class": class_counts,
            "yaml_path": str(yaml_path.resolve()),
            "classes": TARGET_NAMES,
        }


class TrainingService:
    """Runs reproducible training and validation jobs without competing for the accelerator."""

    def __init__(self, repository: ProjectRepository, models_dir: Path):
        self.repository = repository
        self.models_dir = models_dir
        self.runs_dir = repository.storage_dir / "training_runs"
        self._work_lock = asyncio.Lock()

    def submit(self, dataset: Dict[str, Any], base_model: Dict[str, Any], config: Dict[str, int]) -> Dict[str, Any]:
        job = self.repository.create_job(
            "training",
            {
                "dataset_id": dataset["id"],
                "dataset_name": dataset["name"],
                "base_model_id": base_model["id"],
                "base_model_name": base_model["name"],
                "config": config,
            },
        )
        asyncio.create_task(self._run(job["id"], dataset, base_model, config))
        return job

    async def _run(
        self, job_id: str, dataset: Dict[str, Any], base_model: Dict[str, Any], config: Dict[str, int]
    ) -> None:
        async with self._work_lock:
            self.repository.update_job(job_id, status="running", progress=5, message="正在初始化训练")
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self._train_sync, job_id, dataset, base_model, config
                )
                trained_path = Path(result["best_weights"])
                self.models_dir.mkdir(parents=True, exist_ok=True)
                destination = self.models_dir / f"trained_{job_id[:8]}.pt"
                shutil.copy2(trained_path, destination)
                model = self.repository.register_model(
                    name=f"{dataset['name']}训练-{job_id[:4]}", path=destination, source="training", activate=False
                )
                self.repository.update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    message="训练完成，最佳权重已注册",
                    result={"model": model, **result},
                )
            except Exception as error:
                self.repository.update_job(job_id, status="failed", message=str(error), result={"error": str(error)})

    def _train_sync(
        self, job_id: str, dataset: Dict[str, Any], base_model: Dict[str, Any], config: Dict[str, int]
    ) -> Dict[str, Any]:
        data_yaml = Path(dataset["summary"]["yaml_path"])
        if not data_yaml.exists():
            raise FileNotFoundError("数据集配置文件不存在")
        if not Path(base_model["path"]).exists():
            raise FileNotFoundError("基础模型权重不存在")

        model = YOLO(base_model["path"])
        self.repository.update_job(job_id, progress=15, message="YOLOv8 正在训练")
        model.train(
            data=str(data_yaml),
            epochs=config["epochs"],
            batch=config["batch"],
            imgsz=config["imgsz"],
            project=str(self.runs_dir.resolve()),
            name=job_id,
            exist_ok=True,
            workers=0,
        )
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError("训练结束后未找到 best.pt")
        run_dir = Path(model.trainer.save_dir)
        return {
            "best_weights": str(best_weights.resolve()),
            "run_dir": str(run_dir.resolve()),
            "metrics": _read_training_metrics(run_dir),
        }

    def submit_comparison(
        self, dataset: Dict[str, Any], models: List[Dict[str, Any]], config: Dict[str, int]
    ) -> Dict[str, Any]:
        """Queue validation of multiple weights against the same held-out split."""
        job = self.repository.create_job(
            "benchmark",
            {
                "dataset_id": dataset["id"],
                "dataset_name": dataset["name"],
                "model_ids": [model["id"] for model in models],
                "model_names": [model["name"] for model in models],
                "config": config,
            },
        )
        asyncio.create_task(self._run_comparison(job["id"], dataset, models, config))
        return job

    async def _run_comparison(
        self,
        job_id: str,
        dataset: Dict[str, Any],
        models: List[Dict[str, Any]],
        config: Dict[str, int],
    ) -> None:
        async with self._work_lock:
            self.repository.update_job(job_id, status="running", progress=5, message="正在准备验证集对比")
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self._compare_sync, job_id, dataset, models, config
                )
                completed_count = sum(1 for item in result["items"] if "error" not in item)
                if not completed_count:
                    raise RuntimeError("所有模型验证均未完成")
                self.repository.update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    message=f"对比完成，{completed_count} 个模型已完成验证",
                    result=result,
                )
            except Exception as error:
                self.repository.update_job(job_id, status="failed", message=str(error), result={"error": str(error)})

    def _compare_sync(
        self,
        job_id: str,
        dataset: Dict[str, Any],
        models: List[Dict[str, Any]],
        config: Dict[str, int],
    ) -> Dict[str, Any]:
        data_yaml = Path(dataset["summary"]["yaml_path"])
        if not data_yaml.exists():
            raise FileNotFoundError("数据集配置文件不存在")

        evaluation_dir = self.runs_dir / "comparisons"
        items: List[Dict[str, Any]] = []
        total = len(models)
        for index, model_info in enumerate(models):
            model_path = Path(model_info["path"])
            progress = 10 + (index / total * 80)
            self.repository.update_job(
                job_id,
                progress=progress,
                message=f"正在验证 {index + 1}/{total}：{model_info['name']}",
            )
            if not model_path.exists():
                items.append({
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "source": model_info["source"],
                    "error": "模型权重不存在",
                })
                continue

            try:
                started_at = time.perf_counter()
                model = YOLO(model_path)
                validation_result = model.val(
                    data=str(data_yaml),
                    split="val",
                    imgsz=config["imgsz"],
                    batch=config["batch"],
                    project=str(evaluation_dir.resolve()),
                    name=f"{job_id[:8]}_{index + 1}",
                    exist_ok=True,
                    workers=0,
                    plots=True,
                    verbose=False,
                )
                elapsed_seconds = time.perf_counter() - started_at
                run_dir = getattr(validation_result, "save_dir", evaluation_dir / f"{job_id[:8]}_{index + 1}")
                items.append({
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "source": model_info["source"],
                    "run_dir": str(Path(run_dir).resolve()),
                    "metrics": _extract_validation_metrics(validation_result, elapsed_seconds),
                })
            except Exception as error:
                items.append({
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "source": model_info["source"],
                    "error": str(error),
                })

        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "items": items,
        }


ProgressCallback = Callable[[int, int, Dict[str, int], Dict[str, Dict[str, int]]], Awaitable[None]]
BroadcastCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class VideoProcessingService:
    """Runs video detection in a background task and persists the result."""

    def __init__(
        self,
        repository: ProjectRepository,
        detector: IntelligentVehicleDetector,
        output_dir: Path,
        broadcast: BroadcastCallback,
    ):
        self.repository = repository
        self.detector = detector
        self.output_dir = output_dir
        self.broadcast = broadcast

    def submit(
        self,
        source_path: Path,
        source_name: str,
        model_name: str,
        baseline_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"source_name": source_name, "model_name": model_name, "baseline_enabled": str(bool(baseline_config)).lower()}
        if baseline_config:
            payload.update({
                "baseline_orientation": str(baseline_config["orientation"]),
                "baseline_direction": str(baseline_config.get("direction", "")),
                "baseline_position": str(baseline_config["position"]),
            })
        job = self.repository.create_job("video", payload)
        asyncio.create_task(self._run(job["id"], source_path, source_name, model_name, baseline_config, user_id))
        return job

    async def _run(
        self,
        job_id: str,
        source_path: Path,
        source_name: str,
        model_name: str,
        baseline_config: Optional[Dict[str, Any]],
        user_id: Optional[str],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{job_id}.mp4"
        self.repository.update_job(job_id, status="running", progress=1, message="正在读取视频")

        async def report_progress(
            current: int,
            total: int,
            counts: Dict[str, int],
            flow_counts: Dict[str, Dict[str, int]],
        ) -> None:
            progress = (current / total * 100) if total else 0
            job = self.repository.update_job(
                job_id, status="running", progress=progress, message=f"已处理 {current}/{total} 帧"
            )
            await self.broadcast({"type": "video_progress", "data": {**(job or {}), "class_counts": counts, "flow_counts": flow_counts}})

        try:
            summary = await self.detector.analyze_video(
                source_path, output_path, report_progress, baseline_config=baseline_config
            )
            self.repository.add_history(
                media_type="video",
                source_name=source_name,
                class_counts=summary["class_counts"],
                total_objects=sum(summary["class_counts"].values()),
                processing_time=summary["processing_time"],
                output_path=str(output_path),
                model_name=model_name,
                original_path=None,
                user_id=user_id,
            )
            completed = self.repository.update_job(
                job_id,
                status="completed",
                progress=100,
                message="视频处理完成",
                result={**summary, "output_path": str(output_path)},
            )
            await self.broadcast({"type": "video_completed", "data": completed})
        except Exception as error:
            failed = self.repository.update_job(job_id, status="failed", message=str(error), result={"error": str(error)})
            await self.broadcast({"type": "video_failed", "data": failed})

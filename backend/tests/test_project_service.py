import io
import zipfile

import pytest
from PIL import Image
from fastapi import BackgroundTasks, UploadFile
from starlette.datastructures import Headers

from app import main
from app.services.project_service import DatasetService, ProjectRepository, _read_training_metrics


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), color=(32, 96, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_dataset_import_normalizes_raw_yolo_archive(tmp_path):
    archive_buffer = io.BytesIO()
    image = _image_bytes()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for name in ("one", "two", "three"):
            archive.writestr(f"road/images/{name}.jpg", image)
            archive.writestr(f"road/labels/{name}.txt", "1 0.5 0.5 0.2 0.2\n")

    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    dataset = DatasetService(repository)._import_archive_sync(
        archive_buffer.getvalue(), "road.zip", "道路测试集"
    )

    assert dataset["summary"]["images"] == {"train": 2, "val": 1}
    assert dataset["summary"]["labels_by_class"] == {
        "person": 0, "car": 3, "bus": 0, "truck": 0,
    }
    assert repository.get_dataset(dataset["id"])["name"] == "道路测试集"


def test_dataset_import_accepts_empty_yolo_label_files(tmp_path):
    archive_buffer = io.BytesIO()
    image = _image_bytes()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for name in ("one", "two", "three"):
            archive.writestr(f"road/images/{name}.jpg", image)
        archive.writestr("road/labels/one.txt", "0 0.5 0.5 0.2 0.2\n")
        archive.writestr("road/labels/two.txt", "\n")
        archive.writestr("road/labels/three.txt", "\n")

    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    dataset = DatasetService(repository)._import_archive_sync(
        archive_buffer.getvalue(), "empty-labels.zip", "空标签测试集"
    )

    assert dataset["summary"]["images"] == {"train": 2, "val": 1}
    assert dataset["summary"]["labels_by_class"] == {
        "person": 1, "car": 0, "bus": 0, "truck": 0,
    }


def test_job_status_updates_when_no_result_exists(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    job = repository.create_job("training", {"dataset_id": "dataset"})

    updated = repository.update_job(job["id"], status="running", progress=5, message="正在初始化")

    assert updated["status"] == "running"
    assert updated["progress"] == 5
    assert updated["result"] is None

    repository.update_job(job["id"], status="completed", progress=100, message="训练完成")
    assert repository.list_jobs(kind="training", active_only=True) == []


def test_history_normalizes_legacy_enum_keys(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    repository.add_history(
        "image", "road.jpg", {"VehicleType.CAR": 2, "VehicleType.PERSON": 1}, 3, 0.1,
        "annotated.jpg", "base", original_path="original.jpg",
    )

    history_entry = repository.list_history()[0]
    assert history_entry["class_counts"] == {"car": 2, "person": 1}
    assert history_entry["original_path"] == "original.jpg"


def test_delete_history_removes_only_the_requested_entry(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    retained = repository.add_history("image", "retained.jpg", {}, 0, 0.1, None, "base")
    deleted = repository.add_history("image", "deleted.jpg", {}, 0, 0.1, None, "base")

    removed = repository.delete_history(deleted["id"])

    assert removed is not None
    assert removed["id"] == deleted["id"]
    assert [entry["id"] for entry in repository.list_history()] == [retained["id"]]
    assert repository.delete_history(deleted["id"]) is None


def test_delete_history_entries_removes_selected_records_in_input_order(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    first = repository.add_history("image", "first.jpg", {}, 0, 0.1, None, "base")
    retained = repository.add_history("image", "retained.jpg", {}, 0, 0.1, None, "base")
    second = repository.add_history("image", "second.jpg", {}, 0, 0.1, None, "base")

    deleted_ids = repository.delete_history_entries([second["id"], first["id"], second["id"]])

    assert deleted_ids == [second["id"], first["id"]]
    assert [entry["id"] for entry in repository.list_history()] == [retained["id"]]


def test_delete_history_removes_only_managed_media_assets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    output_dir = tmp_path / "output_images"
    output_dir.mkdir()
    original = output_dir / "original.jpg"
    annotated = output_dir / "annotated.jpg"
    external = tmp_path / "external.jpg"
    original.write_bytes(b"original")
    annotated.write_bytes(b"annotated")
    external.write_bytes(b"external")

    entry = repository.add_history(
        "image", "road.jpg", {}, 0, 0.1, str(annotated), "base", str(original)
    )
    # Paths outside application-managed media roots are intentionally retained.
    repository.add_history("image", "external.jpg", {}, 0, 0.1, str(external), "base")

    repository.delete_history(entry["id"])

    assert not original.exists()
    assert not annotated.exists()
    assert external.exists()


@pytest.mark.asyncio
async def test_delete_detection_history_endpoint_returns_removed_id(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    entry = repository.add_history("image", "road.jpg", {}, 0, 0.1, None, "base")

    response = await main.delete_detection_history(entry["id"], repository)

    assert response == {"id": entry["id"]}
    assert repository.list_history() == []


@pytest.mark.asyncio
async def test_delete_detection_history_entries_endpoint_returns_removed_ids(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    first = repository.add_history("image", "first.jpg", {}, 0, 0.1, None, "base")
    second = repository.add_history("image", "second.jpg", {}, 0, 0.1, None, "base")

    response = await main.delete_detection_history_entries(
        main.HistoryDeletionRequest(ids=[second["id"], first["id"]]), repository
    )

    assert response == {"deleted_ids": [second["id"], first["id"]]}
    assert repository.list_history() == []


@pytest.mark.asyncio
async def test_delete_detection_history_entries_endpoint_is_idempotent(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()

    response = await main.delete_detection_history_entries(
        main.HistoryDeletionRequest(ids=["already-removed"]), repository
    )

    assert response == {"deleted_ids": []}


@pytest.mark.asyncio
async def test_image_detection_history_keeps_an_original_copy(tmp_path, monkeypatch):
    class DetectionResult:
        lane_counts = {}
        class_counts = {"person": 1, "car": 2, "bus": 0, "truck": 0}
        total_vehicles = 3
        processing_time = 0.12
        annotated_image_path = "output_images/annotated.jpg"

    class Detector:
        async def analyze_intersection_image(self, _path, save_annotated):
            assert save_annotated is True
            return DetectionResult()

    class Manager:
        async def update_vehicle_counts(self, _lane_counts):
            return None

    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    uploaded_image = _image_bytes()
    upload = UploadFile(
        file=io.BytesIO(uploaded_image),
        filename="road.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "project_repository", repository)

    tasks = BackgroundTasks()
    await main.detect_vehicles_endpoint(
        request=None,
        background_tasks=tasks,
        image=upload,
        source="image",
        record_history=True,
        detector=Detector(),
        manager=Manager(),
        analytics=None,
    )
    await tasks()

    entry = repository.list_history()[0]
    original_path = tmp_path / entry["original_path"]
    assert original_path.read_bytes() == uploaded_image
    assert entry["original_path"].startswith("output_images")
    assert not list((tmp_path / "uploads").glob("traffic_*"))


def test_registers_downloaded_official_models(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    weights_dir = tmp_path / "models"
    weights_dir.mkdir()
    (weights_dir / "yolov8s.pt").write_bytes(b"weights")
    (weights_dir / "yolov8m.pt").touch()

    models = repository.register_available_official_models(weights_dir)

    assert [(model["name"], model["source"]) for model in models] == [
        ("YOLOv8s 官方预训练（均衡）", "官方预训练"),
    ]


def test_training_metrics_are_read_from_the_last_validation_epoch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.4,0.5,0.45,0.31\n"
        "1,0.8,0.6,0.72,0.55\n",
        encoding="utf-8",
    )

    metrics = _read_training_metrics(run_dir)

    assert metrics == {
        "precision": 0.8,
        "recall": 0.6,
        "map50": 0.72,
        "map50_95": 0.55,
        "f1": pytest.approx(0.6857142857),
    }


def test_packaged_models_register_three_official_and_one_custom_model(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    for filename in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "trained_28068449.pt"):
        (model_directory / filename).write_bytes(b"weights")

    official = repository.register_available_official_models(model_directory)
    custom = repository.register_available_custom_models(model_directory)

    assert [model["name"] for model in official] == [
        "YOLOv8n 官方预训练（速度优先）",
        "YOLOv8s 官方预训练（均衡）",
        "YOLOv8m 官方预训练（精度优先）",
    ]
    assert [model["name"] for model in custom] == ["交通车辆自训练样例（1 Epoch）"]
    assert [model["source"] for model in repository.list_models()] == [
        "自训练样例",
        "官方预训练",
        "官方预训练",
        "官方预训练",
    ]


@pytest.mark.asyncio
async def test_compare_models_queues_selected_models_on_one_dataset(tmp_path, monkeypatch):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    dataset = repository.add_dataset("道路数据", tmp_path / "dataset", {"yaml_path": "data.yaml"})
    first = repository.register_model("模型 A", tmp_path / "a.pt", "测试")
    second = repository.register_model("模型 B", tmp_path / "b.pt", "测试")

    class ComparisonService:
        def submit_comparison(self, received_dataset, received_models, config):
            assert received_dataset["id"] == dataset["id"]
            assert [model["id"] for model in received_models] == [first["id"], second["id"]]
            assert config == {"batch": 4, "imgsz": 640}
            return {"id": "benchmark-job", "kind": "benchmark"}

    monkeypatch.setattr(main, "training_service", ComparisonService())
    response = await main.compare_models(
        main.ModelComparisonRequest(
            dataset_id=dataset["id"], model_ids=[first["id"], second["id"]], batch=4, imgsz=640
        ),
        repository,
    )

    assert response == {"id": "benchmark-job", "kind": "benchmark"}

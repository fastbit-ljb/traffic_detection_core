"""AI Traffic Management System - Modern FastAPI Backend
High-performance web API with real-time vehicle detection and traffic optimization
"""

import asyncio
import shutil
import uuid
import psutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import (
    FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect,
    BackgroundTasks, HTTPException, Depends, status, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

# Import core modules
from .core.config import settings
from .core.logger import setup_logging, get_application_logger
from .core.metrics import get_metrics_response, track_emergency_override, update_websocket_connections
from .core.security import (
    SecurityManager,
    check_rate_limit,
    create_access_token,
    get_password_hash,
    sanitize_filename,
    validate_file_type,
    verify_password_hash,
    verify_token,
)

# Import middleware
from .middleware import SecurityMiddleware, MetricsMiddleware, RequestLoggingMiddleware, HealthCheckMiddleware

# Import services with error handling
try:
    from .services.intelligent_vehicle_detector import IntelligentVehicleDetector
    from .services.adaptive_traffic_manager import AdaptiveTrafficManager
    from .services.analytics_service import TrafficAnalyticsService
    from .services.project_service import (
        DatasetService, ProjectRepository, TrainingService, VideoProcessingService,
    )
except ImportError as e:
    # Log import error but continue - services will be None
    print(f"Warning: Could not import services: {e}")
    IntelligentVehicleDetector = None
    AdaptiveTrafficManager = None 
    TrafficAnalyticsService = None
    DatasetService = None
    ProjectRepository = None
    TrainingService = None
    VideoProcessingService = None

# Import models with error handling
try:
    from .models.traffic_models import (
        VehicleDetectionResult, IntersectionStatus, 
        EmergencyAlert, TrafficSnapshot
    )
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    # Create placeholder classes
    class VehicleDetectionResult: pass
    class IntersectionStatus: pass
    class EmergencyAlert: pass
    class TrafficSnapshot: pass

# Initialize logging
setup_logging()
logger = get_application_logger("main")

# Global services - will be initialized in lifespan
vehicle_detector: Optional[IntelligentVehicleDetector] = None
traffic_manager: Optional[AdaptiveTrafficManager] = None
analytics_service: Optional[TrafficAnalyticsService] = None
project_repository: Optional[ProjectRepository] = None
dataset_service: Optional[DatasetService] = None
training_service: Optional[TrainingService] = None
video_processing_service: Optional[VideoProcessingService] = None
security_manager = SecurityManager()

# Application start time for uptime calculation
app_start_time = time.time()


def parse_baseline_config(enabled: bool, orientation: str, position: float, direction: str = "") -> Optional[Dict[str, object]]:
    """Validate the optional entry/exit baseline shared by live and video detection."""
    if not isinstance(enabled, bool) or not enabled:
        return None
    normalized_orientation = orientation.strip().lower()
    if normalized_orientation not in {"horizontal", "vertical"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="基准线方向无效")
    direction_orientations = {"down": "horizontal", "up": "horizontal", "right": "vertical", "left": "vertical"}
    normalized_direction = direction.strip().lower() or ("down" if normalized_orientation == "horizontal" else "right")
    if normalized_direction not in direction_orientations:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="进出方向无效")
    if direction_orientations[normalized_direction] != normalized_orientation:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="进出方向与基准线不匹配")
    if not 0.05 <= position <= 0.95:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="基准线位置必须在 5% 到 95% 之间")
    return {"orientation": normalized_orientation, "direction": normalized_direction, "position": position}


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_users: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_users[websocket] = user_id
        update_websocket_connections(len(self.active_connections))
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.connection_users.pop(websocket, None)
        update_websocket_connections(len(self.active_connections))
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict, user_id: Optional[str] = None):
        """Broadcast a system event or restrict it to one authenticated account."""
        disconnected = []
        for connection in self.active_connections:
            if user_id is not None and self.connection_users.get(connection) != user_id:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


websocket_manager = ConnectionManager()


class TrainingJobRequest(BaseModel):
    dataset_id: str
    base_model_id: str
    epochs: int = Field(default=30, ge=1, le=500)
    batch: int = Field(default=8, ge=1, le=128)
    imgsz: int = Field(default=640, ge=320, le=1280)


class ModelComparisonRequest(BaseModel):
    dataset_id: str
    model_ids: List[str] = Field(min_length=2, max_length=5)
    batch: int = Field(default=8, ge=1, le=128)
    imgsz: int = Field(default=640, ge=320, le=1280)


class InferenceDeviceRequest(BaseModel):
    device: Literal['auto', 'cpu', 'cuda']


class HistoryDeletionRequest(BaseModel):
    ids: List[str] = Field(min_length=1, max_length=50)


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with comprehensive error handling"""
    # Startup
    logger.info("Starting AI Traffic Management System")
    
    try:
        # Initialize services if available
        global vehicle_detector, traffic_manager, analytics_service
        global project_repository, dataset_service, training_service, video_processing_service

        if ProjectRepository:
            project_repository = ProjectRepository(Path("./storage"))
            project_repository.initialize()
            project_repository.register_available_official_models(Path("."), Path("./models"))
            project_repository.register_available_custom_models(Path("./models"))
            dataset_service = DatasetService(project_repository)
            training_service = TrainingService(project_repository, Path("./models"))
        
        if IntelligentVehicleDetector:
            try:
                vehicle_detector = IntelligentVehicleDetector()
                await vehicle_detector.initialize()
                if project_repository:
                    active_model_exists = any(model["is_active"] for model in project_repository.list_models())
                    project_repository.register_model(
                        name=Path(vehicle_detector.get_model_path() or settings.model_name).name,
                        path=Path(vehicle_detector.get_model_path() or settings.model_name),
                        source="base",
                        activate=not active_model_exists,
                    )
                    video_processing_service = VideoProcessingService(
                        project_repository, vehicle_detector, Path("./output_videos"), websocket_manager.broadcast
                    )
                logger.info("Vehicle detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vehicle detector: {e}")
                vehicle_detector = None
        
        if AdaptiveTrafficManager:
            try:
                traffic_manager = AdaptiveTrafficManager()
                await traffic_manager.initialize()
                logger.info("Traffic manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize traffic manager: {e}")
                traffic_manager = None
        
        if TrafficAnalyticsService:
            try:
                analytics_service = TrafficAnalyticsService()
                await analytics_service.initialize()
                logger.info("Analytics service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize analytics service: {e}")
                analytics_service = None
        
        # Start simulation if traffic manager is available
        if traffic_manager:
            try:
                await traffic_manager.start_simulation()
                logger.info("Traffic simulation started")
            except Exception as e:
                logger.error(f"Failed to start simulation: {e}")
        
        # Create necessary directories
        directories = ["./output_images", "./output_videos", "./uploads", "./logs", "./models", "./storage"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("Application startup completed")
        
    except Exception as error:
        logger.error(f"Critical startup error: {error}")
        # Continue startup even if some services fail
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Traffic Management System")
    
    try:
        if traffic_manager:
            await traffic_manager.cleanup()
        if vehicle_detector:
            await vehicle_detector.cleanup()
        if analytics_service:
            await analytics_service.cleanup()
            
        logger.info("All services shut down successfully")
        
    except Exception as error:
        logger.error(f"Error during shutdown: {error}")


# Create FastAPI application
app = FastAPI(
    title=settings.application_name,
    description="Intelligent traffic control with real-time vehicle detection and adaptive signal optimization",
    version=settings.application_version,
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug_mode else None,
    redoc_url="/api/redoc" if settings.debug_mode else None
)

# Security
security = HTTPBearer(auto_error=False)

# Add middleware in correct order (reverse order of execution)
app.add_middleware(HealthCheckMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(MetricsMiddleware)
if settings.debug_mode:
    app.add_middleware(RequestLoggingMiddleware, log_body=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug_mode else settings.allowed_hosts,
)


# Dependency functions with improved error handling
async def get_vehicle_detector() -> Optional[IntelligentVehicleDetector]:
    """Get vehicle detector dependency"""
    if not vehicle_detector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vehicle detection service not available"
        )
    
    if not vehicle_detector.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vehicle detection service not ready"
        )
    return vehicle_detector


async def get_traffic_manager() -> Optional[AdaptiveTrafficManager]:
    """Get traffic manager dependency"""
    if not traffic_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Traffic management service not available"
        )
    
    if not traffic_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Traffic management service not ready"
        )
    return traffic_manager


async def get_analytics_service() -> Optional[TrafficAnalyticsService]:
    """Get analytics service dependency"""
    if not analytics_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics service not available"
        )
    
    if not analytics_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics service not ready"
        )
    return analytics_service


async def get_project_repository() -> ProjectRepository:
    if not project_repository:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="项目数据服务不可用",
        )
    return project_repository


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {key: user[key] for key in ("id", "username", "created_at")}


def _normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not normalized or any(not (character.isalnum() or character in "_-.") for character in normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名只能包含字母、数字、下划线、短横线或点",
        )
    return normalized


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    repository: ProjectRepository = Depends(get_project_repository),
) -> Dict[str, Any]:
    """Resolve the bearer token and return the active account."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub") if payload else None
    user = repository.get_user(user_id) if user_id else None
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效，请重新登录")
    return user


def _dependency_user_id(current_user: object) -> Optional[str]:
    """Keep direct service-level tests backward compatible with dependency defaults."""
    return current_user.get("id") if isinstance(current_user, dict) else None


@app.post("/api/auth/register")
async def register_account(
    credentials: AuthCredentials,
    repository: ProjectRepository = Depends(get_project_repository),
):
    username = _normalize_username(credentials.username)
    user = repository.create_user(username, get_password_hash(credentials.password))
    if not user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
    token = create_access_token({"sub": user["id"], "username": user["username"]})
    return {"access_token": token, "token_type": "bearer", "user": _public_user(user)}


@app.post("/api/auth/login")
async def login_account(
    credentials: AuthCredentials,
    repository: ProjectRepository = Depends(get_project_repository),
):
    username = _normalize_username(credentials.username)
    user = repository.get_user_by_username(username)
    if not user or not verify_password_hash(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token({"sub": user["id"], "username": user["username"]})
    return {"access_token": token, "token_type": "bearer", "user": _public_user(user)}


@app.get("/api/auth/me")
async def current_account(current_user: Dict[str, Any] = Depends(get_current_user)):
    return _public_user(current_user)


# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return get_metrics_response()


# WebSocket endpoint for real-time updates
@app.websocket("/ws/traffic-updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time traffic updates"""
    token = websocket.query_params.get("token")
    payload = verify_token(token) if token else None
    user_id = payload.get("sub") if payload else None
    if not user_id or not project_repository or not project_repository.get_user(user_id):
        await websocket.close(code=4401, reason="请先登录")
        return
    await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Send periodic updates
            if traffic_manager and traffic_manager.is_ready():
                try:
                    intersection_status = await traffic_manager.get_current_status()
                    await websocket.send_json({
                        "type": "intersection_status",
                        "data": jsonable_encoder(intersection_status),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.info(f"WebSocket closed while sending status: {e}")
                    break
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        pass
    except Exception as error:
        logger.error(f"WebSocket error: {error}")
    finally:
        websocket_manager.disconnect(websocket)


# API Routes
@app.post("/api/detect-vehicles")
async def detect_vehicles_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    source: str = Form("image"),
    record_history: bool = Form(True),
    baseline_enabled: bool = Form(False),
    baseline_orientation: str = Form("horizontal"),
    baseline_direction: str = Form(""),
    baseline_position: float = Form(0.5),
    baseline_session: str = Form("camera"),
    detector: IntelligentVehicleDetector = Depends(get_vehicle_detector),
    manager: AdaptiveTrafficManager = Depends(get_traffic_manager),
    analytics: TrafficAnalyticsService = Depends(get_analytics_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Analyze traffic image and detect vehicles using YOLOv8"""
    try:
        # Rate limiting is handled by middleware
        
        # Validate file type and size
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Check file size (10MB limit)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size too large (max 10MB)"
            )
        
        # Validate file type by extension
        if not validate_file_type(image.filename, ['.jpg', '.jpeg', '.png', '.bmp']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Use JPEG, PNG, or BMP."
            )
        
        # Generate secure filename
        secure_filename = sanitize_filename(image.filename or "upload.jpg")
        upload_id = str(uuid.uuid4())
        temp_path = f"./uploads/traffic_{upload_id}_{secure_filename}"
        Path(temp_path).parent.mkdir(exist_ok=True)

        # Save uploaded image
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Processing uploaded image: {image.filename} -> {temp_path}")
        
        # Perform vehicle detection
        baseline_config = parse_baseline_config(baseline_enabled, baseline_orientation, baseline_position, baseline_direction)
        analysis_kwargs = {"baseline_config": baseline_config, "baseline_session": baseline_session} if baseline_config else {}
        detection_result = await detector.analyze_intersection_image(temp_path, save_annotated=True, **analysis_kwargs)
        
        # Update traffic management system
        if hasattr(detection_result, 'lane_counts'):
            await manager.update_vehicle_counts(detection_result.lane_counts)
        
        # Record analytics data
        if analytics:
            background_tasks.add_task(
                analytics.record_detection,
                detection_result,
                datetime.now(timezone.utc)
            )

        if project_repository and record_history:
            active_model = next((model for model in project_repository.list_models() if model["is_active"]), None)
            original_path = Path("./output_images") / f"history_{upload_id}_original{Path(secure_filename).suffix.lower() or '.jpg'}"
            original_path.parent.mkdir(exist_ok=True)
            background_tasks.add_task(shutil.copy2, temp_path, original_path)
            # Persist the ownership row before returning so the just-created
            # annotated image can be authorized immediately by the UI.
            project_repository.add_history(
                media_type="camera" if source == "camera" else "image",
                source_name=image.filename or "image",
                class_counts={getattr(key, "value", str(key)): value for key, value in detection_result.class_counts.items()},
                total_objects=detection_result.total_vehicles,
                processing_time=detection_result.processing_time,
                output_path=detection_result.annotated_image_path,
                model_name=active_model["name"] if active_model else None,
                original_path=str(original_path),
                user_id=_dependency_user_id(current_user),
            )

        # Broadcast updates to WebSocket clients
        background_tasks.add_task(
            websocket_manager.broadcast,
            {
                "type": "vehicle_detection",
                "data": jsonable_encoder(detection_result),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            user_id=_dependency_user_id(current_user),
        )
        
        # Clean up temporary file
        background_tasks.add_task(
            lambda: Path(temp_path).unlink(missing_ok=True)
        )

        total_targets = getattr(detection_result, 'total_vehicles', 0)
        logger.info(f"Road-object detection completed: {total_targets} targets detected")

        return detection_result

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Vehicle detection error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vehicle detection failed: {str(error)}"
        )


@app.get("/api/history")
async def list_detection_history(
    limit: int = 50,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return repository.list_history(max(1, min(limit, 200)), _dependency_user_id(current_user))


@app.delete("/api/history")
async def delete_detection_history_entries(
    deletion_request: HistoryDeletionRequest,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    deleted_ids = repository.delete_history_entries(deletion_request.ids, _dependency_user_id(current_user))
    return {"deleted_ids": deleted_ids}


@app.delete("/api/history/{history_id}")
async def delete_detection_history(
    history_id: str,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    history_entry = repository.delete_history(history_id, _dependency_user_id(current_user))
    if not history_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="检测记录不存在")
    return {"id": history_entry["id"]}


@app.get("/api/datasets")
async def list_datasets(repository: ProjectRepository = Depends(get_project_repository)):
    return repository.list_datasets()


@app.post("/api/datasets/import")
async def import_dataset(
    archive: UploadFile = File(...),
    dataset_name: str = Form(""),
    repository: ProjectRepository = Depends(get_project_repository),
):
    if not dataset_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据集服务不可用")
    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请上传 ZIP 数据集")
    content = await archive.read()
    if len(content) > 1024 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="数据集不能超过 1GB")
    try:
        return await dataset_service.import_archive(content, archive.filename, dataset_name)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.get("/api/models")
async def list_models(repository: ProjectRepository = Depends(get_project_repository)):
    return repository.list_models()


@app.get("/api/inference-device")
async def get_inference_device(detector: IntelligentVehicleDetector = Depends(get_vehicle_detector)):
    return detector.get_device_status()


@app.put("/api/inference-device")
async def set_inference_device(
    request: InferenceDeviceRequest,
    detector: IntelligentVehicleDetector = Depends(get_vehicle_detector),
):
    try:
        status_payload = detector.set_inference_device(request.device)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    await websocket_manager.broadcast({"type": "inference_device_changed", "data": status_payload})
    return status_payload


@app.post("/api/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    model_name: str = Form(""),
    repository: ProjectRepository = Depends(get_project_repository),
):
    if not model_file.filename or not model_file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="模型文件必须是 .pt")
    content = await model_file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="模型文件为空")
    models_directory = Path("./models")
    models_directory.mkdir(exist_ok=True)
    destination = models_directory / f"upload_{uuid.uuid4().hex[:8]}.pt"
    destination.write_bytes(content)
    return repository.register_model(model_name.strip() or Path(model_file.filename).stem, destination, "upload")


@app.post("/api/models/{model_id}/activate")
async def activate_model(
    model_id: str,
    detector: IntelligentVehicleDetector = Depends(get_vehicle_detector),
    repository: ProjectRepository = Depends(get_project_repository),
):
    model = repository.get_model(model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="模型不存在")
    try:
        await detector.set_model(model["path"])
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    active_model = repository.activate_model(model_id)
    await websocket_manager.broadcast({"type": "model_changed", "data": active_model})
    return active_model


@app.get("/api/training/jobs")
async def list_training_jobs(
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return repository.list_jobs(
        kind="training", active_only=True, user_id=_dependency_user_id(current_user)
    )


@app.get("/api/training/runs")
async def list_training_runs(
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return completed and failed runs so thesis metrics remain traceable after restart."""
    return repository.list_jobs(
        kind="training", limit=12, user_id=_dependency_user_id(current_user)
    )


@app.post("/api/training/jobs", status_code=status.HTTP_202_ACCEPTED)
async def start_training_job(
    request: TrainingJobRequest,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if not training_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="训练服务不可用")
    dataset = repository.get_dataset(request.dataset_id)
    model = repository.get_model(request.base_model_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="基础模型不存在")
    config = {"epochs": request.epochs, "batch": request.batch, "imgsz": request.imgsz}
    user_id = _dependency_user_id(current_user)
    if user_id:
        return training_service.submit(dataset, model, config, user_id=user_id)
    # Preserve direct service-level calls used by maintenance scripts and tests.
    return training_service.submit(dataset, model, config)


@app.get("/api/experiments")
async def list_comparison_experiments(
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List persisted validation-set comparison experiments."""
    return repository.list_jobs(
        kind="benchmark", limit=12, user_id=_dependency_user_id(current_user)
    )


@app.post("/api/experiments/compare", status_code=status.HTTP_202_ACCEPTED)
async def compare_models(
    request: ModelComparisonRequest,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if not training_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="训练服务不可用")
    dataset = repository.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    model_ids = list(dict.fromkeys(request.model_ids))
    if len(model_ids) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请至少选择两个不同模型进行对比")
    models = [repository.get_model(model_id) for model_id in model_ids]
    if any(model is None for model in models):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="存在未找到的模型")
    selected_models = [model for model in models if model is not None]
    config = {"batch": request.batch, "imgsz": request.imgsz}
    user_id = _dependency_user_id(current_user)
    if user_id:
        return training_service.submit_comparison(dataset, selected_models, config, user_id=user_id)
    # Preserve direct service-level calls used by maintenance scripts and tests.
    return training_service.submit_comparison(dataset, selected_models, config)


@app.post("/api/detect-video", status_code=status.HTTP_202_ACCEPTED)
async def detect_video(
    video: UploadFile = File(...),
    baseline_enabled: bool = Form(False),
    baseline_orientation: str = Form("horizontal"),
    baseline_direction: str = Form(""),
    baseline_position: float = Form(0.5),
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if not video_processing_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="视频服务不可用")
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    filename = video.filename or "video.mp4"
    if Path(filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 MP4、AVI、MOV、MKV 视频")
    content = await video.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="视频不能超过 500MB")
    source_path = Path("./uploads") / f"video_{uuid.uuid4().hex}_{Path(filename).name}"
    source_path.write_bytes(content)
    active_model = next((model for model in repository.list_models() if model["is_active"]), None)
    baseline_config = parse_baseline_config(baseline_enabled, baseline_orientation, baseline_position, baseline_direction)
    return video_processing_service.submit(
        source_path,
        filename,
        active_model["name"] if active_model else "unknown",
        baseline_config,
        user_id=_dependency_user_id(current_user),
    )


@app.get("/api/video-jobs/{job_id}")
async def get_video_job(
    job_id: str,
    repository: ProjectRepository = Depends(get_project_repository),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    job = repository.get_job(job_id, _dependency_user_id(current_user))
    if not job or job["kind"] != "video":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="视频任务不存在")
    return job


@app.get("/api/intersection-status")
async def get_intersection_status(
    manager: AdaptiveTrafficManager = Depends(get_traffic_manager)
):
    """Get current intersection status and signal states"""
    try:
        return await manager.get_current_status()
    except Exception as error:
        logger.error(f"Intersection status error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get intersection status: {str(error)}"
        )


@app.post("/api/emergency-override")
async def emergency_override(
    alert: EmergencyAlert,
    background_tasks: BackgroundTasks,
    manager: AdaptiveTrafficManager = Depends(get_traffic_manager)
):
    """Handle emergency vehicle detection and override signals"""
    try:
        await manager.handle_emergency_override(alert)
        
        # Track emergency override metrics
        track_emergency_override(
            alert.emergency_type.value,
            alert.detected_lane.value,
        )
        
        # Broadcast emergency alert
        background_tasks.add_task(
            websocket_manager.broadcast,
            {
                "type": "emergency_alert",
                "data": jsonable_encoder(alert),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        logger.warning(f"Emergency override activated: {alert.alert_id}")
        
        return {
            "status": "emergency_override_activated",
            "alert_id": alert.alert_id,
            "message": f"Emergency override activated for {alert.detected_lane.value} lane"
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Emergency override error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency override failed: {str(error)}"
        )


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        # Get system metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_bytes = memory.used
        except Exception:
            cpu_percent = 0.0
            memory_percent = 0.0
            memory_bytes = 0
        
        # Check service health
        services = {
            "vehicle_detector": vehicle_detector.is_ready() if vehicle_detector else False,
            "traffic_manager": traffic_manager.is_ready() if traffic_manager else False,
            "analytics": analytics_service.is_ready() if analytics_service else False
        }
        
        # Calculate health score
        healthy_services = sum(1 for status in services.values() if status)
        total_services = len(services)
        health_score = (healthy_services / total_services) if total_services > 0 else 0.0
        
        # Determine overall status
        if health_score >= 0.8:
            overall_status = "healthy"
        elif health_score >= 0.5:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "health_score": health_score,
            "services": services,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_bytes": memory_bytes
            },
            "websocket_connections": len(websocket_manager.active_connections),
            "version": settings.application_version,
            "environment": getattr(settings, 'environment', 'unknown')
        }
        
    except Exception as error:
        logger.error(f"Health check error: {error}")
        return {
            "status": "unhealthy",
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.application_version
        }


@app.get("/api/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "application_name": settings.application_name,
        "version": settings.application_version,
        "environment": getattr(settings, 'environment', 'development'),
        "api_prefix": settings.api_prefix,
        "debug_mode": settings.debug_mode,
        "features": {
            "vehicle_detection": vehicle_detector is not None,
            "adaptive_signals": traffic_manager is not None,
            "emergency_override": True,
            "real_time_analytics": analytics_service is not None,
            "websocket_support": True,
            "metrics_collection": True,
            "rate_limiting": True,
            "security_middleware": True
        },
        "model_info": {
            "model_name": settings.model_name,
            "confidence_threshold": settings.detection_confidence_threshold,
            "gpu_acceleration": settings.enable_gpu_acceleration
        }
    }


# Analytics endpoints (with error handling)
@app.get("/api/analytics/summary")
async def get_traffic_analytics(
    period: str = "current",
    analytics: TrafficAnalyticsService = Depends(get_analytics_service)
):
    """Get traffic analytics and performance metrics"""
    try:
        return await analytics.generate_summary(period)
    except Exception as error:
        logger.error(f"Analytics error: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics unavailable: {str(error)}"
        )


# Serve generated detection media through an ownership-checked API. Create the
# directories before registering the routes so startup works before first use.
try:
    output_images_directory = Path("./output_images")
    output_images_directory.mkdir(exist_ok=True)
    output_videos_directory = Path("./output_videos")
    output_videos_directory.mkdir(exist_ok=True)
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


@app.get("/api/media/{media_kind}/{filename}")
async def get_owned_media(
    media_kind: Literal["images", "videos"],
    filename: str,
    token: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    repository: ProjectRepository = Depends(get_project_repository),
):
    """Return generated media only when it belongs to the authenticated user."""
    access_token = credentials.credentials if credentials else token
    payload = verify_token(access_token) if access_token else None
    user_id = payload.get("sub") if payload else None
    if not user_id or not repository.get_user(user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="媒体文件名无效")
    if not repository.user_owns_media(user_id, filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="媒体文件不存在")
    root = output_images_directory if media_kind == "images" else output_videos_directory
    media_path = (root / filename).resolve()
    if not media_path.is_relative_to(root.resolve()) or not media_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="媒体文件不存在")
    return FileResponse(media_path)


# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url.path)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    error_id = str(uuid.uuid4())
    logger.error(f"Internal server error [{error_id}] on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_id": error_id}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )

"""
Intelligent Vehicle Detection Service using YOLOv8
Handles real-time vehicle detection and traffic analysis
"""

import asyncio
import inspect
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from ..core.config import settings
from ..core.logger import LoggerMixin
from ..models.traffic_models import VehicleDetectionResult, DetectedVehicle


TARGET_CLASS_NAMES = ("person", "car", "bus", "truck")


@dataclass(frozen=True)
class BaselineConfig:
    """Normalized screen-space line configuration for entry and exit counts."""

    orientation: str = "horizontal"
    position: float = 0.5
    direction: str = "down"

    @classmethod
    def from_mapping(cls, value: Dict[str, Any]) -> "BaselineConfig":
        orientation = str(value.get("orientation", "horizontal")).lower()
        position = float(value.get("position", 0.5))
        direction = str(value.get("direction", "down" if orientation == "horizontal" else "right")).lower()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("基准线方向必须是 horizontal 或 vertical")
        direction_orientations = {"down": "horizontal", "up": "horizontal", "right": "vertical", "left": "vertical"}
        if direction not in direction_orientations or direction_orientations[direction] != orientation:
            raise ValueError("进出方向与基准线不匹配")
        if not 0.05 <= position <= 0.95:
            raise ValueError("基准线位置必须在 5% 到 95% 之间")
        return cls(orientation=orientation, position=position, direction=direction)

    def as_dict(self) -> Dict[str, Any]:
        return {"orientation": self.orientation, "direction": self.direction, "position": self.position}


@dataclass
class _CrossingTrack:
    target_type: str
    center: Tuple[float, float]
    side: int
    last_seen_frame: int


class BaselineCrossingCounter:
    """Counts same-class centroids that move through one configurable baseline."""

    def __init__(self, baseline: BaselineConfig):
        self.baseline = baseline
        self._next_track_id = 0
        self._frame_index = 0
        self._tracks: Dict[int, _CrossingTrack] = {}
        self._flow_counts = self.empty_counts()

    @staticmethod
    def empty_counts() -> Dict[str, Dict[str, int]]:
        return {
            "entry": {target_type: 0 for target_type in TARGET_CLASS_NAMES},
            "exit": {target_type: 0 for target_type in TARGET_CLASS_NAMES},
        }

    def _side(self, center: Tuple[float, float]) -> int:
        coordinate = center[1] if self.baseline.orientation == "horizontal" else center[0]
        offset = coordinate - self.baseline.position
        if abs(offset) < 0.015:
            return 0
        return 1 if offset > 0 else -1

    def _match_track(self, target_type: str, center: Tuple[float, float], claimed: set[int]) -> Optional[int]:
        candidates = [
            (track_id, math.dist(track.center, center))
            for track_id, track in self._tracks.items()
            if track_id not in claimed and track.target_type == target_type and self._frame_index - track.last_seen_frame <= 45
        ]
        if not candidates:
            return None
        track_id, distance = min(candidates, key=lambda item: item[1])
        return track_id if distance <= 0.14 else None

    def update(self, targets: List[DetectedVehicle]) -> Dict[str, Dict[str, int]]:
        self._frame_index += 1
        claimed: set[int] = set()

        for target in targets:
            target_type = str(getattr(target.vehicle_type, "value", target.vehicle_type))
            center = (target.center_coordinates["x"], target.center_coordinates["y"])
            side = self._side(center)
            track_id = self._match_track(target_type, center, claimed)

            if track_id is None:
                self._next_track_id += 1
                self._tracks[self._next_track_id] = _CrossingTrack(target_type, center, side, self._frame_index)
                claimed.add(self._next_track_id)
                continue

            track = self._tracks[track_id]
            if side and track.side and side != track.side:
                moved_toward_positive_side = track.side < side
                entry_toward_positive_side = self.baseline.direction in {"down", "right"}
                direction = "entry" if moved_toward_positive_side == entry_toward_positive_side else "exit"
                self._flow_counts[direction][target_type] += 1
            if side:
                track.side = side
            track.center = center
            track.last_seen_frame = self._frame_index
            claimed.add(track_id)

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if self._frame_index - track.last_seen_frame <= 45
        }
        return {direction: counts.copy() for direction, counts in self._flow_counts.items()}


class IntelligentVehicleDetector(LoggerMixin):
    """YOLOv8 detector for the four road-object classes shown in the dashboard."""
    
    # COCO class mapping. Keep this deliberately small so the dashboard only
    # reports the four categories agreed for this project.
    TARGET_CLASSES = {
        0: 'person',
        2: 'car',
        5: 'bus',
        7: 'truck'
    }
    DISPLAY_LABELS = {
        'person': '行人',
        'car': '汽车',
        'bus': '公交车',
        'truck': '卡车',
    }
    _annotation_fonts: Dict[int, ImageFont.ImageFont] = {}
    VEHICLE_CLASSES = TARGET_CLASSES  # Backwards-compatible alias for callers.
    
    # Lane detection zones (normalized coordinates)
    LANE_ZONES = {
        'north': {'x_min': 0.45, 'x_max': 0.55, 'y_min': 0.0, 'y_max': 0.45},
        'south': {'x_min': 0.45, 'x_max': 0.55, 'y_min': 0.55, 'y_max': 1.0},
        'east': {'x_min': 0.55, 'x_max': 1.0, 'y_min': 0.45, 'y_max': 0.55},
        'west': {'x_min': 0.0, 'x_max': 0.45, 'y_min': 0.45, 'y_max': 0.55}
    }
    
    def __init__(self):
        super().__init__()
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
        self.model_initialized = False
        self.requested_device = 'auto'
        self._model_lock = threading.RLock()
        self._flow_lock = threading.RLock()
        self._live_crossing_counters: Dict[str, BaselineCrossingCounter] = {}
        self.performance_metrics = {
            'total_detections': 0,
            'average_inference_time': 0.0,
            'last_detection_time': None
        }
    
    async def initialize(self) -> None:
        """Initialize the YOLOv8 model asynchronously"""
        if self.model_initialized and self.model is not None:
            return

        try:
            start_time = time.time()
            self.logger.info("Initializing YOLOv8 model...")
            
            # Create model directory if it doesn't exist
            model_path = Path(settings.model_cache_directory)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                self._load_model
            )
            self.model_path = str(self._resolve_model_path())
            
            initialization_time = time.time() - start_time
            self.model_initialized = True
            
            self.log_performance(
                "model_initialization", 
                initialization_time
            )
            self.logger.info(f"YOLOv8 model initialized successfully")
            
        except Exception as error:
            self.log_error_with_context(error, "model_initialization")
            raise
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model (runs in thread pool)"""
        model_file = Path(settings.model_cache_directory) / settings.model_name

        if model_file.exists():
            self.logger.info(f"Loading cached model from {model_file}")
            return YOLO(str(model_file))

        local_model_file = Path(settings.model_name)
        if local_model_file.exists():
            self.logger.info(f"Loading local model from {local_model_file}")
            return YOLO(str(local_model_file))

        self.logger.info(f"Downloading {settings.model_name} model...")
        return YOLO(settings.model_name)

    def _resolve_model_path(self) -> Path:
        cached_model = Path(settings.model_cache_directory) / settings.model_name
        if cached_model.exists():
            return cached_model.resolve()
        local_model = Path(settings.model_name)
        return local_model.resolve() if local_model.exists() else Path(settings.model_name)

    async def set_model(self, model_path: str) -> None:
        """Atomically switch the detector to an uploaded or trained weight file."""
        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists() or resolved_path.suffix.lower() != ".pt":
            raise ValueError("模型文件必须是存在的 .pt 权重")

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(None, lambda: YOLO(str(resolved_path)))
        with self._model_lock:
            self.model = model
            self.model_path = str(resolved_path)
            self.model_initialized = True
        self.logger.info(f"YOLOv8 model switched to {resolved_path.name}")

    def get_device_status(self) -> Dict[str, Optional[str]]:
        """Return the requested and resolved inference device for the dashboard."""
        cuda_available = torch.cuda.is_available()
        return {
            'requested_device': self.requested_device,
            'active_device': self._resolve_device(),
            'cuda_available': cuda_available,
            'cuda_version': torch.version.cuda,
            'device_name': torch.cuda.get_device_name(0) if cuda_available else None,
        }

    def set_inference_device(self, device: str) -> Dict[str, Optional[str]]:
        """Select CPU, CUDA, or automatic device selection for later predictions."""
        normalized_device = device.strip().lower()
        if normalized_device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('推理设备必须是 auto、cpu 或 cuda')
        if normalized_device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('当前环境未检测到可用 CUDA 设备')
        with self._model_lock:
            self.requested_device = normalized_device
        return self.get_device_status()

    def _resolve_device(self) -> str:
        if self.requested_device == 'cpu':
            return 'cpu'
        if self.requested_device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('当前环境未检测到可用 CUDA 设备')
            return 'cuda'
        return 'cuda' if settings.enable_gpu_acceleration and torch.cuda.is_available() else 'cpu'
    
    async def analyze_intersection_image(
        self, 
        image_path: str,
        save_annotated: bool = True,
        baseline_config: Optional[Dict[str, Any]] = None,
        baseline_session: str = "camera",
    ) -> VehicleDetectionResult:
        """Analyze intersection image for vehicle detection"""
        if not self.model_initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Run detection
            results = await self._run_detection(image)
            
            # Process results
            detected_vehicles = self._process_detection_results(results, image.shape)
            lane_counts = self._count_vehicles_by_lane(detected_vehicles)
            class_counts = self._count_targets_by_class(detected_vehicles)
            baseline = BaselineConfig.from_mapping(baseline_config) if baseline_config else None
            flow_counts = self._count_live_crossings(baseline, baseline_session, detected_vehicles)
            
            # Save annotated image if requested
            annotated_image_path = None
            if save_annotated:
                annotated_image_path = await self._save_annotated_image(
                    image, detected_vehicles, image_path, baseline
                )
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self._update_performance_metrics(inference_time)
            
            result = VehicleDetectionResult(
                total_vehicles=len(detected_vehicles),
                class_counts=class_counts,
                flow_counts=flow_counts,
                lane_counts=lane_counts,
                detected_vehicles=detected_vehicles,
                confidence_scores=[v.confidence for v in detected_vehicles],
                processing_time=inference_time,
                image_path=image_path,
                annotated_image_path=annotated_image_path
            )
            
            self.log_performance("vehicle_detection", inference_time)
            self.logger.info(
                f"Detected {len(detected_vehicles)} road targets in {inference_time:.3f}s"
            )
            
            return result
            
        except Exception as error:
            self.log_error_with_context(error, "vehicle_detection")
            raise

    def _count_live_crossings(
        self,
        baseline: Optional[BaselineConfig],
        session: str,
        targets: List[DetectedVehicle],
    ) -> Dict[str, Dict[str, int]]:
        if not baseline:
            return BaselineCrossingCounter.empty_counts()

        safe_session = (session or "camera").strip()[:80]
        counter_key = f"{safe_session}:{baseline.orientation}:{baseline.position:.4f}"
        with self._flow_lock:
            counter = self._live_crossing_counters.get(counter_key)
            if counter is None:
                counter = BaselineCrossingCounter(baseline)
                self._live_crossing_counters[counter_key] = counter
                if len(self._live_crossing_counters) > 12:
                    self._live_crossing_counters.pop(next(iter(self._live_crossing_counters)))
            return counter.update(targets)
    
    async def _run_detection(self, image: np.ndarray) -> List:
        """Run YOLOv8 detection on image"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._predict,
            image,
        )

    def _predict(self, image: np.ndarray) -> List:
        with self._model_lock:
            if self.model is None:
                raise RuntimeError("YOLOv8 model is not initialized")
            device = self._resolve_device()
            return self.model(
                image,
                conf=settings.detection_confidence_threshold,
                iou=settings.non_max_suppression_threshold,
                device=device,
            )
    
    def _process_detection_results(
        self, 
        results: List, 
        image_shape: Tuple[int, int, int]
    ) -> List[DetectedVehicle]:
        """Process YOLOv8 detection results into DetectedVehicle objects"""
        detected_vehicles = []
        height, width = image_shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if class_id not in self.TARGET_CLASSES:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Normalize coordinates
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                
                # Determine lane
                lane = self._determine_vehicle_lane(center_x, center_y)
                
                vehicle = DetectedVehicle(
                    vehicle_type=self.TARGET_CLASSES[class_id],
                    display_label=self._display_label(self.TARGET_CLASSES[class_id]),
                    confidence=confidence,
                    bounding_box={
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    },
                    center_coordinates={'x': center_x, 'y': center_y},
                    lane=lane
                )
                
                detected_vehicles.append(vehicle)
        
        return detected_vehicles
    
    def _determine_vehicle_lane(self, center_x: float, center_y: float) -> str:
        """Determine which lane a vehicle belongs to based on position"""
        for lane, zone in self.LANE_ZONES.items():
            if (zone['x_min'] <= center_x <= zone['x_max'] and 
                zone['y_min'] <= center_y <= zone['y_max']):
                return lane
        return 'unknown'
    
    def _count_vehicles_by_lane(self, vehicles: List[DetectedVehicle]) -> Dict[str, int]:
        """Count traffic vehicles in each lane, excluding pedestrians."""
        lane_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        for vehicle in vehicles:
            if vehicle.vehicle_type != 'person' and vehicle.lane in lane_counts:
                lane_counts[vehicle.lane] += 1
        
        return lane_counts

    def _count_targets_by_class(self, targets: List[DetectedVehicle]) -> Dict[str, int]:
        """Return a stable zero-filled count for every dashboard class."""
        counts = {target_class: 0 for target_class in self.TARGET_CLASSES.values()}
        for target in targets:
            counts[target.vehicle_type] += 1
        return counts

    @classmethod
    def _annotation_font(cls, size: int) -> ImageFont.ImageFont:
        """Load a CJK-capable font once for image and video annotations."""
        if size in cls._annotation_fonts:
            return cls._annotation_fonts[size]

        font_paths = (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "NotoSansCJK-Regular.ttc",
        )
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size)
                cls._annotation_fonts[size] = font
                return font
            except OSError:
                continue

        font = ImageFont.load_default()
        cls._annotation_fonts[size] = font
        return font

    @classmethod
    def _display_label(cls, vehicle_type: str) -> str:
        return cls.DISPLAY_LABELS.get(vehicle_type, vehicle_type)

    @staticmethod
    def _resolve_ffmpeg_binary() -> str:
        """Return a usable FFmpeg binary for browser-compatible video encoding."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg

        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("未找到 FFmpeg，无法生成浏览器兼容的视频") from error
        return imageio_ffmpeg.get_ffmpeg_exe()

    @staticmethod
    def _browser_transcode_command(ffmpeg_binary: str, source_path: Path, output_path: Path) -> List[str]:
        """Build an H.264 MP4 command supported by modern browser video elements."""
        return [
            ffmpeg_binary,
            "-y",
            "-i", str(source_path),
            "-map", "0:v:0",
            "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(output_path),
        ]

    def _transcode_for_browser(self, source_path: Path, output_path: Path) -> None:
        """Convert OpenCV's MP4V output to a browser-playable H.264 MP4."""
        command = self._browser_transcode_command(
            self._resolve_ffmpeg_binary(), source_path, output_path
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            error_message = completed.stderr.strip() or "FFmpeg 未能输出视频"
            raise RuntimeError(f"浏览器兼容视频转码失败: {error_message}")

    @classmethod
    def _draw_targets(
        cls,
        image: np.ndarray,
        targets: List[DetectedVehicle],
        baseline: Optional[BaselineConfig] = None,
    ) -> np.ndarray:
        """Draw Chinese labels through Pillow because OpenCV cannot render CJK text."""
        colors = {
            'person': (168, 85, 247),
            'car': (20, 118, 110),
            'bus': (6, 121, 217),
            'truck': (235, 145, 25),
        }
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = cls._annotation_font(18)

        if baseline:
            width, height = pil_image.size
            if baseline.orientation == "horizontal":
                y = round(height * baseline.position)
                draw.line([(0, y), (width, y)], fill=(220, 38, 38), width=3)
            else:
                x = round(width * baseline.position)
                draw.line([(x, 0), (x, height)], fill=(220, 38, 38), width=3)

        for target in targets:
            bbox = target.bounding_box
            color = colors[target.vehicle_type]
            draw.rectangle(
                [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])], outline=color, width=2
            )
            label = f"{target.display_label or cls._display_label(target.vehicle_type)} {target.confidence:.0%}"
            text_bounds = draw.textbbox((0, 0), label, font=font)
            text_width = text_bounds[2] - text_bounds[0]
            text_height = text_bounds[3] - text_bounds[1]
            label_x = max(0, bbox['x1'])
            label_y = max(0, bbox['y1'] - text_height - 8)
            draw.rectangle(
                [(label_x, label_y), (label_x + text_width + 8, label_y + text_height + 6)],
                fill=color,
            )
            draw.text((label_x + 4, label_y + 2), label, fill="white", font=font)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    async def analyze_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback=None,
        baseline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Annotate a video frame by frame and report the latest object counts."""
        if not self.model_initialized:
            await self.initialize()

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("无法读取视频文件")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not width or not height:
            capture.release()
            raise ValueError("视频尺寸无效")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
        writer = cv2.VideoWriter(
            str(rendered_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("无法创建输出视频")

        started_at = time.time()
        processed_frames = 0
        latest_counts = {target_class: 0 for target_class in self.TARGET_CLASSES.values()}
        baseline = BaselineConfig.from_mapping(baseline_config) if baseline_config else None
        crossing_counter = BaselineCrossingCounter(baseline) if baseline else None
        flow_counts = BaselineCrossingCounter.empty_counts()
        try:
            while True:
                read_ok, frame = capture.read()
                if not read_ok:
                    break
                results = await self._run_detection(frame)
                targets = self._process_detection_results(results, frame.shape)
                latest_counts = self._count_targets_by_class(targets)
                if crossing_counter:
                    flow_counts = crossing_counter.update(targets)
                writer.write(self._draw_targets(frame.copy(), targets, baseline))
                processed_frames += 1

                if progress_callback and (processed_frames % 10 == 0 or processed_frames == frame_count):
                    update = progress_callback(processed_frames, frame_count, latest_counts, flow_counts)
                    if inspect.isawaitable(update):
                        await update
        finally:
            capture.release()
            writer.release()

        try:
            if not processed_frames:
                raise ValueError("视频不包含可处理的帧")
            await asyncio.to_thread(self._transcode_for_browser, rendered_path, output_path)
        finally:
            rendered_path.unlink(missing_ok=True)

        processing_time = time.time() - started_at
        self._update_performance_metrics(processing_time)
        return {
            "frames_processed": processed_frames,
            "frame_count": frame_count,
            "fps": fps,
            "class_counts": latest_counts,
            "flow_counts": flow_counts,
            "baseline": baseline.as_dict() if baseline else None,
            "processing_time": processing_time,
        }
    
    async def _save_annotated_image(
        self, 
        image: np.ndarray, 
        vehicles: List[DetectedVehicle], 
        original_path: str,
        baseline: Optional[BaselineConfig] = None,
    ) -> str:
        """Save image with vehicle detection annotations"""
        try:
            # Save annotated image
            output_dir = Path("./output_images")
            output_dir.mkdir(exist_ok=True)
            
            original_name = Path(original_path).stem
            output_path = output_dir / f"{original_name}_annotated.jpg"
            
            cv2.imwrite(str(output_path), self._draw_targets(image.copy(), vehicles, baseline))
            
            return str(output_path)
            
        except Exception as error:
            self.log_error_with_context(error, "save_annotated_image")
            return None
    
    def _update_performance_metrics(self, inference_time: float) -> None:
        """Update performance metrics"""
        self.performance_metrics['total_detections'] += 1
        self.performance_metrics['last_detection_time'] = time.time()
        
        # Calculate rolling average
        current_avg = self.performance_metrics['average_inference_time']
        total_detections = self.performance_metrics['total_detections']
        
        self.performance_metrics['average_inference_time'] = (
            (current_avg * (total_detections - 1) + inference_time) / total_detections
        )
    
    def is_ready(self) -> bool:
        """Check if detector is ready for inference"""
        return self.model_initialized and self.model is not None
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_model_path(self) -> Optional[str]:
        return self.model_path
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up vehicle detector resources")
        self.model = None
        self.model_initialized = False

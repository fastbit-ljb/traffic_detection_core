import { useCallback, useEffect, useRef, useState, type CSSProperties, type DragEvent as ReactDragEvent, type FormEvent, type MouseEvent as ReactMouseEvent } from 'react';
import { flushSync } from 'react-dom';
import { Bar } from 'react-chartjs-2';
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
  type ChartData,
} from 'chart.js';
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Camera,
  Car,
  ChevronRight,
  Cpu,
  StopCircle,
  Clock3,
  Database,
  Eye,
  EyeOff,
  Trash2,
  FileArchive,
  FileImage,
  FolderOpen,
  History,
  Loader2,
  LogOut,
  PackageOpen,
  Play,
  Radio,
  RefreshCw,
  Sparkles,
  Sun,
  Moon,
  Upload,
  UploadCloud,
  Video,
  X,
  Zap,
} from 'lucide-react';
import { PillNav, type PillNavItem } from './PillNav';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { ElasticSlider } from './ui/ElasticSlider';
import { GooeyDeviceSwitch } from './ui/GooeyDeviceSwitch';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_API_URL;
const API_BASE_URL = configuredApiBaseUrl
  ? `${/^https?:\/\//i.test(configuredApiBaseUrl) ? configuredApiBaseUrl : `https://${configuredApiBaseUrl}`}`.replace(/\/+$/, '')
  : window.location.origin;
const AUTH_TOKEN_KEY = 'traffic-auth-token';
const AUTH_CLICK_SOUND_URL = '/audio/login-click.mp3';
const IMAGE_ACCEPT = '.jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp';
const VIDEO_ACCEPT = '.mp4,.avi,.mov,.mkv,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska';
const DATASET_ACCEPT = '.zip,application/zip';
const MODEL_ACCEPT = '.pt';
const CUDA_UNAVAILABLE_MESSAGE = import.meta.env.VITE_CUDA_UNAVAILABLE_MESSAGE ?? 'CUDA 运行时未安装或不可用';

const TARGETS = [
  { key: 'person', label: '行人', color: '#7c3aed' },
  { key: 'car', label: '汽车', color: '#9acfab' },
  { key: 'bus', label: '公交车', color: '#d97706' },
  { key: 'truck', label: '卡车', color: '#2563eb' },
] as const;

const targetColorForTheme = (target: (typeof TARGETS)[number], darkMode: boolean) => target.key === 'car'
  ? (darkMode ? '#21633b' : '#9acfab')
  : target.color;

type TargetKey = (typeof TARGETS)[number]['key'];
type TargetCounts = Record<TargetKey, number>;
type FlowCounts = { entry: TargetCounts; exit: TargetCounts };
type BaselineDirection = 'down' | 'up' | 'right' | 'left';
type BaselineConfig = { enabled: boolean; orientation: 'horizontal' | 'vertical'; direction: BaselineDirection; position: number };
type ThemeMode = 'light' | 'dark';
type ActiveView = 'live' | 'image' | 'video' | 'data' | 'history';
type NavigationItem = { id: ActiveView; label: string; icon: typeof Radio };

const BASELINE_DIRECTIONS: Array<{ value: BaselineDirection; label: string; orientation: BaselineConfig['orientation']; icon: typeof ArrowDown }> = [
  { value: 'down', label: '上进下出', orientation: 'horizontal', icon: ArrowDown },
  { value: 'up', label: '下进上出', orientation: 'horizontal', icon: ArrowUp },
  { value: 'right', label: '左进右出', orientation: 'vertical', icon: ArrowRight },
  { value: 'left', label: '右进左出', orientation: 'vertical', icon: ArrowLeft },
];

const DETECTION_NAV_ITEMS: NavigationItem[] = [
  { id: 'live', label: '实时检测', icon: Radio },
  { id: 'image', label: '图片检测', icon: FileImage },
  { id: 'video', label: '视频检测', icon: Video },
];

const QUICK_NAV_ITEMS: NavigationItem[] = [
  { id: 'data', label: '数据与训练', icon: Database },
  { id: 'history', label: '检测记录', icon: History },
];

const TOP_NAV_ITEMS: PillNavItem[] = DETECTION_NAV_ITEMS.map(({ id, label }) => ({ id, label }));

interface DetectionResult {
  total_vehicles: number;
  class_counts: Partial<TargetCounts>;
  flow_counts?: Partial<FlowCounts>;
  processing_time: number;
  annotated_image_path?: string | null;
  detection_timestamp: string;
  detected_vehicles?: DetectedTarget[];
}

interface DetectedTarget {
  vehicle_type: TargetKey;
  display_label?: string | null;
  confidence: number;
  bounding_box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface ProjectModel {
  id: string;
  name: string;
  path: string;
  source: string;
  is_active: boolean;
  created_at: string;
}

interface Dataset {
  id: string;
  name: string;
  summary: {
    images: { train: number; val: number };
    labels_by_class: TargetCounts;
  };
  created_at: string;
}

interface ModelMetrics {
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  map50?: number | null;
  map50_95?: number | null;
  inference_ms?: number | null;
  evaluation_seconds?: number | null;
}

interface EvaluationItem {
  model_id: string;
  model_name: string;
  source: string;
  metrics?: ModelMetrics;
  error?: string;
}

interface Job {
  id: string;
  kind: 'video' | 'training' | 'benchmark';
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  payload: Record<string, string>;
  result?: {
    output_path?: string;
    model?: ProjectModel;
    flow_counts?: Partial<FlowCounts>;
    metrics?: ModelMetrics;
    run_dir?: string;
    dataset_name?: string;
    items?: EvaluationItem[];
    error?: string;
  } | null;
  flow_counts?: Partial<FlowCounts>;
  created_at: string;
}

interface HistoryEntry {
  id: string;
  media_type: string;
  source_name: string;
  class_counts: TargetCounts;
  total_objects: number;
  processing_time: number;
  output_path?: string | null;
  original_path?: string | null;
  model_name?: string | null;
  created_at: string;
}

interface InferenceDeviceStatus {
  requested_device: 'auto' | 'cpu' | 'cuda';
  active_device: 'cpu' | 'cuda';
  cuda_available: boolean;
  cuda_version?: string | null;
  device_name?: string | null;
}

interface AuthUser {
  id: string;
  username: string;
  created_at: string;
}

interface AuthResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

const emptyCounts = (): TargetCounts => ({ person: 0, car: 0, bus: 0, truck: 0 });
const normalizeCounts = (counts?: Partial<TargetCounts>): TargetCounts => ({ ...emptyCounts(), ...counts });
const emptyFlowCounts = (): FlowCounts => ({ entry: emptyCounts(), exit: emptyCounts() });
const normalizeFlowCounts = (counts?: Partial<FlowCounts> | null): FlowCounts => ({
  entry: normalizeCounts(counts?.entry),
  exit: normalizeCounts(counts?.exit),
});
const targetByKey = (key: TargetKey) => TARGETS.find((target) => target.key === key)!;

type ViewTransitionDocument = Document & {
  startViewTransition?: (callback: () => void) => { ready: Promise<void> };
};

class ApiRequestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

const annotatedImageUrl = (path?: string | null) => {
  const fileName = path?.split(/[\\/]/).pop();
  const token = window.localStorage.getItem(AUTH_TOKEN_KEY);
  return fileName && token ? `${API_BASE_URL}/api/media/images/${encodeURIComponent(fileName)}?token=${encodeURIComponent(token)}` : null;
};

const videoUrl = (path?: string | null, version?: string) => {
  const fileName = path?.split(/[\\/]/).pop();
  const token = window.localStorage.getItem(AUTH_TOKEN_KEY);
  return fileName && token ? `${API_BASE_URL}/api/media/videos/${encodeURIComponent(fileName)}?v=${encodeURIComponent(version ?? fileName)}&token=${encodeURIComponent(token)}` : null;
};

const formatTime = (value?: string) => value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '-';

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = window.localStorage.getItem(AUTH_TOKEN_KEY);
  if (token && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  const payload: { detail?: string } = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = response.status === 429 ? '请求过于频繁，请稍后再试' : payload.detail ?? '请求未完成';
    throw new ApiRequestError(detail, response.status);
  }
  return payload as T;
}

export function TrafficDashboard() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const liveOverlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureBusyRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const cameraSessionRef = useRef(0);
  const imageUploadBusyRef = useRef(false);
  const imageRequestRef = useRef(0);
  const [activeView, setActiveView] = useState<ActiveView>('live');
  const [theme, setTheme] = useState<ThemeMode>(() => window.localStorage.getItem('traffic-dashboard-theme') === 'dark' ? 'dark' : 'light');
  const [cameraLive, setCameraLive] = useState(false);
  const [socketConnected, setSocketConnected] = useState(false);
  const [counts, setCounts] = useState<TargetCounts>(emptyCounts);
  const [liveResult, setLiveResult] = useState<DetectionResult | null>(null);
  const [liveFlowCounts, setLiveFlowCounts] = useState<FlowCounts>(emptyFlowCounts);
  const [baselineConfig, setBaselineConfig] = useState<BaselineConfig>({ enabled: false, orientation: 'horizontal', direction: 'down', position: 0.5 });
  const [baselineSession, setBaselineSession] = useState(0);
  const [imageResult, setImageResult] = useState<DetectionResult | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [models, setModels] = useState<ProjectModel[]>([]);
  const [deviceStatus, setDeviceStatus] = useState<InferenceDeviceStatus | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selectedHistoryEntry, setSelectedHistoryEntry] = useState<HistoryEntry | null>(null);
  const [selectedHistoryIds, setSelectedHistoryIds] = useState<string[]>([]);
  const [deletingHistoryIds, setDeletingHistoryIds] = useState<string[]>([]);
  const [pendingHistoryDeletion, setPendingHistoryDeletion] = useState<HistoryEntry[] | null>(null);
  const [historyDeletionError, setHistoryDeletionError] = useState<string | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<Job[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<Job[]>([]);
  const [comparisonJobs, setComparisonJobs] = useState<Job[]>([]);
  const [videoJob, setVideoJob] = useState<Job | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedBaseModelId, setSelectedBaseModelId] = useState('');
  const [selectedComparisonModelIds, setSelectedComparisonModelIds] = useState<string[]>([]);
  const [trainingConfig, setTrainingConfig] = useState({ epochs: 30, batch: 8, imgsz: 640 });
  const [datasetName, setDatasetName] = useState('');
  const [busyOperation, setBusyOperation] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authForm, setAuthForm] = useState({ username: '', password: '' });
  const [authBusy, setAuthBusy] = useState(false);
  const [authMessage, setAuthMessage] = useState<string | null>(null);
  const [authTransitioning, setAuthTransitioning] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('traffic-dashboard-theme', theme);
  }, [theme]);

  useEffect(() => {
    const token = window.localStorage.getItem(AUTH_TOKEN_KEY);
    if (!token) {
      setAuthReady(true);
      return;
    }
    void api<AuthUser>('/api/auth/me')
      .then(setAuthUser)
      .catch(() => window.localStorage.removeItem(AUTH_TOKEN_KEY))
      .finally(() => setAuthReady(true));
  }, []);

  const submitAuth = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthBusy(true);
    setAuthMessage(null);
    try {
      const response = await api<AuthResponse>(`/api/auth/${authMode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(authForm),
      });
      window.localStorage.setItem(AUTH_TOKEN_KEY, response.access_token);
      const completeAuthentication = () => {
        setAuthUser(response.user);
        setAuthForm({ username: '', password: '' });
        setAuthTransitioning(false);
      };
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        completeAuthentication();
      } else {
        setAuthTransitioning(true);
        window.setTimeout(completeAuthentication, 420);
      }
    } catch (error) {
      setAuthMessage(error instanceof Error ? error.message : '认证失败，请稍后重试');
    } finally {
      setAuthBusy(false);
    }
  };

  const toggleTheme = (event: ReactMouseEvent<HTMLButtonElement>) => {
    const nextTheme: ThemeMode = theme === 'light' ? 'dark' : 'light';
    const commitTheme = () => {
      document.documentElement.dataset.theme = nextTheme;
      flushSync(() => setTheme(nextTheme));
    };
    const transitionDocument = document as ViewTransitionDocument;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!transitionDocument.startViewTransition || reducedMotion) {
      commitTheme();
      return;
    }

    const x = event.clientX;
    const y = event.clientY;
    const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    const transition = transitionDocument.startViewTransition(commitTheme);
    transition.ready.then(() => {
      const clipPath = [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`];
      document.documentElement.animate(
        { clipPath: nextTheme === 'dark' ? clipPath : clipPath.reverse() },
        { duration: 600, easing: 'ease-in-out', pseudoElement: nextTheme === 'dark' ? '::view-transition-new(root)' : '::view-transition-old(root)' },
      );
    });
  };

  const applyLiveResult = useCallback((result: DetectionResult) => {
    setLiveResult(result);
    setCounts(normalizeCounts(result.class_counts));
  }, []);

  const clearLiveResult = useCallback(() => {
    setLiveResult(null);
    setCounts(emptyCounts());
    setLiveFlowCounts(emptyFlowCounts());
    const overlay = liveOverlayCanvasRef.current;
    overlay?.getContext('2d')?.clearRect(0, 0, overlay.width, overlay.height);
  }, []);

  useEffect(() => {
    const canvas = liveOverlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return undefined;

    const draw = () => {
      const context = canvas.getContext('2d');
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!context || !width || !height) return;

      const pixelRatio = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(width * pixelRatio);
      const pixelHeight = Math.round(height * pixelRatio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);

      const targets = liveResult?.detected_vehicles ?? [];
      if (!cameraLive || !video.videoWidth || !video.videoHeight || !targets.length) return;

      // The video uses object-fit: contain, so account for letterboxing before mapping model coordinates.
      const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
      const renderedWidth = video.videoWidth * scale;
      const renderedHeight = video.videoHeight * scale;
      const offsetX = (width - renderedWidth) / 2;
      const offsetY = (height - renderedHeight) / 2;
      const lineWidth = Math.max(2, Math.min(4, width / 320));
      const fontSize = Math.max(11, Math.min(16, width / 52));

      context.lineWidth = lineWidth;
      context.font = `700 ${fontSize}px "Microsoft YaHei", Arial, sans-serif`;
      targets.forEach((target) => {
        const targetMeta = targetByKey(target.vehicle_type);
        const box = target.bounding_box;
        const boxX = offsetX + box.x1 * scale;
        const boxY = offsetY + box.y1 * scale;
        const boxWidth = Math.max(1, (box.x2 - box.x1) * scale);
        const boxHeight = Math.max(1, (box.y2 - box.y1) * scale);
        const color = targetColorForTheme(targetMeta, theme === 'dark');
        const label = `${target.display_label || targetMeta.label} ${(target.confidence * 100).toFixed(0)}%`;
        const labelPadding = 4;
        const labelHeight = fontSize + labelPadding * 2;
        const labelWidth = context.measureText(label).width + labelPadding * 2;
        const labelX = Math.max(0, Math.min(width - labelWidth, boxX));
        const labelY = boxY >= labelHeight ? boxY - labelHeight : boxY;

        context.strokeStyle = color;
        context.strokeRect(boxX, boxY, boxWidth, boxHeight);
        context.fillStyle = color;
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = '#ffffff';
        context.fillText(label, labelX + labelPadding, labelY + fontSize + labelPadding - 1);
      });
    };

    draw();
    const redraw = () => window.requestAnimationFrame(draw);
    window.addEventListener('resize', redraw);
    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(redraw);
    if (resizeObserver) resizeObserver.observe(canvas);
    return () => {
      window.removeEventListener('resize', redraw);
      resizeObserver?.disconnect();
    };
  }, [cameraLive, liveResult, theme]);

  const refreshResources = useCallback(async () => {
    try {
      const [nextModels, nextDatasets, nextTrainingJobs, nextTrainingRuns, nextComparisonJobs, nextDeviceStatus] = await Promise.all([
        api<ProjectModel[]>('/api/models'),
        api<Dataset[]>('/api/datasets'),
        api<Job[]>('/api/training/jobs'),
        api<Job[]>('/api/training/runs'),
        api<Job[]>('/api/experiments'),
        api<InferenceDeviceStatus>('/api/inference-device'),
      ]);
      setModels(nextModels);
      setDatasets(nextDatasets);
      setTrainingJobs(nextTrainingJobs.filter((job) => ['queued', 'running'].includes(job.status)));
      setTrainingRuns(nextTrainingRuns);
      setComparisonJobs(nextComparisonJobs);
      setDeviceStatus(nextDeviceStatus);
      setSelectedDatasetId((current) => current || nextDatasets[0]?.id || '');
      setSelectedBaseModelId((current) => current || nextModels.find((model) => model.is_active)?.id || nextModels[0]?.id || '');
      setSelectedComparisonModelIds((current) => {
        const available = current.filter((id) => nextModels.some((model) => model.id === id));
        return available.length ? available : nextModels.slice(0, 2).map((model) => model.id);
      });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '无法读取项目资源');
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const nextHistory = await api<HistoryEntry[]>('/api/history?limit=50');
      setHistory(nextHistory);
      setSelectedHistoryIds((current) => current.filter((id) => nextHistory.some((entry) => entry.id === id)));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '无法读取检测记录');
    }
  }, []);

  const deleteHistoryEntries = useCallback(async (entries: HistoryEntry[]) => {
    const ids = [...new Set(entries.map((entry) => entry.id))];
    if (!ids.length) return false;
    setHistoryDeletionError(null);
    setDeletingHistoryIds(ids);
    try {
      const result = await api<{ deleted_ids: string[] }>('/api/history', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids }),
      });
      const deletedIds = new Set(Array.isArray(result.deleted_ids) ? result.deleted_ids : []);
      if (!deletedIds.size) {
        const errorMessage = '未删除任何检测记录。记录可能已被移除，或当前账号没有删除权限。';
        await refreshHistory();
        setHistoryDeletionError(errorMessage);
        setMessage(errorMessage);
        return false;
      }
      setHistory((current) => current.filter((item) => !deletedIds.has(item.id)));
      setSelectedHistoryIds((current) => current.filter((id) => !deletedIds.has(id)));
      setSelectedHistoryEntry((current) => current && deletedIds.has(current.id) ? null : current);
      const retainedCount = ids.length - deletedIds.size;
      setMessage(retainedCount > 0
        ? `已删除 ${deletedIds.size} 条检测记录，${retainedCount} 条未删除`
        : deletedIds.size > 1 ? `已删除 ${deletedIds.size} 条检测记录` : '检测记录已删除');
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '删除检测记录失败';
      setHistoryDeletionError(errorMessage);
      setMessage(errorMessage);
      return false;
    } finally {
      setDeletingHistoryIds([]);
    }
  }, [refreshHistory]);

  const requestHistoryDeletion = useCallback((entries: HistoryEntry[]) => {
    const seenIds = new Set<string>();
    const uniqueEntries = entries.filter((entry) => {
      if (seenIds.has(entry.id)) return false;
      seenIds.add(entry.id);
      return true;
    });
    if (uniqueEntries.length) {
      setHistoryDeletionError(null);
      setPendingHistoryDeletion(uniqueEntries);
    }
  }, []);

  const deleteHistory = useCallback((entry: HistoryEntry) => {
    requestHistoryDeletion([entry]);
  }, [requestHistoryDeletion]);

  const deleteSelectedHistory = useCallback(() => {
    requestHistoryDeletion(history.filter((entry) => selectedHistoryIds.includes(entry.id)));
  }, [history, requestHistoryDeletion, selectedHistoryIds]);

  const confirmHistoryDeletion = useCallback(async () => {
    if (!pendingHistoryDeletion) return;
    if (await deleteHistoryEntries(pendingHistoryDeletion)) setPendingHistoryDeletion(null);
  }, [deleteHistoryEntries, pendingHistoryDeletion]);

  const toggleHistorySelection = useCallback((historyId: string, selected: boolean) => {
    setSelectedHistoryIds((current) => selected
      ? (current.includes(historyId) ? current : [...current, historyId])
      : current.filter((id) => id !== historyId));
  }, []);

  const toggleAllHistorySelection = useCallback((selected: boolean) => {
    setSelectedHistoryIds(selected ? history.map((entry) => entry.id) : []);
  }, [history]);

  useEffect(() => {
    if (authUser) void refreshResources();
  }, [authUser, refreshResources]);

  useEffect(() => {
    if (authUser && activeView === 'history') void refreshHistory();
  }, [activeView, authUser, refreshHistory]);

  useEffect(() => {
    if (!authUser) {
      setSocketConnected(false);
      return undefined;
    }
    const token = window.localStorage.getItem(AUTH_TOKEN_KEY);
    if (!token) return undefined;
    const socket = new WebSocket(`${API_BASE_URL.replace(/^http/, 'ws')}/ws/traffic-updates?token=${encodeURIComponent(token)}`);
    socket.onopen = () => setSocketConnected(true);
    socket.onclose = () => setSocketConnected(false);
    socket.onerror = () => setSocketConnected(false);
    socket.onmessage = (event) => {
      const update = JSON.parse(event.data) as { type: string; data: DetectionResult | Job | ProjectModel };
      if (update.type === 'video_progress' || update.type === 'video_completed' || update.type === 'video_failed') {
        setVideoJob(update.data as Job);
      }
    };
    return () => socket.close();
  }, [applyLiveResult, authUser, refreshResources]);

  useEffect(() => {
    if (!videoJob || !['queued', 'running'].includes(videoJob.status)) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const job = await api<Job>(`/api/video-jobs/${videoJob.id}`);
        setVideoJob(job);
      } catch { /* WebSocket will retry on a later update. */ }
    }, 1200);
    return () => window.clearInterval(interval);
  }, [videoJob, refreshResources]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (activeView === 'data') void refreshResources();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [activeView, refreshResources]);

  const submitImage = useCallback(async (
    file: Blob,
    filename: string,
    source: 'image' | 'camera',
    recordHistory: boolean,
    baseline?: BaselineConfig,
    baselineSessionId?: number,
  ) => {
    const formData = new FormData();
    formData.append('image', file, filename);
    formData.append('source', source);
    formData.append('record_history', String(recordHistory));
    if (baseline?.enabled) {
      formData.append('baseline_enabled', 'true');
      formData.append('baseline_orientation', baseline.orientation);
      formData.append('baseline_direction', baseline.direction);
      formData.append('baseline_position', String(baseline.position));
      formData.append('baseline_session', `camera-${baselineSessionId ?? 0}`);
    }
    return api<DetectionResult>('/api/detect-vehicles', { method: 'POST', body: formData });
  }, []);

  const captureCameraFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!cameraActiveRef.current || !video || !canvas || !video.videoWidth || captureBusyRef.current) return;
    const sessionId = cameraSessionRef.current;
    captureBusyRef.current = true;
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      if (!blob) throw new Error('无法读取摄像头画面');
      const result = await submitImage(blob, 'camera-frame.jpg', 'camera', false, baselineConfig, baselineSession);
      if (!cameraActiveRef.current || cameraSessionRef.current !== sessionId) return;
      applyLiveResult(result);
      setLiveFlowCounts(normalizeFlowCounts(result.flow_counts));
      setMessage(null);
    } catch (error) {
      if (cameraActiveRef.current && cameraSessionRef.current === sessionId) {
        setMessage(error instanceof Error ? error.message : '摄像头检测失败');
      }
    } finally {
      captureBusyRef.current = false;
    }
  }, [applyLiveResult, baselineConfig, baselineSession, submitImage]);

  useEffect(() => {
    if (!cameraLive) return undefined;
    const interval = window.setInterval(() => void captureCameraFrame(), 1000);
    return () => window.clearInterval(interval);
  }, [cameraLive, captureCameraFrame]);

  const startCamera = async () => {
    const sessionId = cameraSessionRef.current + 1;
    cameraSessionRef.current = sessionId;
    cameraActiveRef.current = false;
    clearLiveResult();
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        throw new Error('摄像头实时检测需要通过 HTTPS 或本机 localhost 访问当前页面');
      }
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (error) {
        const errorName = error instanceof DOMException ? error.name : '';
        if (!['NotFoundError', 'OverconstrainedError'].includes(errorName)) throw error;
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      }
      if (cameraSessionRef.current !== sessionId) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setBaselineSession((current) => current + 1);
      setLiveFlowCounts(emptyFlowCounts());
      cameraActiveRef.current = true;
      setCameraLive(true);
      setMessage(null);
    } catch (error) {
      if (cameraSessionRef.current !== sessionId) return;
      const errorName = error instanceof DOMException ? error.name : '';
      const message = errorName === 'NotFoundError'
        ? '未检测到摄像头，请连接摄像头后重试'
        : errorName === 'NotAllowedError' || errorName === 'SecurityError'
          ? '摄像头权限被拒绝，请在浏览器地址栏中允许摄像头访问'
          : error instanceof Error ? error.message : '无法打开摄像头';
      setMessage(message);
    }
  };

  const stopCamera = useCallback(() => {
    cameraSessionRef.current += 1;
    cameraActiveRef.current = false;
    captureBusyRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraLive(false);
    clearLiveResult();
    setMessage(null);
  }, [clearLiveResult]);

  const clearImageResult = useCallback(() => {
    imageRequestRef.current += 1;
    imageUploadBusyRef.current = false;
    setBusyOperation((current) => current === 'image' ? null : current);
    setImageResult(null);
    setImageSize(null);
    setImagePreviewUrl(null);
  }, []);

  const changeActiveView = useCallback((nextView: ActiveView) => {
    if (nextView === activeView) return;
    if (activeView === 'live') stopCamera();
    if (activeView === 'image') clearImageResult();
    setMessage(null);
    setActiveView(nextView);
  }, [activeView, clearImageResult, stopCamera]);

  const handleLogout = useCallback(() => {
    stopCamera();
    window.localStorage.removeItem(AUTH_TOKEN_KEY);
    setAuthUser(null);
    setHistory([]);
    setMessage(null);
  }, [stopCamera]);

  useEffect(() => stopCamera, [stopCamera]);

  useEffect(() => () => {
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  }, [imagePreviewUrl]);

  const onImageUpload = async (file: File) => {
    if (imageUploadBusyRef.current) return;
    const requestId = imageRequestRef.current + 1;
    imageRequestRef.current = requestId;
    imageUploadBusyRef.current = true;
    setImageResult(null);
    setImageSize(null);
    setImagePreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(file);
    });
    setBusyOperation('image');
    setMessage(`已导入 ${file.name}，正在检测`);
    try {
      const result = await submitImage(file, file.name, 'image', true);
      if (imageRequestRef.current !== requestId) return;
      setImageResult(result);
      setMessage('图片检测完成，已保存到检测记录');
    } catch (error) {
      if (imageRequestRef.current === requestId) {
        setMessage(error instanceof Error ? error.message : '图片检测失败');
      }
    } finally {
      if (imageRequestRef.current === requestId) {
        imageUploadBusyRef.current = false;
        setBusyOperation(null);
      }
    }
  };

  const onVideoUpload = async (file: File) => {
    setBusyOperation('video');
    try {
      const formData = new FormData();
      formData.append('video', file, file.name);
      if (baselineConfig.enabled) {
        formData.append('baseline_enabled', 'true');
        formData.append('baseline_orientation', baselineConfig.orientation);
        formData.append('baseline_direction', baselineConfig.direction);
        formData.append('baseline_position', String(baselineConfig.position));
      }
      setVideoJob(await api<Job>('/api/detect-video', { method: 'POST', body: formData }));
      setMessage('视频任务已创建');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '视频任务创建失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const updateBaseline = (nextConfig: BaselineConfig) => {
    setBaselineConfig(nextConfig);
    setBaselineSession((current) => current + 1);
    setLiveFlowCounts(emptyFlowCounts());
  };

  const onDatasetUpload = async (file: File) => {
    setBusyOperation('dataset');
    try {
      const formData = new FormData();
      formData.append('archive', file, file.name);
      formData.append('dataset_name', datasetName);
      await api<Dataset>('/api/datasets/import', { method: 'POST', body: formData });
      setDatasetName('');
      await refreshResources();
      setMessage('数据集已完成校验和标准化处理');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '数据集导入失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const onModelUpload = async (file: File) => {
    setBusyOperation('model');
    try {
      const formData = new FormData();
      formData.append('model_file', file, file.name);
      formData.append('model_name', '');
      await api<ProjectModel>('/api/models/upload', { method: 'POST', body: formData });
      await refreshResources();
      setMessage('权重已注册，可在列表中启用');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '权重上传失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const activateModel = async (modelId: string) => {
    setBusyOperation(modelId);
    try {
      await api<ProjectModel>(`/api/models/${modelId}/activate`, { method: 'POST' });
      await refreshResources();
      setMessage('当前检测权重已切换');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '模型切换失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const selectInferenceDevice = async (device: 'cpu' | 'cuda') => {
    setBusyOperation('inference-device');
    try {
      const nextStatus = await api<InferenceDeviceStatus>('/api/inference-device', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device }),
      });
      setDeviceStatus(nextStatus);
      setMessage(`推理设备已切换至 ${nextStatus.active_device === 'cuda' ? 'CUDA' : 'CPU'}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '推理设备切换失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const startTraining = async () => {
    if (!selectedDatasetId || !selectedBaseModelId) {
      setMessage('请选择数据集和基础模型');
      return;
    }
    setBusyOperation('training');
    try {
      await api<Job>('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: selectedDatasetId, base_model_id: selectedBaseModelId, ...trainingConfig }),
      });
      await refreshResources();
      setMessage('训练任务已进入队列');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '训练任务创建失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const toggleComparisonModel = (modelId: string, selected: boolean) => {
    setSelectedComparisonModelIds((current) => selected
      ? (current.includes(modelId) ? current : [...current, modelId])
      : current.filter((id) => id !== modelId));
  };

  const startComparison = async () => {
    if (!selectedDatasetId) {
      setMessage('请选择用于验证的数据集');
      return;
    }
    if (selectedComparisonModelIds.length < 2) {
      setMessage('请至少选择两个模型进行对比');
      return;
    }
    setBusyOperation('comparison');
    try {
      await api<Job>('/api/experiments/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          model_ids: selectedComparisonModelIds,
          batch: trainingConfig.batch,
          imgsz: trainingConfig.imgsz,
        }),
      });
      await refreshResources();
      setMessage('模型对比已进入验证队列');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '模型对比任务创建失败');
    } finally {
      setBusyOperation(null);
    }
  };

  const chartData: ChartData<'bar', number[], string> = {
    labels: TARGETS.map((target) => target.label),
    datasets: [{
      label: '当前数量',
      data: TARGETS.map((target) => counts[target.key]),
      backgroundColor: TARGETS.map((target) => targetColorForTheme(target, theme === 'dark')),
      borderRadius: 4,
      barThickness: 34,
    }],
  };
  // Use the local source image whenever available so the browser can render
  // Chinese labels itself, even while an older backend is still being restarted.
  const imageUrl = imagePreviewUrl ?? annotatedImageUrl(imageResult?.annotated_image_path);
  const processedVideoUrl = videoUrl(videoJob?.result?.output_path, videoJob?.id);

  if (!authReady) return <div className="auth-screen"><section className="auth-loading"><p className="eyebrow">YOLOV8 ROAD OBJECT DETECTION</p><h1>正在检查登录状态</h1></section></div>;
  if (!authUser) return <AuthPanel mode={authMode} form={authForm} busy={authBusy} leaving={authTransitioning} message={authMessage} onModeChange={(mode) => { setAuthMode(mode); setAuthMessage(null); }} onChange={setAuthForm} onSubmit={submitAuth} />;

  return (
    <div className="app-shell app-shell--entering">
      <header className="topbar">
        <div className="topbar-brand">
          <p className="eyebrow">YOLOV8 ROAD OBJECT DETECTION</p>
          <h1>道路车辆与行人检测系统</h1>
        </div>
        <div className="topbar-actions">
          {QUICK_NAV_ITEMS.map(({ id, label, icon: Icon }) => <button key={id} className={activeView === id ? 'icon-button active-tool' : 'icon-button'} type="button" title={label} aria-label={label} onClick={() => changeActiveView(id)}><Icon size={17} aria-hidden="true" /></button>)}
          <button className="icon-button theme-toggle" type="button" title={theme === 'light' ? '切换深色主题' : '切换浅色主题'} aria-label={theme === 'light' ? '切换深色主题' : '切换浅色主题'} onClick={toggleTheme}>
            {theme === 'light' ? <Moon size={17} aria-hidden="true" /> : <Sun size={17} aria-hidden="true" />}
          </button>
          <button className="icon-button" type="button" title="刷新项目数据" onClick={() => void refreshResources()}>
            <RefreshCw size={17} aria-hidden="true" />
          </button>
          <div className="connection-state" aria-label={socketConnected ? '实时连接正常' : '实时连接断开'}>
            <span className={socketConnected ? 'status-dot online' : 'status-dot'} />
            {socketConnected ? '实时连接' : '连接中断'}
          </div>
          <span className="auth-user">{authUser.username}</span>
          <button className="text-button auth-logout" type="button" title="退出登录" onClick={handleLogout}><LogOut size={15} aria-hidden="true" />退出</button>
        </div>
      </header>

      <div className="detection-nav-band">
        <div className="detection-nav-inner">
          <PillNav
            items={TOP_NAV_ITEMS}
            activeId={activeView}
            onSelect={(id) => changeActiveView(id as ActiveView)}
            baseColor={theme === 'dark' ? '#ffffff' : '#111111'}
            pillColor={theme === 'dark' ? '#111111' : '#ffffff'}
            pillTextColor={theme === 'dark' ? '#ffffff' : '#111111'}
            hoveredPillTextColor={theme === 'dark' ? '#111111' : '#ffffff'}
          />
        </div>
      </div>

      <div className="application-layout">
        <main className="workspace">
          {message && <div className="notice" role="status">{message}</div>}

          {activeView === 'live' && (
            <div className="view-grid live-grid">
              <section className="panel media-panel" aria-label="摄像头实时检测">
                <div className="panel-heading">
                  <div><p className="section-kicker">实时输入</p><h2>摄像头检测</h2></div>
                  <div className="control-group">
                    {cameraLive ? (
                      <button className="icon-button danger" type="button" title="关闭摄像头" onClick={stopCamera}><StopCircle size={18} aria-hidden="true" /></button>
                    ) : (
                      <button className="icon-button" type="button" title="打开摄像头" aria-label="打开摄像头" onClick={() => void startCamera()}><Camera size={18} aria-hidden="true" /></button>
                    )}
                  </div>
                </div>
                <div className={cameraLive ? 'video-stage is-live' : 'video-stage is-idle'}>
                  <video ref={videoRef} playsInline muted />
                  <canvas ref={liveOverlayCanvasRef} className="live-detection-overlay" aria-hidden="true" />
                  <BaselineGuide config={baselineConfig} />
                  {!cameraLive && <Radio className="stage-icon" size={36} aria-hidden="true" />}
                </div>
                <canvas ref={canvasRef} className="hidden-canvas" />
              </section>
              <div className="live-stat-stack">
                <MetricsPanel counts={counts} result={liveResult} chartData={chartData} darkMode={theme === 'dark'} flowCounts={baselineConfig.enabled ? liveFlowCounts : undefined} />
              </div>
            </div>
          )}

          {activeView === 'image' && (
            <div className="view-grid image-grid">
              <section className="panel media-panel">
                <div className="panel-heading"><div><p className="section-kicker">单张图片</p><h2>图片检测</h2></div></div>
                <FileDropSurface
                  accept={IMAGE_ACCEPT}
                  label="拖放图片到这里"
                  hint="支持 JPG、PNG、BMP，或点击选择文件"
                  busyLabel="正在检测图片"
                  icon={<FileImage size={40} aria-hidden="true" />}
                  onFile={onImageUpload}
                  onReject={(file) => setMessage(`${file.name} 不是支持的图片格式`)}
                  busy={busyOperation === 'image'}
                  className="video-stage image-stage"
                >
                  {imageUrl ? <div className="detection-image-wrap"><img src={imageUrl} alt="图片检测结果" onLoad={(event) => setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />{imagePreviewUrl && imageSize && imageResult?.detected_vehicles?.map((target, index) => <ChineseAnnotation key={`${target.vehicle_type}-${index}`} target={target} imageSize={imageSize} />)}</div> : null}
                </FileDropSurface>
              </section>
              <MetricsPanel counts={normalizeCounts(imageResult?.class_counts)} result={imageResult} chartData={{ ...chartData, datasets: [{ ...chartData.datasets[0], data: TARGETS.map((target) => normalizeCounts(imageResult?.class_counts)[target.key]) }] }} darkMode={theme === 'dark'} />
            </div>
          )}

          {activeView === 'video' && (
            <div className="view-grid video-grid">
              <section className="panel media-panel">
                <div className="panel-heading"><div><p className="section-kicker">后台任务</p><h2>视频检测</h2></div></div>
                <FileDropSurface
                  accept={VIDEO_ACCEPT}
                  label="拖放视频到这里"
                  hint="支持 MP4、AVI、MOV、MKV，或点击选择文件"
                  busyLabel="正在上传视频"
                  icon={<Video size={40} aria-hidden="true" />}
                  onFile={onVideoUpload}
                  onReject={(file) => setMessage(`${file.name} 不是支持的视频格式`)}
                  busy={busyOperation === 'video'}
                  className="video-drop-surface"
                >
                  {videoJob ? <VideoJobPanel job={videoJob} url={processedVideoUrl} /> : null}
                </FileDropSurface>
              </section>
              <VideoStatusPanel job={videoJob} darkMode={theme === 'dark'} />
            </div>
          )}

          {activeView === 'data' && (
            <>
              <div className="management-grid">
                <section className="panel dataset-panel">
                <div className="panel-heading"><div><p className="section-kicker">YOLO 格式</p><h2>数据集处理</h2></div></div>
                <FileDropSurface accept={DATASET_ACCEPT} label="拖放 ZIP 数据集" hint="松开后自动校验目录结构" busyLabel="正在导入数据集" icon={<FileArchive size={28} aria-hidden="true" />} onFile={onDatasetUpload} onReject={(file) => setMessage(`${file.name} 不是 ZIP 数据集`)} busy={busyOperation === 'dataset'} compact />
                <label className="field-label">数据集名称<input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} placeholder="道路人车数据集" /></label>
                <div className="list-stack">{datasets.map((dataset) => <DatasetRow key={dataset.id} dataset={dataset} selected={selectedDatasetId === dataset.id} onSelect={() => setSelectedDatasetId(dataset.id)} />)}{!datasets.length && <EmptyState icon={<FolderOpen size={28} />} label="导入 ZIP 数据集后显示" />}</div>
                <div className="dataset-format-hint">
                  <strong>导入规范</strong>
                  <p>请将数据集根目录直接压缩为 ZIP，解压后目录应为：</p>
                  <pre>{`dataset.zip
├─ data.yaml
├─ images/train/*.jpg
├─ images/val/*.jpg
├─ labels/train/*.txt
└─ labels/val/*.txt`}</pre>
                  <p><code>data.yaml</code> 至少需指定 <code>train</code>、<code>val</code> 与类别：</p>
                  <pre>{`train: images/train
val: images/val
names: [person, car, bus, truck]`}</pre>
                  <small>支持 JPG、JPEG、PNG 与同名 TXT 标注。每行格式为 <code>类别ID 中心X 中心Y 宽 高</code>，坐标均归一化到 0-1；ID 依次为 0 行人、1 汽车、2 公交车、3 卡车。</small>
                </div>
                </section>
                <section className="panel">
                  <div className="panel-heading"><div><p className="section-kicker">Ultralytics YOLOv8</p><h2>训练任务</h2></div><button className="command-button" type="button" onClick={() => void startTraining()} disabled={busyOperation === 'training'}><Play size={16} aria-hidden="true" />开始训练</button></div>
                  <label className="field-label">数据集<select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
                  <label className="field-label">基础模型<select value={selectedBaseModelId} onChange={(event) => setSelectedBaseModelId(event.target.value)}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label>
                  <div className="numeric-grid">
                    <NumberField label="轮数" value={trainingConfig.epochs} onChange={(value) => setTrainingConfig({ ...trainingConfig, epochs: value })} min={1} max={500} />
                    <NumberField label="批次" value={trainingConfig.batch} onChange={(value) => setTrainingConfig({ ...trainingConfig, batch: value })} min={1} max={128} />
                    <NumberField label="尺寸" value={trainingConfig.imgsz} onChange={(value) => setTrainingConfig({ ...trainingConfig, imgsz: value })} min={320} max={1280} step={32} />
                  </div>
                  <div className="list-stack compact">{trainingJobs.map((job) => <JobRow key={job.id} job={job} />)}{!trainingJobs.length && <EmptyState icon={<PackageOpen size={28} />} label="暂无训练任务" />}</div>
                  <div className="training-results-list">{trainingRuns.filter((job) => job.status === 'completed').slice(0, 2).map((job) => <TrainingRunSummary key={job.id} job={job} />)}</div>
                </section>
              </div>
              <ModelComparisonPanel
                models={models}
                datasets={datasets}
                selectedDatasetId={selectedDatasetId}
                selectedModelIds={selectedComparisonModelIds}
                jobs={comparisonJobs}
                busy={busyOperation === 'comparison'}
                onDatasetChange={setSelectedDatasetId}
                onToggleModel={toggleComparisonModel}
                onStart={() => void startComparison()}
              />
            </>
          )}

          {activeView === 'history' && (
            <section className="panel wide-panel"><div className="panel-heading"><div><p className="section-kicker">SQLite</p><h2>检测记录</h2></div><div className="history-panel-actions"><button className="command-button danger" type="button" title="删除选中检测记录" disabled={!selectedHistoryIds.length || !!deletingHistoryIds.length} onClick={deleteSelectedHistory}><Trash2 size={16} aria-hidden="true" />删除选中（{selectedHistoryIds.length}）</button><button className="icon-button" type="button" title="刷新检测记录" onClick={() => void refreshHistory()}><RefreshCw size={18} /></button></div></div><HistoryTable entries={history} selectedId={selectedHistoryEntry?.id} selectedIds={selectedHistoryIds} deletingIds={deletingHistoryIds} onSelect={setSelectedHistoryEntry} onDelete={deleteHistory} onToggleSelection={toggleHistorySelection} onToggleAllSelection={toggleAllHistorySelection} /></section>
          )}

        </main>
        <aside className="model-side-area">
          <ModelManagementCard
            models={models}
            deviceStatus={deviceStatus}
            uploadBusy={busyOperation === 'model'}
            deviceBusy={busyOperation === 'inference-device'}
            onModelUpload={onModelUpload}
            onUploadReject={(file) => setMessage(`${file.name} 不是 PT 权重文件`)}
            onActivate={activateModel}
            onDeviceSelect={selectInferenceDevice}
            baselineConfig={baselineConfig}
            showBaselineControls={activeView === 'live' || activeView === 'video'}
            onBaselineChange={updateBaseline}
          />
        </aside>
      </div>
      {selectedHistoryEntry && <HistoryDetail entry={selectedHistoryEntry} darkMode={theme === 'dark'} onClose={() => setSelectedHistoryEntry(null)} />}
      {pendingHistoryDeletion && <HistoryDeleteDialog entries={pendingHistoryDeletion} busy={deletingHistoryIds.length > 0} error={historyDeletionError} onCancel={() => { setHistoryDeletionError(null); setPendingHistoryDeletion(null); }} onConfirm={() => void confirmHistoryDeletion()} />}
    </div>
  );
}

function AuthPanel({
  mode,
  form,
  busy,
  leaving,
  message,
  onModeChange,
  onChange,
  onSubmit,
}: {
  mode: 'login' | 'register';
  form: { username: string; password: string };
  busy: boolean;
  leaving: boolean;
  message: string | null;
  onModeChange: (mode: 'login' | 'register') => void;
  onChange: (form: { username: string; password: string }) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const [focusedField, setFocusedField] = useState<'username' | 'password' | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [isTyping, setIsTyping] = useState(false);
  const [isLookingAtEachOther, setIsLookingAtEachOther] = useState(false);
  const [isPurpleBlinking, setIsPurpleBlinking] = useState(false);
  const [isBlackBlinking, setIsBlackBlinking] = useState(false);
  const [isPurplePeeking, setIsPurplePeeking] = useState(false);
  const typingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const peekTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showPasswordRef = useRef(showPassword);
  const passwordRef = useRef(form.password);
  const usernameInputRef = useRef<HTMLInputElement>(null);
  const purpleRef = useRef<HTMLDivElement>(null);
  const blackRef = useRef<HTMLDivElement>(null);
  const orangeRef = useRef<HTMLDivElement>(null);
  const yellowRef = useRef<HTMLDivElement>(null);
  const purpleEyeRef = useRef<HTMLElement>(null);
  const blackEyeRef = useRef<HTMLElement>(null);
  const orangePupilRef = useRef<HTMLElement>(null);
  const yellowPupilRef = useRef<HTMLElement>(null);
  const formReady = form.username.trim().length >= 3 && form.password.length >= 8;
  const playAuthClickSound = () => {
    const sound = new Audio(AUTH_CLICK_SOUND_URL);
    sound.currentTime = 0;
    void sound.play().catch(() => undefined);
  };
  const signalState = formReady ? 'green' : form.username.trim() || form.password ? 'yellow' : 'red';
  const signalLabel = signalState === 'green' ? '登录条件已满足' : signalState === 'yellow' ? '正在填写登录信息' : '等待填写登录信息';

  useEffect(() => {
    showPasswordRef.current = showPassword;
    passwordRef.current = form.password;
  }, [form.password, showPassword]);

  useEffect(() => {
    const onMouseMove = (event: globalThis.MouseEvent) => {
      if (!isTyping && !message) setPointer({ x: event.clientX, y: event.clientY });
    };
    document.addEventListener('mousemove', onMouseMove);
    return () => document.removeEventListener('mousemove', onMouseMove);
  }, [isTyping, message]);

  useEffect(() => {
    let purpleTimer: ReturnType<typeof setTimeout>;
    let blackTimer: ReturnType<typeof setTimeout>;
    const schedulePurpleBlink = () => {
      purpleTimer = setTimeout(() => {
        setIsPurpleBlinking(true);
        purpleTimer = setTimeout(() => {
          setIsPurpleBlinking(false);
          schedulePurpleBlink();
        }, 150);
      }, Math.random() * 4000 + 3000);
    };
    const scheduleBlackBlink = () => {
      blackTimer = setTimeout(() => {
        setIsBlackBlinking(true);
        blackTimer = setTimeout(() => {
          setIsBlackBlinking(false);
          scheduleBlackBlink();
        }, 150);
      }, Math.random() * 4000 + 3000);
    };
    schedulePurpleBlink();
    scheduleBlackBlink();
    return () => {
      clearTimeout(purpleTimer);
      clearTimeout(blackTimer);
    };
  }, []);

  useEffect(() => () => {
    if (typingTimerRef.current) clearTimeout(typingTimerRef.current);
    if (peekTimerRef.current) clearTimeout(peekTimerRef.current);
  }, []);

  const calcPosition = (element: HTMLElement | null) => {
    if (!element) return { faceX: 0, faceY: 0, bodySkew: 0 };
    const rect = element.getBoundingClientRect();
    const dx = pointer.x - (rect.left + rect.width / 2);
    const dy = pointer.y - (rect.top + rect.height / 3);
    return {
      faceX: Math.max(-15, Math.min(15, dx / 20)),
      faceY: Math.max(-10, Math.min(10, dy / 30)),
      bodySkew: Math.max(-6, Math.min(6, -dx / 120)),
    };
  };

  const calcPupilOffset = (element: HTMLElement | null, maxDistance: number) => {
    if (!element) return { x: 0, y: 0 };
    const rect = element.getBoundingClientRect();
    const dx = pointer.x - (rect.left + rect.width / 2);
    const dy = pointer.y - (rect.top + rect.height / 2);
    const distance = Math.min(Math.hypot(dx, dy), maxDistance);
    const angle = Math.atan2(dy, dx);
    return { x: Math.cos(angle) * distance, y: Math.sin(angle) * distance };
  };

  const startUsernameInteraction = () => {
    setFocusedField('username');
    setIsTyping(true);
    setIsLookingAtEachOther(true);
    if (typingTimerRef.current) clearTimeout(typingTimerRef.current);
    typingTimerRef.current = setTimeout(() => setIsLookingAtEachOther(false), 800);
  };

  const endUsernameInteraction = () => {
    setFocusedField(null);
    setIsTyping(false);
    setIsLookingAtEachOther(false);
    if (typingTimerRef.current) clearTimeout(typingTimerRef.current);
  };

  const schedulePurplePeek = () => {
    if (peekTimerRef.current) clearTimeout(peekTimerRef.current);
    if (!showPasswordRef.current || !passwordRef.current) return;
    peekTimerRef.current = setTimeout(() => {
      if (!showPasswordRef.current || !passwordRef.current) return;
      setIsPurplePeeking(true);
      peekTimerRef.current = setTimeout(() => {
        setIsPurplePeeking(false);
        schedulePurplePeek();
      }, 800);
    }, Math.random() * 3000 + 2000);
  };

  const togglePassword = () => {
    setShowPassword((visible) => {
      const nextVisible = !visible;
      showPasswordRef.current = nextVisible;
      if (nextVisible) schedulePurplePeek();
      else {
        if (peekTimerRef.current) clearTimeout(peekTimerRef.current);
        setIsPurplePeeking(false);
      }
      return nextVisible;
    });
  };

  const switchMode = () => {
    onModeChange(mode === 'login' ? 'register' : 'login');
    window.setTimeout(() => usernameInputRef.current?.focus(), 0);
  };

  const purplePosition = calcPosition(purpleRef.current);
  const blackPosition = calcPosition(blackRef.current);
  const orangePosition = calcPosition(orangeRef.current);
  const yellowPosition = calcPosition(yellowRef.current);
  const isLoginError = Boolean(message);
  const isShowingPassword = form.password.length > 0 && showPassword;
  const isLookingAway = focusedField === 'password' && !showPassword;
  const purplePupilOffset = calcPupilOffset(purpleEyeRef.current, 5);
  const blackPupilOffset = calcPupilOffset(blackEyeRef.current, 4);
  const orangePupilOffset = calcPupilOffset(orangePupilRef.current, 5);
  const yellowPupilOffset = calcPupilOffset(yellowPupilRef.current, 5);

  const purpleStyle = isShowingPassword
    ? { transform: 'skewX(0deg)', height: 370 }
    : isLookingAway
      ? { transform: 'skewX(-14deg) translateX(-20px)', height: 410 }
      : isTyping
        ? { transform: `skewX(${purplePosition.bodySkew - 12}deg) translateX(40px)`, height: 410 }
        : { transform: `skewX(${purplePosition.bodySkew}deg)`, height: 370 };
  const blackStyle = isShowingPassword
    ? { transform: 'skewX(0deg)' }
    : isLookingAway
      ? { transform: 'skewX(12deg) translateX(-10px)' }
      : isLookingAtEachOther
        ? { transform: `skewX(${blackPosition.bodySkew * 1.5 + 10}deg) translateX(20px)` }
        : isTyping
          ? { transform: `skewX(${blackPosition.bodySkew * 1.5}deg)` }
          : { transform: `skewX(${blackPosition.bodySkew}deg)` };

  const purpleEyes = isLoginError
    ? { left: 30, top: 55, pupil: { x: -3, y: 4 } }
    : isLookingAway
      ? { left: 20, top: 25, pupil: { x: -5, y: -5 } }
      : isShowingPassword
        ? { left: 20, top: 35, pupil: isPurplePeeking ? { x: 4, y: 5 } : { x: -4, y: -4 } }
        : isLookingAtEachOther
          ? { left: 55, top: 65, pupil: { x: 3, y: 4 } }
          : { left: 45 + purplePosition.faceX, top: 40 + purplePosition.faceY, pupil: purplePupilOffset };
  const blackEyes = isLoginError
    ? { left: 15, top: 40, pupil: { x: -3, y: 4 } }
    : isLookingAway
      ? { left: 10, top: 20, pupil: { x: -4, y: -5 } }
      : isShowingPassword
        ? { left: 10, top: 28, pupil: { x: -4, y: -4 } }
        : isLookingAtEachOther
          ? { left: 32, top: 12, pupil: { x: 0, y: -4 } }
          : { left: 26 + blackPosition.faceX, top: 32 + blackPosition.faceY, pupil: blackPupilOffset };
  const orangeEyes = isLoginError
    ? { left: 60, top: 95, pupil: { x: -3, y: 4 } }
    : isLookingAway
      ? { left: 50, top: 75, pupil: { x: -5, y: -5 } }
      : isShowingPassword
        ? { left: 50, top: 85, pupil: { x: -5, y: -4 } }
        : { left: 82 + orangePosition.faceX, top: 90 + orangePosition.faceY, pupil: orangePupilOffset };
  const yellowEyes = isLoginError
    ? { left: 35, top: 45, pupil: { x: -3, y: 4 } }
    : isLookingAway
      ? { left: 20, top: 30, pupil: { x: -5, y: -5 } }
      : isShowingPassword
        ? { left: 20, top: 35, pupil: { x: -5, y: -4 } }
        : { left: 52 + yellowPosition.faceX, top: 40 + yellowPosition.faceY, pupil: yellowPupilOffset };
  const yellowMouth = isLoginError
    ? { left: 30, top: 92, transform: 'rotate(-8deg)' }
    : isLookingAway
      ? { left: 15, top: 78, transform: 'rotate(0deg)' }
      : isShowingPassword
        ? { left: 10, top: 88, transform: 'rotate(0deg)' }
        : { left: 40 + yellowPosition.faceX, top: 88 + yellowPosition.faceY, transform: 'rotate(0deg)' };
  const orangeMouth = isLoginError
    ? { left: 80 + orangePosition.faceX, top: 130 }
    : { left: 90, top: 120 };

  return <div className="auth-screen">
    <div className="auth-stage">
      <section className={`auth-panel animated-auth-panel auth-panel--${mode} ${message ? 'has-error' : ''} ${leaving ? 'is-leaving' : ''}`} data-focus={focusedField ?? 'idle'} data-showing-password={showPassword ? 'true' : 'false'} aria-label={mode === 'login' ? '用户登录' : '用户注册'}>
        <div className="animated-auth-visual">
          <div className="animated-auth-brand"><Sparkles size={20} aria-hidden="true" /><span>TRAFFIC FLOW</span></div>
          <div className="animated-characters-wrapper">
            <div className="animated-characters-scene" aria-hidden="true">
              <div ref={purpleRef} className="animated-character char-purple" style={purpleStyle}>
                <div className="animated-eyes purple-eyes" style={{ left: purpleEyes.left, top: purpleEyes.top }}><i ref={purpleEyeRef} className="animated-eyeball" style={{ height: isPurpleBlinking ? 2 : 18 }}><b className="animated-pupil" style={{ transform: `translate(${purpleEyes.pupil.x}px, ${purpleEyes.pupil.y}px)` }} /></i><i className="animated-eyeball" style={{ height: isPurpleBlinking ? 2 : 18 }}><b className="animated-pupil" style={{ transform: `translate(${purpleEyes.pupil.x}px, ${purpleEyes.pupil.y}px)` }} /></i></div>
              </div>
              <div ref={blackRef} className="animated-character char-black" style={blackStyle}>
                <div className="animated-eyes black-eyes" style={{ left: blackEyes.left, top: blackEyes.top }}><i ref={blackEyeRef} className="animated-eyeball" style={{ height: isBlackBlinking ? 2 : 16 }}><b className="animated-pupil" style={{ transform: `translate(${blackEyes.pupil.x}px, ${blackEyes.pupil.y}px)` }} /></i><i className="animated-eyeball" style={{ height: isBlackBlinking ? 2 : 16 }}><b className="animated-pupil" style={{ transform: `translate(${blackEyes.pupil.x}px, ${blackEyes.pupil.y}px)` }} /></i></div>
              </div>
              <div ref={orangeRef} className="animated-character char-orange" style={{ transform: isShowingPassword ? 'skewX(0deg)' : `skewX(${orangePosition.bodySkew}deg)` }}>
                <div className="animated-eyes orange-eyes" style={{ left: orangeEyes.left, top: orangeEyes.top }}><b ref={orangePupilRef} className="animated-bare-pupil" style={{ transform: `translate(${orangeEyes.pupil.x}px, ${orangeEyes.pupil.y}px)` }} /><b className="animated-bare-pupil" style={{ transform: `translate(${orangeEyes.pupil.x}px, ${orangeEyes.pupil.y}px)` }} /></div>
                <i className="animated-orange-mouth" style={orangeMouth} />
              </div>
              <div ref={yellowRef} className="animated-character char-yellow" style={{ transform: isShowingPassword ? 'skewX(0deg)' : `skewX(${yellowPosition.bodySkew}deg)` }}>
                <div className="animated-eyes yellow-eyes" style={{ left: yellowEyes.left, top: yellowEyes.top }}><b ref={yellowPupilRef} className="animated-bare-pupil" style={{ transform: `translate(${yellowEyes.pupil.x}px, ${yellowEyes.pupil.y}px)` }} /><b className="animated-bare-pupil" style={{ transform: `translate(${yellowEyes.pupil.x}px, ${yellowEyes.pupil.y}px)` }} /></div>
                <i className="animated-yellow-mouth" style={yellowMouth} />
              </div>
            </div>
          </div>
          <p className="animated-auth-caption">道路目标检测平台</p>
        </div>
        <div className="auth-content">
          <div className="auth-road-scene">
            <div className="auth-road-surface" aria-hidden="true" />
            <div className="auth-car" aria-hidden="true"><Car size={40} strokeWidth={1.7} /></div>
            <div className="auth-crosswalk" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /></div>
            <div className="auth-signal" data-signal={signalState} role="status" aria-label={signalLabel}>
              <i className="auth-signal-lamp auth-signal-lamp--red" />
              <i className="auth-signal-lamp auth-signal-lamp--yellow" />
              <i className="auth-signal-lamp auth-signal-lamp--green" />
            </div>
          </div>
          <div className="auth-form-container">
            <div className="auth-heading">
              <h1>{mode === 'login' ? '欢迎回来！' : '创建账号'}</h1>
              <p>{mode === 'login' ? '请输入你的账号和密码' : '填写账号和密码以开始使用'}</p>
            </div>
            <form className="auth-form" onSubmit={onSubmit}>
              <label className={focusedField === 'username' ? 'auth-input-field focused' : 'auth-input-field'}>
                <span>账号</span>
                <span className="auth-input-wrapper"><input ref={usernameInputRef} value={form.username} autoComplete="username" onFocus={startUsernameInteraction} onBlur={endUsernameInteraction} onChange={(event) => onChange({ ...form, username: event.target.value })} placeholder="输入用户名" minLength={3} maxLength={32} required /></span>
              </label>
              <label className={focusedField === 'password' ? 'auth-input-field focused' : 'auth-input-field'}>
                <span>密码</span>
                <span className="auth-input-wrapper"><input value={form.password} type={showPassword ? 'text' : 'password'} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} onFocus={() => setFocusedField('password')} onBlur={() => setFocusedField(null)} onChange={(event) => onChange({ ...form, password: event.target.value })} placeholder="输入密码" minLength={8} required /><button className="auth-password-toggle" type="button" aria-label={showPassword ? '隐藏密码' : '显示密码'} title={showPassword ? '隐藏密码' : '显示密码'} onMouseDown={(event) => event.preventDefault()} onClick={togglePassword}>{showPassword ? <EyeOff size={20} aria-hidden="true" /> : <Eye size={20} aria-hidden="true" />}</button></span>
              </label>
              {message && <p className="auth-error" role="alert">{message}</p>}
              <button className={formReady ? 'auth-submit is-ready' : 'auth-submit'} type="submit" disabled={busy || leaving || !formReady} onClick={playAuthClickSound}>
                {busy ? <Loader2 className="spin" size={17} aria-hidden="true" /> : <><span className="auth-submit-label">{mode === 'login' ? '登录' : '注册'}</span><span className="auth-submit-hover">{mode === 'login' ? '登录' : '注册'}<ArrowRight size={18} aria-hidden="true" /></span></>}
              </button>
            </form>
            <p className="auth-mode-switch">{mode === 'login' ? '还没有账号？' : '已经有账号？'}<button type="button" onClick={switchMode}>{mode === 'login' ? '注册' : '登录'}</button></p>
          </div>
        </div>
      </section>
    </div>
  </div>;
}

function ChineseAnnotation({ target, imageSize }: { target: DetectedTarget; imageSize: { width: number; height: number } }) {
  const targetMeta = targetByKey(target.vehicle_type);
  const { x1, y1, x2, y2 } = target.bounding_box;
  const style = {
    left: `${(x1 / imageSize.width) * 100}%`,
    top: `${(y1 / imageSize.height) * 100}%`,
    width: `${((x2 - x1) / imageSize.width) * 100}%`,
    height: `${((y2 - y1) / imageSize.height) * 100}%`,
    '--detection-color': targetMeta.color,
  } as CSSProperties;

  return <div className="detection-box" style={style} aria-label={`${targetMeta.label} ${(target.confidence * 100).toFixed(0)}%`}>
    <span>{targetMeta.label} {(target.confidence * 100).toFixed(0)}%</span>
  </div>;
}

function BaselineGuide({ config }: { config: BaselineConfig }) {
  if (!config.enabled) return null;
  const style = config.orientation === 'horizontal'
    ? { top: `${config.position * 100}%` }
    : { left: `${config.position * 100}%` };
  return <span className={`baseline-guide ${config.orientation}`} style={style} aria-hidden="true" />;
}

function BaselineControls({ config, onChange, compact = false }: { config: BaselineConfig; onChange: (config: BaselineConfig) => void; compact?: boolean }) {
  return <div className={compact ? 'baseline-controls model-baseline-controls' : 'baseline-controls'}>
    <label className="baseline-toggle"><input type="checkbox" checked={config.enabled} onChange={(event) => onChange({ ...config, enabled: event.target.checked })} />启用进出基准线</label>
    {config.enabled && <div className="baseline-fields">
      <div className="baseline-direction-options" role="group" aria-label="进出方向">{BASELINE_DIRECTIONS.map(({ value, label, orientation, icon: Icon }) => <button key={value} className={config.direction === value ? 'baseline-direction-option selected' : 'baseline-direction-option'} type="button" title={label} aria-label={label} onClick={() => onChange({ ...config, direction: value, orientation })}><Icon size={17} strokeWidth={2.4} aria-hidden="true" /></button>)}</div>
      <label className="baseline-position-field"><span>位置 <output>{Math.round(config.position * 100)}%</output></span><ElasticSlider value={Math.round(config.position * 100)} min={5} max={95} step={1} ariaLabel="基准线位置" onChange={(value) => onChange({ ...config, position: value / 100 })} /></label>
    </div>}
  </div>;
}

function MetricsPanel({ counts, result, chartData, darkMode, flowCounts }: { counts: TargetCounts; result: DetectionResult | null; chartData: ChartData<'bar', number[], string>; darkMode: boolean; flowCounts?: FlowCounts }) {
  const chartColor = darkMode ? '#d9e2e8' : '#53616e';
  return <section className="panel metrics-panel" aria-label="当前目标统计"><div className="panel-heading"><div><p className="section-kicker">当前帧</p><h2>目标数量</h2></div><span className="frame-total">{result?.total_vehicles ?? 0} 个目标</span></div><div className="stat-strip">{TARGETS.map((target) => <span key={target.key}><small>{target.label}</small><strong>{counts[target.key]}</strong></span>)}</div><div className="chart-area"><Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false }, tooltip: { displayColors: false } }, scales: { x: { grid: { display: false }, border: { display: false }, ticks: { color: chartColor } }, y: { beginAtZero: true, ticks: { precision: 0, color: chartColor }, border: { display: false } } } }} /></div>{flowCounts && <div className="inline-flow-summary" aria-label="进出统计"><div className="flow-strip">{TARGETS.map((target) => <span key={target.key}><b>{target.label}</b><small>入</small><strong>{flowCounts.entry[target.key]}</strong><small>出</small><strong>{flowCounts.exit[target.key]}</strong></span>)}</div></div>}<div className="last-update"><Clock3 size={15} aria-hidden="true" /><span>{result ? `${formatTime(result.detection_timestamp)} · ${(result.processing_time * 1000).toFixed(0)} ms` : '等待检测数据'}</span></div></section>;
}

function FlowMetricsPanel({ flowCounts, embedded = false, darkMode }: { flowCounts: FlowCounts; embedded?: boolean; darkMode: boolean }) {
  const entryTotal = TARGETS.reduce((total, target) => total + flowCounts.entry[target.key], 0);
  const exitTotal = TARGETS.reduce((total, target) => total + flowCounts.exit[target.key], 0);
  const chartColor = darkMode ? '#d9e2e8' : '#53616e';
  const flowChartData: ChartData<'bar', number[], string> = {
    labels: TARGETS.map((target) => target.label),
    datasets: [
      { label: '入库/入区', data: TARGETS.map((target) => flowCounts.entry[target.key]), backgroundColor: darkMode ? '#21633b' : '#9acfab', borderRadius: 4, barThickness: 20 },
      { label: '出库/出区', data: TARGETS.map((target) => flowCounts.exit[target.key]), backgroundColor: '#dc2626', borderRadius: 4, barThickness: 20 },
    ],
  };
  return (
    <section className={embedded ? 'flow-panel embedded' : 'panel flow-panel'} aria-label="进出统计">
      <div className="panel-heading"><div><p className="section-kicker">跨线累计</p><h2>进出统计</h2></div><span className="frame-total">入 {entryTotal} · 出 {exitTotal}</span></div>
      <div className="flow-strip">{TARGETS.map((target) => <span key={target.key}><b>{target.label}</b><small>入</small><strong>{flowCounts.entry[target.key]}</strong><small>出</small><strong>{flowCounts.exit[target.key]}</strong></span>)}</div>
      <div className="chart-area"><Bar data={flowChartData} options={{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { position: 'top', labels: { boxWidth: 10, usePointStyle: true, color: chartColor } }, tooltip: { displayColors: true } },
        scales: { x: { grid: { display: false }, border: { display: false }, ticks: { color: chartColor } }, y: { beginAtZero: true, ticks: { precision: 0, color: chartColor }, border: { display: false } }, },
      }} /></div>
    </section>
  );
}

const fileMatchesAccept = (file: File, accept: string) => accept.split(',').some((rawToken) => {
  const token = rawToken.trim().toLowerCase();
  const fileName = file.name.toLowerCase();
  const fileType = file.type.toLowerCase();
  if (token.startsWith('.')) return fileName.endsWith(token);
  if (token.endsWith('/*')) return fileType.startsWith(token.slice(0, -1));
  return fileType === token;
});

function FileDropSurface({ accept, label, hint, busyLabel, icon, onFile, onReject, busy, compact = false, className = '', children }: {
  accept: string;
  label: string;
  hint: string;
  busyLabel: string;
  icon: React.ReactNode;
  onFile: (file: File) => void | Promise<void>;
  onReject: (file: File) => void;
  busy: boolean;
  compact?: boolean;
  className?: string;
  children?: React.ReactNode;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);
  const [dragging, setDragging] = useState(false);
  const hasFiles = (event: ReactDragEvent<HTMLDivElement>) => event.dataTransfer.types.includes('Files');
  const resetDrag = () => {
    dragDepthRef.current = 0;
    setDragging(false);
  };
  const acceptFile = (file?: File) => {
    if (!file || busy) return;
    if (!fileMatchesAccept(file, accept)) {
      onReject(file);
      return;
    }
    void onFile(file);
  };
  const handleDragEnter = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepthRef.current += 1;
    setDragging(true);
  };
  const handleDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = busy ? 'none' : 'copy';
  };
  const handleDragLeave = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (!dragDepthRef.current) setDragging(false);
  };
  const handleDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    resetDrag();
    acceptFile(file);
  };

  return <div className={`file-drop-surface${compact ? ' compact' : ''}${dragging ? ' is-dragging' : ''}${children ? ' has-content' : ''}${className ? ` ${className}` : ''}`} aria-busy={busy} onDragEnter={handleDragEnter} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
    <input ref={inputRef} className="file-drop-input" type="file" accept={accept} disabled={busy} onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; acceptFile(file); }} />
    {children ?? <button className="file-drop-prompt" type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
      <span className="file-drop-icon">{busy ? <Loader2 className="spin" size={compact ? 26 : 36} aria-hidden="true" /> : icon}</span>
      <span className="file-drop-copy"><strong>{busy ? busyLabel : label}</strong><small>{busy ? '文件正在处理，请稍候' : hint}</small></span>
      {!compact && <span className="file-drop-action">选择文件</span>}
    </button>}
    {dragging && <div className="file-drop-overlay" aria-hidden="true"><UploadCloud size={compact ? 30 : 44} /><strong>{busy ? '当前任务处理中' : '松开即可导入'}</strong><span>{busy ? '完成后可继续添加文件' : label}</span></div>}
    {busy && children && <div className="file-drop-busy"><Loader2 className="spin" size={24} aria-hidden="true" /><span>{busyLabel}</span></div>}
  </div>;
}

function VideoUploadStage({ label = '选择视频开始检测', busy = false }: { label?: string; busy?: boolean }) {
  return <div className="video-stage video-upload-stage" aria-busy={busy}>{busy ? <Loader2 className="stage-icon spin" size={34} aria-hidden="true" /> : <Video className="stage-icon" size={34} aria-hidden="true" />}<span>{label}</span></div>;
}

function VideoProcessingStage({ job }: { job: Job }) {
  const progress = Math.max(0, Math.min(100, Math.round(job.progress)));
  const processing = job.status === 'running';
  return <div className="video-stage video-processing-stage" aria-busy="true">
    <div className="video-processing-loader">
      <div className="video-processing-header"><span className="video-processing-status"><i aria-hidden="true" /><span>{processing ? '正在分析视频' : '视频任务排队中'}</span></span><strong>{progress}%</strong></div>
      <div className="video-processing-track"><span className="video-processing-fill" style={{ width: `${progress}%` }}><i aria-hidden="true" /></span></div>
      <div className="video-processing-ticks" aria-hidden="true">{Array.from({ length: 11 }, (_, index) => <i key={index} />)}</div>
      <div className="video-processing-footer"><span>{job.message}</span><span>{processing ? '实时进度' : '等待处理资源'}</span></div>
    </div>
  </div>;
}

function VideoJobPanel({ job, url }: { job: Job; url: string | null }) {
  const [playbackError, setPlaybackError] = useState(false);
  const isProcessing = ['queued', 'running'].includes(job.status);
  const videoPlayer = job.status === 'completed' && url && !playbackError
    ? <div className="processed-video-stage"><video key={url} className="processed-video" src={url} controls autoPlay muted playsInline preload="auto" onError={() => setPlaybackError(true)} /></div>
    : isProcessing ? <VideoProcessingStage job={job} /> : <VideoUploadStage label={playbackError ? '处理视频无法在当前浏览器中播放' : job.message} />;

  useEffect(() => setPlaybackError(false), [url]);

  return videoPlayer;
}

function VideoStatusPanel({ job, darkMode }: { job: Job | null; darkMode: boolean }) {
  const hasBaseline = job?.payload.baseline_enabled === 'true';
  if (job && hasBaseline) return <FlowMetricsPanel flowCounts={normalizeFlowCounts(job.result?.flow_counts ?? job.flow_counts)} darkMode={darkMode} />;
  return <section className="panel video-status-panel" aria-label="视频任务状态">
    <div className="panel-heading"><div><p className="section-kicker">视频任务</p><h2>检测状态</h2></div></div>
    {job ? <div className="video-task-status"><div className="job-status-line"><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.message}</strong><span>{Math.round(job.progress)}%</span></div><div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div></div> : <EmptyState icon={<Video size={30} />} label="等待选择视频" />}
  </section>;
}

function DatasetRow({ dataset, selected, onSelect }: { dataset: Dataset; selected: boolean; onSelect: () => void }) {
  const counts = normalizeCounts(dataset.summary.labels_by_class);
  return <button className={selected ? 'resource-row selected' : 'resource-row'} type="button" onClick={onSelect}><span><strong>{dataset.name}</strong><small>训练 {dataset.summary.images.train} / 验证 {dataset.summary.images.val}</small></span><span className="mini-counts">{TARGETS.map((target) => <i key={target.key}>{target.label[0]} {counts[target.key]}</i>)}</span></button>;
}

function InferenceDeviceSwitch({ status, busy, onSelect }: { status: InferenceDeviceStatus | null; busy: boolean; onSelect: (device: 'cpu' | 'cuda') => void }) {
  if (!status) return null;
  return <div className="model-device-options"><GooeyDeviceSwitch activeId={status.active_device} ariaLabel="推理设备" onSelect={(device) => void onSelect(device as 'cpu' | 'cuda')} items={[{ id: 'cpu', label: 'CPU', icon: <Cpu size={15} aria-hidden="true" />, disabled: busy }, { id: 'cuda', label: 'CUDA', icon: <Zap size={15} aria-hidden="true" />, disabled: busy || !status.cuda_available, title: status.cuda_available ? '使用 CUDA 加速推理' : CUDA_UNAVAILABLE_MESSAGE }]} /></div>;
}

function ModelManagementCard({ models, deviceStatus, uploadBusy, deviceBusy, onModelUpload, onUploadReject, onActivate, onDeviceSelect, baselineConfig, showBaselineControls, onBaselineChange }: {
  models: ProjectModel[];
  deviceStatus: InferenceDeviceStatus | null;
  uploadBusy: boolean;
  deviceBusy: boolean;
  onModelUpload: (file: File) => void | Promise<void>;
  onUploadReject: (file: File) => void;
  onActivate: (modelId: string) => Promise<void>;
  onDeviceSelect: (device: 'cpu' | 'cuda') => Promise<void>;
  baselineConfig: BaselineConfig;
  showBaselineControls: boolean;
  onBaselineChange: (config: BaselineConfig) => void;
}) {
  const activeModel = models.find((model) => model.is_active);
  return <section className="panel model-management-card" aria-label="模型管理">
    <div className="panel-heading"><div><p className="section-kicker">推理资源</p><h2>模型选择</h2></div></div>
    <FileDropSurface accept={MODEL_ACCEPT} label="拖放 PT 权重" hint="或点击选择模型文件" busyLabel="正在上传权重" icon={<Upload size={27} aria-hidden="true" />} onFile={onModelUpload} onReject={onUploadReject} busy={uploadBusy} compact />
    <InferenceDeviceSwitch status={deviceStatus} busy={deviceBusy} onSelect={(device) => void onDeviceSelect(device)} />
    <div className="field-label model-select-field">
      <span>当前模型</span>
      <Select className="model-dropdown" selectedKey={activeModel?.id ?? null} onSelectionChange={(id) => void onActivate(String(id))} isDisabled={!models.length || uploadBusy} placeholder="暂无可选模型" aria-label="当前模型">
        <SelectTrigger className="model-dropdown-trigger"><SelectValue /></SelectTrigger>
        <SelectContent className="model-dropdown-popover">
          <SelectGroup>
            <SelectLabel>可用模型</SelectLabel>
            {models.map((model) => <SelectItem className="model-dropdown-item" key={model.id} id={model.id} textValue={model.name}><span className="model-dropdown-option"><strong>{model.name}</strong><small>{model.source === 'official' ? '官方预训练' : model.source === 'upload' ? '自定义权重' : model.source}</small></span></SelectItem>)}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
    {activeModel && <div className="selected-model-summary"><span>当前使用</span><small>{activeModel.source} · {formatTime(activeModel.created_at)}</small></div>}
    {showBaselineControls && <BaselineControls config={baselineConfig} onChange={onBaselineChange} compact />}
  </section>;
}

function JobRow({ job }: { job: Job }) {
  return <div className="job-row"><div><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.payload.dataset_name ?? '训练任务'}</strong><small>{job.message}</small></div><span>{Math.round(job.progress)}%</span></div>;
}

const formatMetricPercent = (value?: number | null) => value == null ? '-' : `${(value * 100).toFixed(1)}%`;

function TrainingRunSummary({ job }: { job: Job }) {
  const metrics = job.result?.metrics;
  if (!metrics) return null;
  return <div className="training-run-summary"><div><span className="section-kicker">最近训练结果</span><strong>{job.result?.model?.name ?? job.payload.dataset_name ?? '训练模型'}</strong></div><MetricStrip metrics={metrics} /></div>;
}

function MetricStrip({ metrics }: { metrics: ModelMetrics }) {
  return <div className="metric-strip" aria-label="模型验证指标"><span><small>P</small><strong>{formatMetricPercent(metrics.precision)}</strong></span><span><small>R</small><strong>{formatMetricPercent(metrics.recall)}</strong></span><span><small>F1</small><strong>{formatMetricPercent(metrics.f1)}</strong></span><span><small>mAP50</small><strong>{formatMetricPercent(metrics.map50)}</strong></span><span><small>mAP50-95</small><strong>{formatMetricPercent(metrics.map50_95)}</strong></span></div>;
}

function ModelComparisonPanel({ models, datasets, selectedDatasetId, selectedModelIds, jobs, busy, onDatasetChange, onToggleModel, onStart }: {
  models: ProjectModel[];
  datasets: Dataset[];
  selectedDatasetId: string;
  selectedModelIds: string[];
  jobs: Job[];
  busy: boolean;
  onDatasetChange: (datasetId: string) => void;
  onToggleModel: (modelId: string, selected: boolean) => void;
  onStart: () => void;
}) {
  const latestJobs = jobs.slice(0, 3);
  return <section className="panel comparison-panel" aria-label="模型验证对比">
    <div className="panel-heading"><div><p className="section-kicker">验证集实验</p><h2>模型对比</h2></div><button className="command-button" type="button" disabled={busy || selectedModelIds.length < 2 || !selectedDatasetId} onClick={onStart}>{busy ? <Loader2 className="spin" size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}运行对比</button></div>
    <div className="comparison-controls">
      <label className="field-label">验证数据集<select value={selectedDatasetId} onChange={(event) => onDatasetChange(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
      <div className="comparison-model-picker" role="group" aria-label="选择参与对比的模型">{models.map((model) => <label key={model.id} className={selectedModelIds.includes(model.id) ? 'comparison-model-option selected' : 'comparison-model-option'}><input type="checkbox" checked={selectedModelIds.includes(model.id)} disabled={busy} onChange={(event) => onToggleModel(model.id, event.target.checked)} /><span><strong>{model.name}</strong><small>{model.source}</small></span></label>)}</div>
    </div>
    <div className="comparison-results">{latestJobs.map((job) => <ComparisonJobResult key={job.id} job={job} />)}{!latestJobs.length && <EmptyState icon={<Database size={28} />} label="暂无模型对比结果" />}</div>
  </section>;
}

function ComparisonJobResult({ job }: { job: Job }) {
  const items = job.result?.items ?? [];
  return <div className="comparison-job"><div className="comparison-job-heading"><div><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.result?.dataset_name ?? job.payload.dataset_name ?? '验证集'}</strong></div><small>{formatTime(job.created_at)}</small></div>{['queued', 'running'].includes(job.status) && <div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div>}{job.status === 'failed' && <p className="comparison-error">{job.message}</p>}{items.length > 0 && <ComparisonResultTable items={items} />}</div>;
}

function ComparisonResultTable({ items }: { items: EvaluationItem[] }) {
  const successful = items.filter((item) => !item.error && item.metrics);
  const highestMap = successful.reduce((highest, item) => Math.max(highest, item.metrics?.map50_95 ?? -1), -1);
  return <div className="comparison-table-wrap"><table className="comparison-table"><thead><tr><th>模型</th><th>P</th><th>R</th><th>F1</th><th>mAP50</th><th>mAP50-95</th><th>推理</th></tr></thead><tbody>{items.map((item) => item.error ? <tr key={item.model_id} className="failed"><td>{item.model_name}</td><td colSpan={6}>{item.error}</td></tr> : <tr key={item.model_id} className={item.metrics?.map50_95 === highestMap ? 'best' : ''}><td><strong>{item.model_name}</strong><small>{item.source}</small></td><td>{formatMetricPercent(item.metrics?.precision)}</td><td>{formatMetricPercent(item.metrics?.recall)}</td><td>{formatMetricPercent(item.metrics?.f1)}</td><td>{formatMetricPercent(item.metrics?.map50)}</td><td>{formatMetricPercent(item.metrics?.map50_95)}</td><td>{item.metrics?.inference_ms == null ? '-' : `${item.metrics.inference_ms.toFixed(1)} ms`}</td></tr>)}</tbody></table></div>;
}

function NumberField({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (value: number) => void; min: number; max: number; step?: number }) {
  return <label className="field-label">{label}<input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function HistoryTable({ entries, selectedId, selectedIds, deletingIds, onSelect, onDelete, onToggleSelection, onToggleAllSelection }: { entries: HistoryEntry[]; selectedId?: string; selectedIds: string[]; deletingIds: string[]; onSelect: (entry: HistoryEntry) => void; onDelete: (entry: HistoryEntry) => void; onToggleSelection: (historyId: string, selected: boolean) => void; onToggleAllSelection: (selected: boolean) => void }) {
  if (!entries.length) return <EmptyState icon={<History size={30} />} label="暂无检测记录" />;
  const allSelected = entries.every((entry) => selectedIds.includes(entry.id));
  const selectionLocked = deletingIds.length > 0;
  return <div className="history-table"><div className="history-header"><span className="history-select"><input type="checkbox" aria-label="全选当前检测记录" title="全选当前检测记录" checked={allSelected} disabled={selectionLocked} onChange={(event) => onToggleAllSelection(event.target.checked)} /></span><span>时间</span><span>来源</span><span>目标数量</span><span>模型</span><span>操作</span></div>{entries.map((entry) => <div className={selectedId === entry.id ? 'history-row selected' : 'history-row'} key={entry.id}><span className="history-select"><input type="checkbox" aria-label={`选择 ${entry.source_name} 的检测记录`} checked={selectedIds.includes(entry.id)} disabled={selectionLocked} onChange={(event) => onToggleSelection(entry.id, event.target.checked)} /></span><span>{formatTime(entry.created_at)}</span><span>{entry.media_type} · {entry.source_name}</span><span>{TARGETS.map((target) => `${target.label} ${normalizeCounts(entry.class_counts)[target.key]}`).join('  ')}</span><span>{entry.model_name ?? '-'}</span><span className="history-actions"><button className="history-open" type="button" onClick={() => onSelect(entry)}>查看详情<ChevronRight size={16} aria-hidden="true" /></button><button className="icon-button danger" type="button" title={`删除 ${entry.source_name} 的检测记录`} aria-label={`删除 ${entry.source_name} 的检测记录`} disabled={deletingIds.includes(entry.id)} onClick={() => onDelete(entry)}>{deletingIds.includes(entry.id) ? <Loader2 className="spin" size={16} aria-hidden="true" /> : <Trash2 size={16} aria-hidden="true" />}</button></span></div>)}</div>;
}

function HistoryDetail({ entry, darkMode, onClose }: { entry: HistoryEntry; darkMode: boolean; onClose: () => void }) {
  const counts = normalizeCounts(entry.class_counts);
  const originalUrl = annotatedImageUrl(entry.original_path);
  const resultUrl = annotatedImageUrl(entry.output_path);
  const chartData: ChartData<'bar', number[], string> = {
    labels: TARGETS.map((target) => target.label),
    datasets: [{
      label: '检测数量',
      data: TARGETS.map((target) => counts[target.key]),
      backgroundColor: TARGETS.map((target) => targetColorForTheme(target, darkMode)),
      borderRadius: 4,
      barThickness: 30,
    }],
  };

  return <div className="history-dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
    <section className="history-detail" role="dialog" aria-modal="true" aria-labelledby="history-detail-title">
      <div className="history-detail-heading"><div><p className="section-kicker">检测详情</p><h3 id="history-detail-title">{entry.source_name}</h3></div><button className="icon-button" type="button" title="关闭检测详情" onClick={onClose}><X size={18} aria-hidden="true" /></button></div>
      <div className="history-detail-grid">
        {entry.media_type === 'image' && <><figure className="history-media"><figcaption>原图</figcaption>{originalUrl ? <img src={originalUrl} alt={`${entry.source_name} 原图`} /> : <div className="history-media-empty">此历史记录未保存原图</div>}</figure><figure className="history-media"><figcaption>检测结果</figcaption>{resultUrl ? <img src={resultUrl} alt={`${entry.source_name} 检测结果`} /> : <div className="history-media-empty">检测结果图不可用</div>}</figure></>}
        {entry.media_type !== 'image' && <div className="history-media-empty">暂不支持播放历史视频</div>}
        <div className="history-chart"><p className="section-kicker">目标数量</p><div><Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false }, tooltip: { displayColors: false } }, scales: { x: { grid: { display: false }, border: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0 }, border: { display: false } } } }} /></div><small>{entry.total_objects} 个目标 · {(entry.processing_time * 1000).toFixed(0)} ms</small></div>
      </div>
    </section>
  </div>;
}

function HistoryDeleteDialog({ entries, busy, error, onCancel, onConfirm }: { entries: HistoryEntry[]; busy: boolean; error: string | null; onCancel: () => void; onConfirm: () => void }) {
  const label = entries.length === 1 ? `删除“${entries[0].source_name}”` : `删除 ${entries.length} 条检测记录`;
  return <div className="history-dialog-backdrop" role="presentation" onMouseDown={(event) => { if (!busy && event.currentTarget === event.target) onCancel(); }}>
    <section className="history-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="history-delete-title" aria-describedby="history-delete-description">
      <p className="section-kicker">删除确认</p><h3 id="history-delete-title">{label}</h3><p id="history-delete-description">删除后无法恢复。</p>{error && <p className="history-delete-error" role="alert">{error}</p>}
      <div className="history-delete-actions"><button className="text-button" type="button" disabled={busy} onClick={onCancel}>取消</button><button className="command-button danger" type="button" disabled={busy} onClick={onConfirm}>{busy ? <Loader2 className="spin" size={16} aria-hidden="true" /> : <Trash2 size={16} aria-hidden="true" />}确认删除</button></div>
    </section>
  </div>;
}

function EmptyState({ icon, label }: { icon: React.ReactNode; label: string }) {
  return <div className="empty-state">{icon}<span>{label}</span></div>;
}

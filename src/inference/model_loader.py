"""
Model loading utilities với caching
Supports both PyTorch (.pt) and ONNX (.onnx) models
ONNX models provide 2-3x speedup with same quality
"""
import os
import sys
import builtins
import importlib

# Prevent OpenMP runtime crashes and thread oversubscription on mixed stacks
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # PowerShell can expose the working directory as \\?\D:\..., which some
    # third-party libs don't handle consistently. Normalize it early.
    try:
        _cwd = os.getcwd()
        if _cwd.startswith("\\\\?\\"):
            os.chdir(_cwd[4:])
    except Exception:
        pass

# Keep CPU thread pressure bounded for stable real-time latency
_cpu_count = os.cpu_count() or 4
_cpu_threads = max(1, min(8, _cpu_count // 2))
os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_threads))

try:
    import streamlit as st
except Exception:
    st = None
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.core import config
import torch

# Try to import ONNX support
try:
    import onnxruntime as ort
    from src.inference.onnx_model import ONNXModel
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available - using PyTorch models")
    print("   Install with: pip install onnxruntime-gpu")
ONNX_CUDA_SESSION_OK = None

try:
    torch.set_num_threads(_cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, _cpu_threads // 2))
except Exception:
    pass


def _import_yolo_safely():
    """
    Import Ultralytics with a narrow workaround for Windows path edge-cases.

    Some environments surface OSError(22) for '/etc/os-release' during
    ultralytics import, while ultralytics only expects FileNotFoundError on
    non-Linux systems. Convert just that case so import can continue.
    """
    original_open = builtins.open

    def safe_open(file, *args, **kwargs):
        try:
            return original_open(file, *args, **kwargs)
        except OSError as exc:
            if file == "/etc/os-release" and getattr(exc, "errno", None) == 22:
                raise FileNotFoundError(2, "No such file or directory", file) from exc
            raise

    builtins.open = safe_open
    try:
        return importlib.import_module("ultralytics").YOLO
    finally:
        builtins.open = original_open


YOLO = _import_yolo_safely()


def _get_active_app_state():
    """
    Return the live app_state without re-importing pyqt_app.

    When `pyqt_app.py` is launched as a script, its module is `__main__`.
    Importing `pyqt_app` from here would create a second module instance and
    reset the user's current selection back to defaults.
    """
    main_module = sys.modules.get('__main__')
    if main_module is not None and hasattr(main_module, 'app_state'):
        return getattr(main_module, 'app_state')

    pyqt_module = sys.modules.get('pyqt_app')
    if pyqt_module is not None and hasattr(pyqt_module, 'app_state'):
        return getattr(pyqt_module, 'app_state')

    return None


def _tag_model(model, model_path: str):
    """Attach lightweight metadata used by the PyQt runtime summary."""
    try:
        model._loaded_model_path = model_path
        model._loaded_model_name = os.path.basename(model_path)
    except Exception:
        pass
    return model


def _prefer_gpu_sibling_pt(model_path: str, use_gpu: bool = True) -> str:
    """
    Prefer a sibling `.pt` file when a custom `.onnx` path would otherwise keep
    the pipeline on CPU despite CUDA being available for torch.
    """
    if not use_gpu or not torch.cuda.is_available():
        return model_path

    if not model_path.lower().endswith('.onnx'):
        return model_path

    if ONNX_AVAILABLE and ONNX_CUDA_SESSION_OK is not False:
        return model_path

    sibling_pt = os.path.splitext(model_path)[0] + '.pt'
    if os.path.exists(sibling_pt):
        print(f"[INFO] GPU is available. Prefer sibling PyTorch model for CUDA: {sibling_pt}")
        return sibling_pt

    return model_path


def _load_model(model_path: str, use_gpu: bool = True):
    """
    Load a model - automatically choose ONNX if available, otherwise PyTorch
    
    Args:
        model_path: Path to .pt model file
        use_gpu: Use GPU acceleration
        
    Returns:
        Model object (ONNXModel or YOLO)
    """
    # Check for ONNX version
    onnx_path = model_path.replace('.pt', '.onnx')
    
    if ONNX_AVAILABLE and os.path.exists(onnx_path):
        # Use ONNX for 2-3x speedup
        print(f"🚀 Loading ONNX model: {onnx_path}")
        return _tag_model(ONNXModel(onnx_path, use_gpu=use_gpu), onnx_path)
    else:
        # Fallback to PyTorch
        if ONNX_AVAILABLE:
            print(f"⚠️  ONNX not found, using PyTorch: {model_path}")
            print(f"   Convert with: python UI/convert_models_to_onnx.py")
        return _tag_model(YOLO(model_path), model_path)


def _load_model_runtime_aware(model_path: str, use_gpu: bool = True):
    """
    Load a model and prefer the real GPU path.

    ONNX Runtime on Windows may advertise CUDA support but still fail to load
    the CUDA provider at session-creation time and silently fall back to CPU.
    When that happens and torch CUDA is available, PyTorch CUDA is typically
    faster than ONNX CPU, so prefer the .pt model.
    """
    global ONNX_CUDA_SESSION_OK

    if use_gpu and torch.cuda.is_available() and model_path.lower().endswith('.pt'):
        return _tag_model(YOLO(model_path), model_path)

    if (
        use_gpu
        and torch.cuda.is_available()
        and ONNX_AVAILABLE
        and ONNX_CUDA_SESSION_OK is False
        and model_path.lower().endswith('.pt')
    ):
        return _tag_model(YOLO(model_path), model_path)

    model = _load_model(model_path, use_gpu=use_gpu)

    if not (use_gpu and torch.cuda.is_available() and ONNX_AVAILABLE):
        return model

    try:
        if isinstance(model, ONNXModel):
            provider = model.session.get_providers()[0]
            if provider != 'CUDAExecutionProvider':
                ONNX_CUDA_SESSION_OK = False
                if model_path.lower().endswith('.pt'):
                    print(f"[WARN] ONNX provider for {model_path} is {provider}. Falling back to PyTorch CUDA.")
                    return _tag_model(YOLO(model_path), model_path)
                sibling_pt = os.path.splitext(model_path)[0] + '.pt'
                if os.path.exists(sibling_pt):
                    print(f"[WARN] ONNX provider for {model_path} is {provider}. Falling back to sibling PyTorch CUDA model: {sibling_pt}")
                    return _tag_model(YOLO(sibling_pt), sibling_pt)
                print(f"[WARN] ONNX provider for {model_path} is {provider}. Keeping ONNX model because no .pt fallback was requested.")
            ONNX_CUDA_SESSION_OK = True
    except Exception as exc:
        print(f"[WARN] Failed to inspect ONNX provider for {model_path}: {exc}")

    return model


def load_yolo_models():
    """
    Load cả 2 YOLO models (Person + Vehicle) - NO CACHE for PyQt5
    Automatically uses ONNX models if available for 2-3x speedup
    
    Returns:
        Tuple[YOLO/ONNXModel, YOLO/ONNXModel]: (model_person, model_vehicle)
    """
    # Kiểm tra GPU setting từ config
    if not config.USE_GPU:
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Get model choice from session state or app_state (for PyQt5)
    model_choice = 'YOLOv26n (Fastest)'  # Default
    best_model_choice = 'None (Use base YOLO only)'  # Default
    
    # Prefer the live PyQt state. Importing pyqt_app here would create a new
    # module instance when pyqt_app.py is launched as a script.
    active_app_state = _get_active_app_state()
    if active_app_state is not None:
        model_choice = getattr(active_app_state, 'model_choice', model_choice)
        best_model_choice = getattr(active_app_state, 'best_model_choice', best_model_choice)
    elif st is not None:
        try:
            if st.runtime.exists() and 'model_choice' in st.session_state:
                model_choice = st.session_state.get('model_choice', 'YOLOv8n')
                best_model_choice = st.session_state.get('best_model_choice', 'Train1 (Person only) - ../best.pt')
        except Exception:
            pass
    
    # Map model choice to model file
    model_map = {
        'YOLOv3': 'yolov3u.pt',
        'YOLOv8n': 'yolov8n.pt',
        'YOLOv11n': 'yolo11n.pt',
        'YOLOv11s (Fast)': 'yolo11s.pt',
        'YOLOv26n (Fastest)': 'yolo26n.pt',
        'YOLOv26': 'yolo26n.pt'
    }
    
    base_model = model_map.get(model_choice, 'yolov8n.pt')
    print(f"Loading detector selection -> base: {model_choice} | trained: {best_model_choice}")
    
    # Determine which best.pt to use
    if "None" in best_model_choice:
        # No custom model, use base YOLO for everything
        model_person = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
        model_vehicle = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
        
    elif "Train2" in best_model_choice:
        # Use Train2 best.pt (person, car, bus, truck, motorcycle)
        best_pt_path = '../Train2/best.pt'
        if os.path.exists(best_pt_path):
            # Train2 is a single multi-class detector. Reuse one instance so the
            # processors run one inference pass and keep native class IDs.
            shared_model = _load_model_runtime_aware(best_pt_path, use_gpu=config.USE_GPU)
            model_person = shared_model
            model_vehicle = shared_model
        else:
            # Fallback to base model
            print(f"Warning: Train2 best.pt not found at {best_pt_path}, using base model")
            model_person = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
            model_vehicle = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
            
    elif "Custom:" in best_model_choice:
        # Use custom model path
        custom_path = best_model_choice.replace("Custom: ", "").strip()
        custom_path = _prefer_gpu_sibling_pt(custom_path, use_gpu=config.USE_GPU)
        if os.path.exists(custom_path):
            try:
                # A custom detector chosen from the UI should become the active
                # detector for the whole pipeline instead of being mixed with the
                # default COCO model.
                shared_model = _load_model_runtime_aware(custom_path, use_gpu=config.USE_GPU)
                model_person = shared_model
                model_vehicle = shared_model
            except Exception as e:
                print(f"Error loading custom model: {e}, using base model")
                model_person = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
                model_vehicle = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
        else:
            print(f"Warning: Custom model not found at {custom_path}, using base model")
            model_person = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
            model_vehicle = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
            
    else:
        # Use Train1 best.pt (person only) + base model for vehicles
        if os.path.exists(config.MODEL_PERSON_PATH):
            model_person = _load_model_runtime_aware(config.MODEL_PERSON_PATH, use_gpu=config.USE_GPU)
        else:
            model_person = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
        
        # Load Model gốc cho Vehicle
        model_vehicle = _load_model_runtime_aware(base_model, use_gpu=config.USE_GPU)
    
    # Set device based on config and availability
    if config.USE_GPU and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Only set device for PyTorch models (ONNX handles device internally)
    if hasattr(model_person, 'to'):
        model_person.to(device)
    if model_vehicle is not model_person and hasattr(model_vehicle, 'to'):
        model_vehicle.to(device)
    
    return model_person, model_vehicle


def initialize_tracker(tracker_type='SORT (Fast)'):
    """
    Khởi tạo Tracker (SORT hoặc DeepSort) - NO CACHE for PyQt5
    
    Args:
        tracker_type: 'SORT (Fast) ⭐' or 'DeepSort (Stable)'
    """
    tracker_name = (tracker_type or '').lower()

    if 'byte' in tracker_name:
        try:
            from src.tracking.byte_tracker import ByteTracker
            print("✅ Using ByteTrack")
            return ByteTracker(
                max_age=max(5, int(getattr(config, 'TRACKER_MAX_AGE', 20))),
                min_hits=2,
                match_iou_threshold=0.25,
                high_conf_threshold=0.5,
                low_conf_threshold=0.15,
            )
        except ImportError as e:
            print(f"⚠️  ByteTrack not available: {e}")
            print("   Falling back to SORT...")

    if 'simple' in tracker_name:
        try:
            from src.tracking.simple_tracker import SimpleTracker
            print("✅ Using Simple tracker")
            return SimpleTracker(
                max_age=config.TRACKER_MAX_AGE,
                iou_threshold=0.35
            )
        except ImportError as e:
            print(f"⚠️  Simple tracker not available: {e}")
            print("   Falling back to SORT...")

    if tracker_name.startswith('sort'):
        try:
            from src.tracking.sort_tracker import SORTTracker
            print("✅ Using SORT tracker")
            return SORTTracker(
                max_age=1,
                min_hits=2,
                iou_threshold=0.15
            )
        except ImportError as e:
            print(f"⚠️  SORT tracker not available: {e}")
            print("   Install: pip install scipy filterpy")
            print("   Falling back to DeepSort...")
            # Fallback to DeepSort
    
    # Default: DeepSort
    # Check GPU from config
    use_gpu = config.USE_GPU and torch.cuda.is_available()

    if not use_gpu:
        print("Using DeepSort on CPU (user-selected).")

    print("Using DeepSort tracker")
    tracker = DeepSort(
        max_age=max(5, int(getattr(config, 'TRACKER_MAX_AGE', 1))),
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=64,
        nms_max_overlap=config.TRACKER_NMS_MAX_OVERLAP,
        embedder='mobilenet',
        embedder_gpu=use_gpu
    )

    return tracker


def check_model_exists(model_path: str) -> bool:
    """
    Kiểm tra model file có tồn tại không
    
    Args:
        model_path: Đường dẫn tới model file
        
    Returns:
        True nếu model tồn tại, False nếu không
    """
    return os.path.exists(model_path) and os.path.isfile(model_path)


def get_device_info() -> dict:
    """
    Lấy thông tin về device đang sử dụng
    
    Returns:
        Dictionary chứa thông tin device
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': 'CPU',
        'device_count': 0
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_count'] = torch.cuda.device_count()
    
    return info


def display_model_info():
    """
    Hiển thị thông tin về models và device trong Streamlit
    """
    with st.expander("ℹ️ Thông tin hệ thống", expanded=False):
        device_info = get_device_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ Device")
            if device_info['cuda_available']:
                st.success(f"**GPU**: {device_info['device_name']}")
                st.info(f"**Số lượng GPU**: {device_info['device_count']}")
            else:
                st.info("**Device**: CPU only")
        
        with col2:
            st.markdown("### 🤖 Models")
            if check_model_exists(config.MODEL_PERSON_PATH):
                st.success(f"**Person Model**: Trained (best.pt)")
            else:
                st.warning(f"**Person Model**: YOLOv3 gốc")
            st.info(f"**Vehicle Model**: YOLOv3 gốc")
        
        st.markdown("### ⚙️ Tracker Settings")
        st.text(f"Max Age: {config.TRACKER_MAX_AGE} frames")
        st.text(f"N Init: {config.TRACKER_N_INIT} frames")
        st.text(f"Trail Length: {config.TRAIL_LENGTH} points")

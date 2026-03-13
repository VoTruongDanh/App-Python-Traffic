"""
Model loading utilities với caching
Supports both PyTorch (.pt) and ONNX (.onnx) models
ONNX models provide 2-3x speedup with same quality
"""
import os
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
import torch

# Try to import ONNX support
try:
    import onnxruntime as ort
    from onnx_model import ONNXModel
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available - using PyTorch models")
    print("   Install with: pip install onnxruntime-gpu")


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
        return ONNXModel(onnx_path, use_gpu=use_gpu)
    else:
        # Fallback to PyTorch
        if ONNX_AVAILABLE:
            print(f"⚠️  ONNX not found, using PyTorch: {model_path}")
            print(f"   Convert with: python UI/convert_models_to_onnx.py")
        return YOLO(model_path)


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
    model_choice = 'YOLOv8n'  # Default
    best_model_choice = 'Train1 (Person only) - ../best.pt'  # Default
    
    # Try Streamlit session state first
    if hasattr(st, 'session_state') and 'model_choice' in st.session_state:
        model_choice = st.session_state.get('model_choice', 'YOLOv8n')
        best_model_choice = st.session_state.get('best_model_choice', 'Train1 (Person only) - ../best.pt')
    else:
        # Try PyQt5 app_state
        try:
            from pyqt_app import app_state
            model_choice = app_state.model_choice
            best_model_choice = getattr(app_state, 'best_model_choice', 'Train1 (Person only) - ../best.pt')
        except (ImportError, AttributeError):
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
    
    # Determine which best.pt to use
    if "None" in best_model_choice:
        # No custom model, use base YOLO for everything
        model_person = _load_model(base_model, use_gpu=config.USE_GPU)
        model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
        
    elif "Train2" in best_model_choice:
        # Use Train2 best.pt (person, car, bus, truck, motorcycle)
        best_pt_path = '../Train2/best.pt'
        if os.path.exists(best_pt_path):
            # Train2 model can detect all classes, use it for both
            model_person = _load_model(best_pt_path, use_gpu=config.USE_GPU)
            model_vehicle = _load_model(best_pt_path, use_gpu=config.USE_GPU)
        else:
            # Fallback to base model
            print(f"Warning: Train2 best.pt not found at {best_pt_path}, using base model")
            model_person = _load_model(base_model, use_gpu=config.USE_GPU)
            model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
            
    elif "Custom:" in best_model_choice:
        # Use custom model path
        custom_path = best_model_choice.replace("Custom: ", "").strip()
        if os.path.exists(custom_path):
            # Try to detect if it's a multi-class model
            try:
                temp_model = YOLO(custom_path)
                # If model has 5 classes, assume it's multi-class like Train2
                if hasattr(temp_model, 'names') and len(temp_model.names) == 5:
                    model_person = _load_model(custom_path, use_gpu=config.USE_GPU)
                    model_vehicle = _load_model(custom_path, use_gpu=config.USE_GPU)
                else:
                    # Single class or person-only model
                    model_person = _load_model(custom_path, use_gpu=config.USE_GPU)
                    model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
            except Exception as e:
                print(f"Error loading custom model: {e}, using base model")
                model_person = _load_model(base_model, use_gpu=config.USE_GPU)
                model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
        else:
            print(f"Warning: Custom model not found at {custom_path}, using base model")
            model_person = _load_model(base_model, use_gpu=config.USE_GPU)
            model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
            
    else:
        # Use Train1 best.pt (person only) + base model for vehicles
        if os.path.exists(config.MODEL_PERSON_PATH):
            model_person = _load_model(config.MODEL_PERSON_PATH, use_gpu=config.USE_GPU)
        else:
            model_person = _load_model(base_model, use_gpu=config.USE_GPU)
        
        # Load Model gốc cho Vehicle
        model_vehicle = _load_model(base_model, use_gpu=config.USE_GPU)
    
    # Set device based on config and availability
    if config.USE_GPU and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Only set device for PyTorch models (ONNX handles device internally)
    if hasattr(model_person, 'to'):
        model_person.to(device)
    if hasattr(model_vehicle, 'to'):
        model_vehicle.to(device)
    
    return model_person, model_vehicle


def initialize_tracker(tracker_type='SORT (Fast) ⭐'):
    """
    Khởi tạo Tracker (SORT hoặc DeepSort) - NO CACHE for PyQt5
    
    Args:
        tracker_type: 'SORT (Fast) ⭐' or 'DeepSort (Stable)'
    """
    if 'SORT' in tracker_type:
        try:
            from sort_tracker import SORTTracker
            print("✅ Using SORT tracker")
            return SORTTracker(
                max_age=config.TRACKER_MAX_AGE,
                min_hits=config.TRACKER_N_INIT,
                iou_threshold=0.2  # Giảm từ 0.3 → 0.2 (match dễ hơn khi lag)
            )
        except ImportError as e:
            print(f"⚠️  SORT tracker not available: {e}")
            print("   Install: pip install scipy filterpy")
            print("   Falling back to DeepSort...")
            # Fallback to DeepSort
    
    # Default: DeepSort
    print("✅ Using DeepSort tracker")
    # Kiểm tra GPU từ config
    use_gpu = config.USE_GPU and torch.cuda.is_available()
    
    tracker = DeepSort(
        max_age=config.TRACKER_MAX_AGE,
        n_init=config.TRACKER_N_INIT,
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

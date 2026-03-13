"""
PyQt5 Desktop App - Real-time Object Tracking
High-performance alternative to Streamlit for local use
"""
import config
from model_loader import load_yolo_models, initialize_tracker

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QFileDialog, QLineEdit, QGroupBox, QGridLayout, QTextEdit,
                             QDialog, QProgressBar, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from pathlib import Path

from video_processor import VideoProcessor
# from video_processor_optimized import VideoProcessorOptimized as VideoProcessor
from roi_manager import ROIManager

# Global state for model selection (replacement for st.session_state)
class AppState:
    model_choice = "YOLOv8n"
    tracker_choice = "Simple (Fastest)"
    best_model_choice = "Train1 (Person only) - ../best.pt"  # NEW: best.pt selection

app_state = AppState()


class VideoThread(QThread):
    """Thread for processing video/livestream - SIMPLE & STABLE"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.source = None
        self.processor = None
        self.frame_skip = 0  # No skip - always process latest frame
        self.resize_scale = 100
        self.smooth_mode = False  # Tắt smooth mode để tránh ghost frames
        self.max_det = 20  # Default max detections
        
        self.frame_buffer = []  # DISABLED: Buffer causes 2-3 frame delay
        self.max_buffer_size = 1  # Keep only 1 frame (effectively disabled)
        
    def set_source(self, source):
        self.source = source
        
    def set_processor(self, processor):
        self.processor = processor
        
    def set_params(self, frame_skip, resize_scale):
        self.frame_skip = frame_skip
        self.resize_scale = resize_scale
        
    def stop(self):
        self.running = False
        
    def run(self):
        """Main processing loop - OPTIMIZED FOR RTSP with FFMPEG"""
        if not self.source or not self.processor:
            return
            
        self.running = True
        
        # Detect if RTSP stream
        is_rtsp = isinstance(self.source, str) and self.source.startswith('rtsp://')
        
        # For RTSP, use FFMPEG backend with special flags
        if is_rtsp:
            # Set environment variable for FFMPEG options
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            # Ultra-low latency settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # No buffering
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.source)
            # For HLS/HTTP streams
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            if isinstance(self.source, str) and ('http' in self.source or 'https' in self.source):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not cap.isOpened():
            print(f"Error: Cannot open source {self.source}")
            self.finished.emit()
            return
            
        frame_count = 0
        skip_counter = 0
        last_processed_frame = None
        last_success_time = time.time()
        consecutive_failures = 0
        
        # FRAME PACING: Only for live streams, not video files
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 120:
            source_fps = 30  # Default to 30 FPS
        frame_interval = 1.0 / source_fps
        last_frame_emit_time = time.time()
        
        # Detect source type for pacing
        is_livestream = is_rtsp or (isinstance(self.source, str) and ('http' in self.source or 'https' in self.source))
        is_video_file = not is_livestream and isinstance(self.source, str)
        
        try:
            while self.running:
                # For RTSP, always grab latest frame (discard buffered frames)
                if is_rtsp:
                    # Single grab is enough - reduced from 2 for better latency
                    cap.grab()
                
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    time_since_last = time.time() - last_success_time
                    
                    # For RTSP, reconnect faster
                    if is_rtsp and consecutive_failures > 10:
                        print(f"⚠️ RTSP connection issue, attempting reconnect...")
                        cap.release()
                        time.sleep(0.5)
                        import os
                        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
                        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        consecutive_failures = 0
                        
                        # Reset tracker immediately on reconnect
                        if self.processor and self.processor.tracker:
                            if hasattr(self.processor.tracker, 'tracks'):
                                self.processor.tracker.tracks.clear()
                            if hasattr(self.processor.tracker, 'frame_count'):
                                self.processor.tracker.frame_count = 0
                            if hasattr(self.processor, 'trails'):
                                self.processor.trails.clear()
                        continue
                    
                    # Reset tracker faster - after 1 second of lag (not 5)
                    if time_since_last > 1.0 and consecutive_failures > 10:
                        print(f"⚠️ Lag detected ({time_since_last:.1f}s), resetting tracker...")
                        if self.processor and self.processor.tracker:
                            if hasattr(self.processor.tracker, 'tracks'):
                                self.processor.tracker.tracks.clear()
                            if hasattr(self.processor.tracker, 'frame_count'):
                                self.processor.tracker.frame_count = 0
                            if hasattr(self.processor, 'trails'):
                                self.processor.trails.clear()
                        last_success_time = time.time()
                        consecutive_failures = 0
                    
                    # Don't show old frame when lagging - causes ghost boxes
                    # if last_processed_frame is not None and consecutive_failures < 30:
                    #     self.frame_ready.emit(last_processed_frame, {'total_objects': 0, 'class_counts': {}})
                    
                    time.sleep(0.005)  # Giảm từ 0.01 → 0.005 (nhanh hơn)
                    continue
                
                consecutive_failures = 0
                last_success_time = time.time()
                frame_count += 1
                
                # OPTIMIZED: Process current frame directly (no buffer delay)
                # Old buffer logic caused 2-3 frame delay
                process_frame = frame
                
                try:
                    # Check if using threaded processor
                    if hasattr(self.processor, 'process_frame_threaded'):
                        processed_frame, stats = self.processor.process_frame_threaded(process_frame, self.resize_scale, self.max_det)
                    else:
                        processed_frame, stats = self.processor.process_frame(process_frame, self.resize_scale, self.max_det)
                    
                    last_processed_frame = processed_frame
                    self.frame_ready.emit(processed_frame, stats)
                    
                    # FRAME PACING: Only for livestream to prevent fast-forward
                    # Video files run at max FPS for fastest processing
                    if is_livestream:
                        current_time = time.time()
                        elapsed = current_time - last_frame_emit_time
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        last_frame_emit_time = time.time()
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        finally:
            cap.release()
            self.finished.emit()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Object Tracking - PyQt5")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.video_thread = None
        self.processor = None
        self.model_person = None
        self.model_vehicle = None
        self.tracker = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection settings
        self.max_det = 20  # Default max detections
        self.tracker_max_age = 15  # Default tracker max age
        self.use_threaded_mode = False  # Default: standard mode
        self.use_optimized_mode = False  # Default: standard mode
        self.use_ultra_mode = False  # NEW: Ultra mode for max FPS
        
        # FPS averaging
        self.fps_history = []
        self.fps_history_max = 30  # Keep last 30 readings
        
        # ROI Manager
        self.roi_manager = ROIManager()
        self.roi_drawing_mode = False
        self.roi_temp_points = []
        self.roi_visible = True  # Toggle ROI visibility
        
        # Store current frame info for ROI coordinate mapping
        self.current_frame_size = None  # (width, height) of actual video frame
        
        # Setup UI
        self.setup_ui()
        
        # Load saved settings
        self.load_settings()
        
        # Load models in background
        QTimer.singleShot(100, self.load_models)
        
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #444;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No video loaded")
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.on_video_label_click
        left_panel.addWidget(self.video_label)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.fps_label.setStyleSheet("color: #4ade80; padding: 5px;")
        left_panel.addWidget(self.fps_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_load_video = QPushButton("📁 Load Video")
        self.btn_load_video.clicked.connect(self.load_video_file)
        button_layout.addWidget(self.btn_load_video)
        
        self.btn_start = QPushButton("▶️ Start")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        button_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_stop)
        
        left_panel.addLayout(button_layout)
        
        # Right panel - Controls and stats
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # Model selection
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Best.pt selection
        model_layout.addWidget(QLabel("Trained Model (best.pt):"))
        self.best_model_combo = QComboBox()
        self.best_model_combo.addItems([
            "None (Use base YOLO only)",
            "Train1 (Person only) - ../best.pt",
            "Train2 (Multi-class) - ../Train2/best.pt",
            "Custom (Browse file)..."
        ])
        self.best_model_combo.setCurrentIndex(1)  # Default: Train1
        self.best_model_combo.setToolTip(
            "None: Use base YOLO model only (no custom training)\n"
            "Train1: best.pt trained for person detection only\n"
            "Train2: best.pt trained for person + vehicles (car, bus, truck, motorcycle)\n"
            "Custom: Browse and select your own best.pt file"
        )
        self.best_model_combo.currentTextChanged.connect(self.on_best_model_changed)
        model_layout.addWidget(self.best_model_combo)
        
        # Custom model path (hidden by default)
        self.custom_model_path = None
        
        model_layout.addWidget(QLabel("Base YOLO Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv26n (Fastest)", "YOLOv8n", "YOLOv11n", "YOLOv11s (Fast)", "YOLOv3"])
        self.model_combo.setCurrentIndex(0)  # Default: YOLOv26n
        self.model_combo.setToolTip(
            "YOLOv26n: NEW! Fastest & Edge-optimized (Recommended) 🚀\n"
            "YOLOv8n: Fast & Accurate\n"
            "YOLOv11n: Latest Nano\n"
            "YOLOv11s: Faster than Nano, slightly larger\n"
            "YOLOv3: Classic, Slower\n\n"
            "Note: Used for vehicle detection when Train1 is selected,\n"
            "or for all detection when 'None' is selected"
        )
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        model_layout.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["SORT (Fast) ⭐", "DeepSort (Stable)"])
        self.tracker_combo.setCurrentIndex(0)  # Default: SORT
        self.tracker_combo.setToolTip(
            "SORT: Fast, Kalman Filter + Hungarian (Recommended)\n"
            "DeepSort: Stable, appearance embedding (slower)"
        )
        self.tracker_combo.currentTextChanged.connect(self.on_tracker_changed)
        model_layout.addWidget(self.tracker_combo)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()
        
        detection_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.confidence_slider.sliderReleased.connect(self.auto_reload_if_running)
        detection_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        detection_layout.addWidget(self.confidence_label)
        
        # Box thickness slider
        detection_layout.addWidget(QLabel("Box Thickness:"))
        self.box_thickness_slider = QSlider(Qt.Horizontal)
        self.box_thickness_slider.setMinimum(1)
        self.box_thickness_slider.setMaximum(5)
        self.box_thickness_slider.setValue(2)  # Default: 2px
        self.box_thickness_slider.valueChanged.connect(self.on_box_thickness_changed)
        detection_layout.addWidget(self.box_thickness_slider)
        
        self.box_thickness_label = QLabel("2 px")
        detection_layout.addWidget(self.box_thickness_label)
        
        # Font size slider
        detection_layout.addWidget(QLabel("Font Size:"))
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(72)  # Increased from 24 → 72 for visible effect
        self.font_size_slider.setValue(20)  # Default: 20pt (more visible)
        self.font_size_slider.valueChanged.connect(self.on_font_size_changed)
        detection_layout.addWidget(self.font_size_slider)
        
        self.font_size_label = QLabel("20 pt")
        detection_layout.addWidget(self.font_size_label)
        
        # Font thickness slider (NEW)
        detection_layout.addWidget(QLabel("Font Thickness:"))
        self.font_thickness_slider = QSlider(Qt.Horizontal)
        self.font_thickness_slider.setMinimum(1)
        self.font_thickness_slider.setMaximum(5)
        self.font_thickness_slider.setValue(1)  # Default: 1
        self.font_thickness_slider.valueChanged.connect(self.on_font_thickness_changed)
        self.font_thickness_slider.setToolTip("Độ đậm của chữ (1=mỏng, 5=rất đậm)")
        detection_layout.addWidget(self.font_thickness_slider)
        
        self.font_thickness_label = QLabel("1")
        detection_layout.addWidget(self.font_thickness_label)
        
        # Display mode toggle
        self.btn_display_mode = QPushButton("🎯 Display: Point Label")
        self.btn_display_mode.setCheckable(True)
        self.btn_display_mode.setChecked(True)  # Default: Point mode
        self.btn_display_mode.setStyleSheet("background-color: #3b82f6; color: white;")
        self.btn_display_mode.setToolTip(
            "Point Label: Show center point + label (Faster, cleaner)\n"
            "Bounding Box: Show full box around object (Traditional)"
        )
        self.btn_display_mode.clicked.connect(self.toggle_display_mode)
        detection_layout.addWidget(self.btn_display_mode)
        
        # Trail toggle
        self.btn_trail_toggle = QPushButton("🔴 Trail: OFF")
        self.btn_trail_toggle.setCheckable(True)
        self.btn_trail_toggle.setChecked(False)  # Default: OFF
        self.btn_trail_toggle.setStyleSheet("background-color: #ef4444; color: white;")
        self.btn_trail_toggle.setToolTip("Toggle object movement trail (OFF = faster)")
        self.btn_trail_toggle.clicked.connect(self.toggle_trail)
        detection_layout.addWidget(self.btn_trail_toggle)
        
        # Max detections slider
        detection_layout.addWidget(QLabel("Max Detections:"))
        self.max_det_slider = QSlider(Qt.Horizontal)
        self.max_det_slider.setMinimum(5)
        self.max_det_slider.setMaximum(100)
        self.max_det_slider.setValue(20)  # Default: 20
        self.max_det_slider.valueChanged.connect(self.on_max_det_changed)
        detection_layout.addWidget(self.max_det_slider)
        
        self.max_det_label = QLabel("20 objects")
        detection_layout.addWidget(self.max_det_label)
        
        # Tracker max age slider
        detection_layout.addWidget(QLabel("Tracker Max Age (frames):"))
        self.tracker_age_slider = QSlider(Qt.Horizontal)
        self.tracker_age_slider.setMinimum(1)
        self.tracker_age_slider.setMaximum(30)
        self.tracker_age_slider.setValue(15)  # Default: 15
        self.tracker_age_slider.valueChanged.connect(self.on_tracker_age_changed)
        self.tracker_age_slider.setToolTip(
            "How long to keep track without detection\n"
            "Higher = more stable ID (better for lag)\n"
            "Lower = remove ghost tracks faster"
        )
        detection_layout.addWidget(self.tracker_age_slider)
        
        self.tracker_age_label = QLabel("15 frames")
        detection_layout.addWidget(self.tracker_age_label)
        self.btn_trail_toggle.clicked.connect(self.toggle_trail)
        detection_layout.addWidget(self.btn_trail_toggle)
        
        detection_group.setLayout(detection_layout)
        right_panel.addWidget(detection_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance Optimization")
        perf_layout = QVBoxLayout()
        
        perf_layout.addWidget(QLabel("Frame Skip:"))
        self.frame_skip_slider = QSlider(Qt.Horizontal)
        self.frame_skip_slider.setMinimum(0)
        self.frame_skip_slider.setMaximum(10)
        self.frame_skip_slider.setValue(0)  # Default: 0 (no skip)
        self.frame_skip_slider.sliderReleased.connect(self.auto_reload_if_running)  # Auto-reload
        perf_layout.addWidget(self.frame_skip_slider)
        
        self.frame_skip_label = QLabel("Skip: 0 frames")
        self.frame_skip_slider.valueChanged.connect(
            lambda v: self.frame_skip_label.setText(f"Skip: {v} frames")
        )
        perf_layout.addWidget(self.frame_skip_label)
        
        perf_layout.addWidget(QLabel("Resize Scale:"))
        self.resize_combo = QComboBox()
        self.resize_combo.addItems([
            "100% (Full Quality)",
            "75% (Balanced) ⭐",
            "50% (Fast)",
            "25% (Ultra Fast)"
        ])
        self.resize_combo.setCurrentIndex(0)  # Default: 100%
        self.resize_combo.setToolTip(
            "Resize inference resolution for speed:\n"
            "100%: Full 1920x1080 (Best quality, slower)\n"
            "75%: 1440x810 (Balanced - Recommended)\n"
            "50%: 960x540 (2-3x faster)\n"
            "25%: 480x270 (4-5x faster, lower quality)"
        )
        self.resize_combo.setEnabled(True)  # ENABLED
        perf_layout.addWidget(self.resize_combo)
        
        # Threaded mode toggle
        self.btn_threaded_mode = QPushButton("🚀 Multi-Threading: OFF")
        self.btn_threaded_mode.setCheckable(True)
        self.btn_threaded_mode.setChecked(False)  # Default: OFF
        self.btn_threaded_mode.setStyleSheet("background-color: #ef4444; color: white;")
        self.btn_threaded_mode.setToolTip(
            "Enable parallel inference and drawing\n"
            "ON: 2x faster, use more CPU/GPU\n"
            "OFF: Standard processing"
        )
        self.btn_threaded_mode.clicked.connect(self.toggle_threaded_mode)
        perf_layout.addWidget(self.btn_threaded_mode)
        
        # Optimized mode toggle
        self.btn_optimized_mode = QPushButton("⚡ Optimized Mode: OFF")
        self.btn_optimized_mode.setCheckable(True)
        self.btn_optimized_mode.setChecked(False)  # Default: OFF
        self.btn_optimized_mode.setStyleSheet("background-color: #ef4444; color: white;")
        self.btn_optimized_mode.setToolTip(
            "Enable experimental optimized processor\n"
            "ON: 30+ FPS, minimal drawing, no trails\n"
            "OFF: Standard drawing with trails"
        )
        self.btn_optimized_mode.clicked.connect(self.toggle_optimized_mode)
        perf_layout.addWidget(self.btn_optimized_mode)
        
        # Ultra mode toggle (NEW - highest FPS)
        self.btn_ultra_mode = QPushButton("🔥 Ultra Mode: OFF")
        self.btn_ultra_mode.setCheckable(True)
        self.btn_ultra_mode.setChecked(False)
        self.btn_ultra_mode.setStyleSheet("background-color: #ef4444; color: white;")
        self.btn_ultra_mode.setToolTip(
            "Enable Ultra processor with async double-buffer pipeline\n"
            "ON: 50+ FPS, async inference, lowest latency\n"
            "OFF: Standard or Optimized mode"
        )
        self.btn_ultra_mode.clicked.connect(self.toggle_ultra_mode)
        perf_layout.addWidget(self.btn_ultra_mode)
        
        # Manual cleanup button
        btn_cleanup = QPushButton("🧹 Clear Cache Now")
        btn_cleanup.setStyleSheet("background-color: #f59e0b; color: white;")
        btn_cleanup.setToolTip("Manually clear memory cache\nUse if app becomes slow")
        btn_cleanup.clicked.connect(self.manual_cleanup)
        perf_layout.addWidget(btn_cleanup)
        
        # OPTIMIZATION: Add preset buttons
        perf_layout.addWidget(QLabel("Quick Presets:"))
        preset_layout = QHBoxLayout()
        
        btn_quality = QPushButton("Quality")
        btn_quality.setToolTip("Skip:0 (~12-15 FPS)\nBest quality, slow")
        btn_quality.clicked.connect(lambda: self.apply_preset(0, 100))
        preset_layout.addWidget(btn_quality)
        
        btn_balanced = QPushButton("Balanced")
        btn_balanced.setToolTip("Skip:2 (~25-30 FPS)\nGood balance")
        btn_balanced.clicked.connect(lambda: self.apply_preset(2, 100))
        preset_layout.addWidget(btn_balanced)
        
        btn_speed = QPushButton("Speed")
        btn_speed.setToolTip("Skip:3 (~35-40 FPS)\nFast, smooth")
        btn_speed.clicked.connect(lambda: self.apply_preset(3, 100))
        btn_speed.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold;")
        preset_layout.addWidget(btn_speed)
        
        perf_layout.addLayout(preset_layout)
        
        # Add smooth mode toggle
        self.smooth_mode = False  # Default: OFF (tránh ghost frames)
        btn_smooth_toggle = QPushButton("🎬 Smooth Mode: OFF")
        btn_smooth_toggle.setCheckable(True)
        btn_smooth_toggle.setChecked(False)  # OFF by default
        btn_smooth_toggle.setChecked(True)  # Default: checked
        btn_smooth_toggle.setStyleSheet("background-color: #4ade80; color: white;")
        btn_smooth_toggle.setToolTip(
            "ON: Display last frame when skipping (smoother but less accurate)\n"
            "OFF: Only show processed frames (accurate but may stutter)"
        )
        btn_smooth_toggle.clicked.connect(lambda checked: self.toggle_smooth_mode(checked, btn_smooth_toggle))
        perf_layout.addWidget(btn_smooth_toggle)
        
        perf_group.setLayout(perf_layout)
        right_panel.addWidget(perf_group)
        
        # Livestream input
        stream_group = QGroupBox("Livestream Input")
        stream_layout = QVBoxLayout()
        
        stream_layout.addWidget(QLabel("YouTube/RTSP URL or Webcam ID:"))
        self.stream_input = QComboBox()
        self.stream_input.setEditable(True)
        self.stream_input.setPlaceholderText("https://youtube.com/... or 0 for webcam")
        self.stream_input.setMaxCount(5)  # Keep only 5 items
        self.stream_input.setInsertPolicy(QComboBox.InsertAtTop)
        
        # Load history from file
        self.load_stream_history()
        
        stream_layout.addWidget(self.stream_input)
        
        # Stream quality selector
        stream_layout.addWidget(QLabel("YouTube Stream Quality:"))
        self.stream_quality_combo = QComboBox()
        self.stream_quality_combo.addItems([
            "1080p (Best Quality)",
            "720p (Balanced) ⭐",
            "480p (Stable)",
            "360p (Fastest)"
        ])
        self.stream_quality_combo.setCurrentIndex(1)  # Default 720p
        self.stream_quality_combo.setToolTip(
            "1080p: Highest quality, may lag\n"
            "720p: Balanced (Recommended)\n"
            "480p: Stable, lower quality\n"
            "360p: Fastest, lowest quality"
        )
        stream_layout.addWidget(self.stream_quality_combo)
        
        # Add video type selector
        stream_layout.addWidget(QLabel("Video Type:"))
        self.video_type_combo = QComboBox()
        self.video_type_combo.addItems([
            "🎥 Regular Video (Stream)",
            "🔴 Livestream"
        ])
        self.video_type_combo.setCurrentIndex(0)  # Default: Regular video
        self.video_type_combo.setToolTip(
            "Regular Video: Stream video thường (không tải hết)\n"
            "Livestream: Video đang phát trực tiếp"
        )
        stream_layout.addWidget(self.video_type_combo)
        
        self.btn_start_stream = QPushButton("📡 Start Livestream")
        self.btn_start_stream.clicked.connect(self.start_livestream)
        stream_layout.addWidget(self.btn_start_stream)
        
        stream_group.setLayout(stream_layout)
        right_panel.addWidget(stream_group)
        
        # ROI Settings
        roi_group = QGroupBox("ROI (Region of Interest)")
        roi_layout = QVBoxLayout()
        
        # ROI drawing button
        self.btn_draw_roi = QPushButton("✏️ Draw ROI")
        self.btn_draw_roi.setCheckable(True)
        self.btn_draw_roi.setToolTip("Click to start drawing ROI polygon on video")
        self.btn_draw_roi.clicked.connect(self.toggle_roi_drawing)
        roi_layout.addWidget(self.btn_draw_roi)
        
        # ROI threshold slider
        roi_layout.addWidget(QLabel("ROI Threshold (% object in ROI):"))
        self.roi_threshold_slider = QSlider(Qt.Horizontal)
        self.roi_threshold_slider.setMinimum(0)
        self.roi_threshold_slider.setMaximum(100)
        self.roi_threshold_slider.setValue(50)
        self.roi_threshold_slider.valueChanged.connect(self.on_roi_threshold_changed)
        roi_layout.addWidget(self.roi_threshold_slider)
        
        self.roi_threshold_label = QLabel("50%")
        roi_layout.addWidget(self.roi_threshold_label)
        
        # ROI control buttons
        roi_button_layout = QHBoxLayout()
        
        self.btn_clear_roi = QPushButton("🗑️ Clear")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        roi_button_layout.addWidget(self.btn_clear_roi)
        
        self.btn_toggle_roi = QPushButton("👁️ Hide")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(self.toggle_roi_visibility)
        roi_button_layout.addWidget(self.btn_toggle_roi)
        
        roi_layout.addLayout(roi_button_layout)
        
        # Save/Load buttons
        roi_save_layout = QHBoxLayout()
        
        self.btn_save_roi = QPushButton("💾 Save")
        self.btn_save_roi.clicked.connect(self.save_roi)
        roi_save_layout.addWidget(self.btn_save_roi)
        
        self.btn_load_roi = QPushButton("📂 Load")
        self.btn_load_roi.clicked.connect(self.load_roi)
        roi_save_layout.addWidget(self.btn_load_roi)
        
        roi_layout.addLayout(roi_save_layout)
        
        # ROI status
        self.roi_status_label = QLabel("ROI: Inactive")
        self.roi_status_label.setStyleSheet("color: #666; padding: 5px;")
        roi_layout.addWidget(self.roi_status_label)
        
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        # Current model info
        self.model_info_label = QLabel("Model: Loading...")
        self.model_info_label.setStyleSheet(
            "background-color: #2d2d2d; color: #fbbf24; padding: 8px; "
            "border-radius: 4px; font-weight: bold; font-size: 11px;"
        )
        self.model_info_label.setWordWrap(True)
        stats_layout.addWidget(self.model_info_label)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setStyleSheet("background-color: #1e1e1e; color: #4ade80; font-family: monospace;")
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; padding: 10px;")
        right_panel.addWidget(self.status_label)
        
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 3)
        
        # Wrap right panel in scroll area
        from PyQt5.QtWidgets import QScrollArea
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(right_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMaximumWidth(420)
        
        main_layout.addWidget(scroll_area, 1)
        
    def load_models(self):
        """Load YOLO models and tracker"""
        self.status_label.setText("Loading models...")
        QApplication.processEvents()
        
        try:
            # Set model choice in app state
            app_state.model_choice = self.model_combo.currentText()
            app_state.tracker_choice = self.tracker_combo.currentText()
            
            self.model_person, self.model_vehicle = load_yolo_models()
            self.tracker = initialize_tracker(self.tracker_combo.currentText())
            
            # Create processor (ultra, optimized, threaded, or standard)
            if self.use_ultra_mode:
                from video_processor_ultra import UltraVideoProcessor
                self.processor = UltraVideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("✅ Using Ultra Processor (Async Pipeline - Max FPS)")
            elif self.use_optimized_mode:
                from video_processor_optimized import VideoProcessorOptimized
                self.processor = VideoProcessorOptimized(self.model_person, self.model_vehicle, self.tracker)
                print("✅ Using Optimized Processor (Zero-Copy)")
            elif self.use_threaded_mode:
                from video_processor_threaded import ThreadedVideoProcessor
                self.processor = ThreadedVideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("✅ Using Threaded Processor (Parallel)")
            else:
                self.processor = VideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("✅ Using Standard Processor")
                
            self.processor.set_confidence(self.confidence_slider.value() / 100.0)
            
            # Connect ROI manager to processor
            self.processor.set_roi_manager(self.roi_manager)
            
            # Update model info display
            self.update_model_info()
            
            self.status_label.setText(f"✅ Models loaded: {app_state.model_choice}")
            self.btn_load_video.setEnabled(True)
            self.btn_start_stream.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"❌ Error loading models: {e}")
    
    def update_model_info(self):
        """Update the model info display"""
        try:
            # Check if using ONNX
            using_onnx = False
            onnx_available = False
            gpu_provider = False
            try:
                from onnx_model import ONNXModel
                import onnxruntime
                onnx_available = True
                if isinstance(self.model_person, ONNXModel) or isinstance(self.model_vehicle, ONNXModel):
                    using_onnx = True
                    # Check if GPU provider
                    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                        gpu_provider = True
            except:
                pass
            
            # Get model names
            base_model = app_state.model_choice
            best_model = app_state.best_model_choice
            tracker_type = app_state.tracker_choice
            
            # Build info text
            info_lines = []
            
            # Model info - handle Custom case
            if "Custom:" in best_model:
                # Extract filename from "Custom: /path/to/file.onnx"
                custom_path = best_model.replace("Custom: ", "").strip()
                custom_name = Path(custom_path).name
                info_lines.append(f"📦 Custom: {custom_name}")
            elif "None" in best_model:
                info_lines.append(f"📦 {base_model}")
            elif "Train2" in best_model:
                info_lines.append(f"📦 Train2 + {base_model}")
            elif "Train1" in best_model:
                info_lines.append(f"📦 Train1 + {base_model}")
            else:
                info_lines.append(f"📦 {base_model}")
            
            # ONNX status with GPU info
            if using_onnx:
                if gpu_provider:
                    info_lines.append("🚀 ONNX GPU (Fast)")
                else:
                    info_lines.append("🚀 ONNX CPU")
            elif onnx_available:
                info_lines.append("⚠️ PyTorch (Slow)")
            else:
                info_lines.append("⚠️ PyTorch")
            
            # Tracker info
            if "Simple" in tracker_type:
                info_lines.append("⚡ Simple Tracker")
            else:
                info_lines.append("🎯 DeepSort Tracker")
            
            # Update label
            self.model_info_label.setText("\n".join(info_lines))
            
        except Exception as e:
            self.model_info_label.setText(f"Model: {app_state.model_choice}")
            
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.status_label.setText(f"🔄 Switching to {model_name}...")
        QApplication.processEvents()  # Update UI immediately
        
        # Update app state
        app_state.model_choice = model_name
        
        # Reload models (no cache in PyQt5)
        try:
            self.model_person, self.model_vehicle = load_yolo_models()
            if self.processor:
                self.processor.model_person = self.model_person
                self.processor.model_vehicle = self.model_vehicle
                # Update using_train2 flag
                self.processor.using_train2 = "Train2" in app_state.best_model_choice
            
            # Update model info display
            self.update_model_info()
            
            self.status_label.setText(f"✅ Switched to {model_name}")
            
            # Auto-reload if running
            self.auto_reload_if_running()
        except Exception as e:
            self.status_label.setText(f"❌ Error loading {model_name}: {e}")
    
    def on_best_model_changed(self, best_model_name):
        """Handle best.pt model selection change"""
        # Check if user selected "Custom"
        if "Custom" in best_model_name:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Custom Model", "", 
                "Model Files (*.pt *.onnx);;PyTorch Models (*.pt);;ONNX Models (*.onnx);;All Files (*.*)"
            )
            
            if file_path:
                self.custom_model_path = file_path
                self.status_label.setText(f"🔄 Loading custom model: {Path(file_path).name}...")
                QApplication.processEvents()
                
                # Update app state with custom path
                app_state.best_model_choice = f"Custom: {file_path}"
            else:
                # User cancelled, revert to previous selection
                self.best_model_combo.blockSignals(True)
                self.best_model_combo.setCurrentIndex(1)  # Revert to Train1
                self.best_model_combo.blockSignals(False)
                return
        else:
            self.custom_model_path = None
            self.status_label.setText(f"🔄 Switching to {best_model_name}...")
            QApplication.processEvents()
            
            # Update app state
            app_state.best_model_choice = best_model_name
        
        # Reload models (no cache in PyQt5)
        try:
            self.model_person, self.model_vehicle = load_yolo_models()
            if self.processor:
                self.processor.model_person = self.model_person
                self.processor.model_vehicle = self.model_vehicle
                # Update using_train2 flag
                self.processor.using_train2 = "Train2" in app_state.best_model_choice
            
            # Check if using ONNX
            onnx_status = ""
            try:
                from onnx_model import ONNXModel
                if isinstance(self.model_person, ONNXModel) or isinstance(self.model_vehicle, ONNXModel):
                    onnx_status = " 🚀 ONNX"
            except:
                pass
            
            # Update status message based on selection
            if "None" in best_model_name:
                self.status_label.setText(f"✅ Using base YOLO only (no custom training){onnx_status}")
            elif "Train2" in best_model_name:
                self.status_label.setText(f"✅ Using Train2 best.pt (multi-class detection){onnx_status}")
            elif "Train1" in best_model_name:
                self.status_label.setText(f"✅ Using Train1 best.pt (person only){onnx_status}")
            elif "Custom" in best_model_name:
                self.status_label.setText(f"✅ Using custom model: {Path(self.custom_model_path).name}{onnx_status}")
            
            # Update model info display
            self.update_model_info()
            
            # Auto-reload if running
            self.auto_reload_if_running()
        except Exception as e:
            self.status_label.setText(f"❌ Error loading {best_model_name}: {e}")
        
    def on_tracker_changed(self, tracker_name):
        """Handle tracker selection change"""
        if self.processor:
            self.tracker = initialize_tracker(tracker_name)
            self.processor.tracker = self.tracker
            self.status_label.setText(f"Tracker changed to {tracker_name}")
            
            # Auto-reload if running
            self.auto_reload_if_running()
            
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        conf = value / 100.0
        self.confidence_label.setText(f"{conf:.2f}")
        if self.processor:
            self.processor.set_confidence(conf)
    
    def on_box_thickness_changed(self, value):
        """Handle box thickness change"""
        self.box_thickness_label.setText(f"{value} px")
        if self.processor:
            self.processor.set_box_thickness(value)
    
    def on_font_size_changed(self, value):
        """Handle font size change"""
        self.font_size_label.setText(f"{value} pt")
        if self.processor:
            self.processor.set_font_size(value)
    
    def on_font_thickness_changed(self, value):
        """Handle font thickness change"""
        self.font_thickness_label.setText(f"{value}")
        if self.processor and hasattr(self.processor, 'set_font_thickness'):
            self.processor.set_font_thickness(value)
    
    def toggle_display_mode(self, checked):
        """Toggle between Point Label and Bounding Box mode"""
        if self.processor:
            self.processor.set_point_mode(checked)
            
        # Update button text and style
        if checked:
            self.btn_display_mode.setText("🎯 Display: Point Label")
            self.btn_display_mode.setStyleSheet("background-color: #3b82f6; color: white;")
        else:
            self.btn_display_mode.setText("📦 Display: Bounding Box")
            self.btn_display_mode.setStyleSheet("background-color: #8b5cf6; color: white;")
    
    def toggle_trail(self, checked):
        """Toggle trail drawing"""
        import config
        if checked:
            config.TRAIL_LENGTH = 3  # Enable trail
            self.btn_trail_toggle.setText("🟢 Trail: ON")
            self.btn_trail_toggle.setStyleSheet("background-color: #22c55e; color: white;")
        else:
            config.TRAIL_LENGTH = 0  # Disable trail
            self.btn_trail_toggle.setText("🔴 Trail: OFF")
            self.btn_trail_toggle.setStyleSheet("background-color: #ef4444; color: white;")
    
    def manual_cleanup(self):
        """Manual memory cleanup"""
        self.status_label.setText("🧹 Cleaning cache...")
        QApplication.processEvents()
        
        try:
            # Clear processor cache
            if self.processor:
                self.processor.trails.clear()
                if hasattr(self.processor.tracker, 'tracks'):
                    self.processor.tracker.tracks.clear()
                self.processor.frame_counter = 0
            
            # Clear Python garbage
            import gc
            gc.collect()
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✅ CUDA cache cleared")
            except:
                pass
            
            self.status_label.setText("✅ Cache cleared!")
            QTimer.singleShot(2000, lambda: self.status_label.setText("▶️ Running"))
        except Exception as e:
            self.status_label.setText(f"⚠️ Cleanup error: {e}")
    
    def toggle_threaded_mode(self, checked):
        """Toggle threaded processing mode"""
        self.use_threaded_mode = checked
        if checked:
            self.btn_threaded_mode.setText("🚀 Multi-Threading: ON")
            self.btn_threaded_mode.setStyleSheet("background-color: #22c55e; color: white;")
        else:
            self.btn_threaded_mode.setText("🚀 Multi-Threading: OFF")
            self.btn_threaded_mode.setStyleSheet("background-color: #ef4444; color: white;")
        
        # Reload if running
        if self.video_thread and self.video_thread.isRunning():
            self.status_label.setText("🔄 Switching mode...")
            QApplication.processEvents()
            self.stop_processing()
            QTimer.singleShot(500, self.start_processing)
    
    def on_max_det_changed(self, value):
        """Handle max detections change"""
        self.max_det = value
        self.max_det_label.setText(f"{value} objects")
    
    def on_tracker_age_changed(self, value):
        """Handle tracker max age change"""
        self.tracker_max_age = value
        self.tracker_age_label.setText(f"{value} frames")
        # Update config
        import config
        config.TRACKER_MAX_AGE = value
        # Update tracker if exists
        if self.tracker:
            if hasattr(self.tracker, 'max_age'):
                self.tracker.max_age = value
    
    def auto_reload_if_running(self):
        """Auto-reload stream/video if currently running"""
        if self.video_thread and self.video_thread.isRunning():
            # Visual feedback
            self.status_label.setText("🔄 Applying changes...")
            self.status_label.setStyleSheet("color: #fbbf24; font-weight: bold;")
            QApplication.processEvents()
            
            # Stop current processing
            self.stop_processing()
            
            # Restart after short delay with success message
            def restart_with_feedback():
                self.start_processing()
                self.status_label.setText("✅ Settings applied!")
                self.status_label.setStyleSheet("color: #4ade80; font-weight: bold;")
                QTimer.singleShot(2000, lambda: self.status_label.setStyleSheet("color: #666; padding: 10px;"))
            
            QTimer.singleShot(300, restart_with_feedback)
    
    def apply_preset(self, frame_skip, resize_scale):
        """Apply performance preset"""
        self.frame_skip_slider.setValue(frame_skip)
        
        # Map resize_scale to combo index
        resize_map = {100: 0, 75: 1, 50: 2, 25: 3}
        if resize_scale in resize_map:
            self.resize_combo.setCurrentIndex(resize_map[resize_scale])
        
        self.status_label.setText(f"✅ Preset applied: Skip={frame_skip}, Resize={resize_scale}%")
        
        # Auto-reload if running (will be triggered by slider/combo change events)
    
    def toggle_optimized_mode(self, checked):
        """Toggle optimized processor mode"""
        self.use_optimized_mode = checked
        if checked:
            self.btn_optimized_mode.setText("⚡ Optimized Mode: ON")
            self.btn_optimized_mode.setStyleSheet("background-color: #4ade80; color: white;")
            self.status_label.setText("✅ Optimized mode enabled (reloading...)")
            
            # Disable threaded mode (mutually exclusive recommendation)
            if self.use_threaded_mode:
                self.btn_threaded_mode.setChecked(False)
                self.toggle_threaded_mode(False)
        else:
            self.btn_optimized_mode.setText("⚡ Optimized Mode: OFF")
            self.btn_optimized_mode.setStyleSheet("background-color: #ef4444; color: white;")
            self.status_label.setText("✅ Optimized mode disabled (reloading...)")
        
        # Reload to apply changes (must recreate processor)
        self.load_models()
        self.auto_reload_if_running()
    
    def toggle_ultra_mode(self, checked):
        """Toggle ultra processor mode (async double-buffer pipeline)"""
        self.use_ultra_mode = checked
        if checked:
            self.btn_ultra_mode.setText("🔥 Ultra Mode: ON")
            self.btn_ultra_mode.setStyleSheet("background-color: #f97316; color: white;")  # Orange for ultra
            self.status_label.setText("✅ Ultra mode enabled (50+ FPS, reloading...)")
            
            # Disable other modes (mutually exclusive)
            if self.use_optimized_mode:
                self.btn_optimized_mode.setChecked(False)
                self.use_optimized_mode = False
                self.btn_optimized_mode.setText("⚡ Optimized Mode: OFF")
                self.btn_optimized_mode.setStyleSheet("background-color: #ef4444; color: white;")
            if self.use_threaded_mode:
                self.btn_threaded_mode.setChecked(False)
                self.use_threaded_mode = False
                self.btn_threaded_mode.setText("🚀 Multi-Threading: OFF")
                self.btn_threaded_mode.setStyleSheet("background-color: #ef4444; color: white;")
        else:
            self.btn_ultra_mode.setText("🔥 Ultra Mode: OFF")
            self.btn_ultra_mode.setStyleSheet("background-color: #ef4444; color: white;")
            self.status_label.setText("✅ Ultra mode disabled (reloading...)")
        
        # Reload to apply changes
        self.load_models()
        self.auto_reload_if_running()
    
    def toggle_smooth_mode(self, checked, button):
        """Toggle smooth mode for frame interpolation"""
        self.smooth_mode = checked
        if checked:
            button.setText("🎬 Smooth Mode: ON")
            button.setStyleSheet("background-color: #4ade80; color: white;")
            self.status_label.setText("✅ Smooth mode enabled (frame interpolation)")
        else:
            button.setText("🎬 Smooth Mode: OFF")
            button.setStyleSheet("")
            self.status_label.setText("✅ Smooth mode disabled (accurate frames only)")
        
        # Update thread if running
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.smooth_mode = self.smooth_mode
    
    def toggle_roi_drawing(self, checked):
        """Toggle ROI drawing mode"""
        self.roi_drawing_mode = checked
        if checked:
            self.btn_draw_roi.setText("✏️ Drawing... (Click to add points)")
            self.btn_draw_roi.setStyleSheet("background-color: #fbbf24; color: white;")
            self.status_label.setText("🖊️ Click on video to draw ROI polygon. Right-click to finish.")
            self.roi_temp_points = []
        else:
            self.btn_draw_roi.setText("✏️ Draw ROI")
            self.btn_draw_roi.setStyleSheet("")
            if len(self.roi_temp_points) >= 3:
                # Finalize ROI
                self.roi_manager.set_points(self.roi_temp_points)
                self.update_roi_status()
                self.status_label.setText(f"✅ ROI set with {len(self.roi_temp_points)} points")
            else:
                self.status_label.setText("⚠️ ROI needs at least 3 points")
            self.roi_temp_points = []
    
    def on_video_label_click(self, event):
        """Handle mouse clicks on video label for ROI drawing"""
        if not self.roi_drawing_mode:
            return
        
        # Must have current frame size
        if self.current_frame_size is None:
            self.status_label.setText("⚠️ Wait for video to start")
            return
        
        # Get click position relative to video label
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # Get current pixmap
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return
        
        # Get label dimensions
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        # Get actual frame size (from current_frame_size, not pixmap)
        frame_width, frame_height = self.current_frame_size
        
        # Calculate how the frame is scaled to fit in the label
        # The frame is scaled to fit while maintaining aspect ratio
        scale_w = label_width / frame_width
        scale_h = label_height / frame_height
        scale = min(scale_w, scale_h)
        
        # Calculate actual displayed size
        display_width = int(frame_width * scale)
        display_height = int(frame_height * scale)
        
        # Calculate offset (centering)
        offset_x = (label_width - display_width) // 2
        offset_y = (label_height - display_height) // 2
        
        # Check if click is within displayed video area
        if click_x < offset_x or click_x > offset_x + display_width:
            self.status_label.setText(f"⚠️ Click inside video (x out of bounds: {click_x})")
            return
        if click_y < offset_y or click_y > offset_y + display_height:
            self.status_label.setText(f"⚠️ Click inside video (y out of bounds: {click_y})")
            return
        
        # Convert to video coordinates (original frame size)
        video_x = int((click_x - offset_x) / scale)
        video_y = int((click_y - offset_y) / scale)
        
        # Clamp to video bounds (safety check)
        video_x = max(0, min(video_x, frame_width - 1))
        video_y = max(0, min(video_y, frame_height - 1))
        
        # Right click to finish
        if event.button() == Qt.RightButton:
            if len(self.roi_temp_points) >= 3:
                self.btn_draw_roi.setChecked(False)
                self.toggle_roi_drawing(False)
            else:
                self.status_label.setText("⚠️ Need at least 3 points. Keep clicking.")
            return
        
        # Left click to add point
        if event.button() == Qt.LeftButton:
            self.roi_temp_points.append((video_x, video_y))
            self.status_label.setText(
                f"🖊️ Point {len(self.roi_temp_points)}: ({video_x}, {video_y}) | "
                f"Frame: {frame_width}x{frame_height} | Scale: {scale:.3f}"
            )
            print(f"DEBUG: Point {len(self.roi_temp_points)}: Video({video_x}, {video_y}) <- Click({click_x}, {click_y}) | "
                  f"Offset({offset_x}, {offset_y}) | Display({display_width}x{display_height}) | Scale({scale:.3f})")
    
    def on_roi_threshold_changed(self, value):
        """Handle ROI threshold change"""
        threshold = value / 100.0
        self.roi_threshold_label.setText(f"{value}%")
        self.roi_manager.set_threshold(threshold)
        self.update_roi_status()
    
    def toggle_roi_visibility(self, checked):
        """Toggle ROI overlay visibility"""
        self.roi_visible = not checked
        self.roi_manager.visible = self.roi_visible
        
        if checked:
            self.btn_toggle_roi.setText("👁️ Show")
            self.status_label.setText("🙈 ROI overlay hidden (filtering still active)")
        else:
            self.btn_toggle_roi.setText("👁️ Hide")
            self.status_label.setText("👁️ ROI overlay visible")
    
    def clear_roi(self):
        """Clear ROI"""
        self.roi_manager.clear()
        self.roi_temp_points = []
        self.btn_draw_roi.setChecked(False)
        self.roi_drawing_mode = False
        self.btn_draw_roi.setText("✏️ Draw ROI")
        self.btn_draw_roi.setStyleSheet("")
        self.update_roi_status()
        self.status_label.setText("🗑️ ROI cleared")
    
    def save_roi(self):
        """Save ROI configuration to file"""
        if not self.roi_manager.is_active():
            self.status_label.setText("⚠️ No ROI to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Configuration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            try:
                config_data = self.roi_manager.get_config()
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                self.status_label.setText(f"💾 ROI saved to {Path(file_path).name}")
            except Exception as e:
                self.status_label.setText(f"❌ Error saving ROI: {e}")
    
    def load_roi(self):
        """Load ROI configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI Configuration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                self.roi_manager.load_config(config_data)
                
                # Update UI
                if 'threshold' in config_data:
                    threshold_value = int(config_data['threshold'] * 100)
                    self.roi_threshold_slider.setValue(threshold_value)
                
                self.update_roi_status()
                self.status_label.setText(f"📂 ROI loaded from {Path(file_path).name}")
            except Exception as e:
                self.status_label.setText(f"❌ Error loading ROI: {e}")
    
    def update_roi_status(self):
        """Update ROI status label"""
        if self.roi_manager.is_active():
            points_count = len(self.roi_manager.roi_points)
            threshold = int(self.roi_manager.threshold * 100)
            self.roi_status_label.setText(f"ROI: Active ({points_count} points, {threshold}% threshold)")
            self.roi_status_label.setStyleSheet("color: #4ade80; padding: 5px; font-weight: bold;")
        else:
            self.roi_status_label.setText("ROI: Inactive")
            self.roi_status_label.setStyleSheet("color: #666; padding: 5px;")
            
    def load_video_file(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        
        if file_path:
            self.video_source = file_path
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.btn_start.setEnabled(True)
            
    def start_livestream(self):
        """Start livestream processing"""
        stream_url = self.stream_input.currentText().strip()
        
        if not stream_url:
            self.status_label.setText("⚠️ Please enter stream URL or webcam ID")
            return
        
        # Save to history
        self.save_stream_to_history(stream_url)
            
        # Check if webcam ID
        if stream_url.isdigit():
            self.video_source = int(stream_url)
        else:
            # Handle YouTube URL  
            if 'youtube.com' in stream_url or 'youtu.be' in stream_url:
                try:
                    import yt_dlp
                    
                    self.status_label.setText("🔄 Checking YouTube URL...")
                    QApplication.processEvents()
                    
                    # Get quality
                    quality_text = self.stream_quality_combo.currentText()
                    if "1080p" in quality_text:
                        quality_name = "1080p"
                        height = 1080
                    elif "720p" in quality_text:
                        quality_name = "720p"
                        height = 720
                    elif "480p" in quality_text:
                        quality_name = "480p"
                        height = 480
                    else:
                        quality_name = "360p"
                        height = 360
                    
                    # First check if it's a livestream
                    ydl_opts_check = {
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts_check) as ydl:
                        info = ydl.extract_info(stream_url, download=False)
                        title = info.get('title', 'Unknown')[:30]
                        is_live = info.get('is_live', False)
                    
                    if is_live:
                        # LIVESTREAM - use URL directly (OpenCV will handle)
                        self.status_label.setText(f"🔴 Live: {title}... (Getting URL)")
                        QApplication.processEvents()
                        
                        ydl_opts = {
                            'format': f'best[height<={height}]',
                            'quiet': True,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(stream_url, download=False)
                            url = info['url']
                        
                        self.video_source = url
                        self.status_label.setText(f"✅ 🔴 Live {quality_name}: {title}...")
                    else:
                        # REGULAR VIDEO - must download
                        self.status_label.setText(f"⏬ Downloading: {title}...")
                        QApplication.processEvents()
                        
                        import tempfile
                        import os
                        
                        # Download to temp folder
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        
                        import random
                        import string
                        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                        temp_file = temp_dir / f"downloaded_{random_id}.mp4"
                        
                        ydl_opts = {
                            'format': f'best[height<={height}][ext=mp4]/best[height<={height}]',
                            'outtmpl': str(temp_file),
                            'quiet': False,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([stream_url])
                        
                        self.video_source = str(temp_file)
                        self.status_label.setText(f"✅ 🎥 Video {quality_name}: {title}...")
                        
                except Exception as e:
                    self.status_label.setText(f"❌ Error: {str(e)[:50]}")
                    print(f"Full error: {e}")
                    return
            else:
                self.video_source = stream_url
                
        self.start_processing()
        
    def start_processing(self):
        """Start video processing"""
        if not self.processor:
            self.status_label.setText("⚠️ Models not loaded")
            return
            
        # Stop existing thread if running
        if self.video_thread and self.video_thread.isRunning():
            self.stop_processing()
            
        # Get parameters
        frame_skip = self.frame_skip_slider.value()
        resize_text = self.resize_combo.currentText().split()[0]  # Get "100%" from "100% (Full)"
        resize_scale = int(resize_text.replace('%', ''))
        
        # Reset processor statistics
        self.processor.reset_statistics()
        
        # Create and start thread
        self.video_thread = VideoThread()
        self.video_thread.set_source(self.video_source)
        self.video_thread.set_processor(self.processor)
        self.video_thread.set_params(frame_skip, resize_scale)
        self.video_thread.max_det = self.max_det  # Pass max_det
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.finished.connect(self.on_processing_finished)
        self.video_thread.start()
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_start_stream.setEnabled(False)
        self.status_label.setText("▶️ Processing...")
        
        # Reset FPS counter
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = []  # Clear FPS history
        
    def stop_processing(self):
        """Stop video processing"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            
    def on_processing_finished(self):
        """Handle processing finished"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load_video.setEnabled(True)
        self.btn_start_stream.setEnabled(True)
        self.status_label.setText("⏹️ Stopped")
        
    def update_frame(self, frame, stats):
        """Update video display with processed frame - OPTIMIZED"""
        # Store original frame size for ROI coordinate mapping
        original_h, original_w = frame.shape[:2]
        self.current_frame_size = (original_w, original_h)
        
        # Draw temporary ROI points if in drawing mode (on original frame)
        if self.roi_drawing_mode and len(self.roi_temp_points) > 0:
            # Draw temporary points and lines on the frame
            for i, point in enumerate(self.roi_temp_points):
                # Draw larger circles for visibility
                cv2.circle(frame, point, 8, (0, 255, 255), -1)  # Yellow fill
                cv2.circle(frame, point, 10, (0, 0, 0), 2)  # Black outline
                
                # Draw point number
                cv2.putText(frame, str(i+1), (point[0]+12, point[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw lines between points
                if i > 0:
                    cv2.line(frame, self.roi_temp_points[i-1], point, (0, 255, 255), 3)
            
            # Draw closing line if we have 3+ points
            if len(self.roi_temp_points) >= 3:
                # Draw dashed line by drawing short segments
                p1 = self.roi_temp_points[-1]
                p2 = self.roi_temp_points[0]
                # Simple dashed line effect
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
            
            # Draw text
            text = f"ROI: {len(self.roi_temp_points)} points (Right-click to finish)"
            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale and display
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.FastTransformation  # Use Fast for better performance
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update FPS counter
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            
            # Add to history (ignore very low FPS from lag/model switch)
            if self.current_fps >= 5.0:
                self.fps_history.append(self.current_fps)
                if len(self.fps_history) > self.fps_history_max:
                    self.fps_history.pop(0)
            
            # AUTO CLEANUP if FPS drops significantly
            if len(self.fps_history) >= 5:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                if self.current_fps < avg_fps * 0.7:  # FPS dropped 30%
                    print(f"⚠️ FPS drop detected ({self.current_fps:.1f} < {avg_fps:.1f}), auto-cleaning...")
                    QTimer.singleShot(0, self.manual_cleanup)
            
            # Calculate average FPS
            if len(self.fps_history) > 0:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.fps_label.setText(f"FPS: {self.current_fps:.1f} (avg: {avg_fps:.1f})")
            else:
                self.fps_label.setText(f"FPS: {self.current_fps:.1f}")
            
            # Color based on performance
            if self.current_fps >= 20:
                color = '#4ade80'
            elif self.current_fps >= 10:
                color = '#fbbf24'
            else:
                color = '#ef4444'
            self.fps_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
            # Update stats only once per second
            if stats['total_objects'] > 0:
                stats_text = f"Total Objects: {stats['total_objects']}\n\n"
                stats_text += "Detections:\n"
                for cls_name, count in stats['class_counts'].items():
                    stats_text += f"  {cls_name}: {count}\n"
                self.stats_text.setText(stats_text)
        
    def closeEvent(self, event):
        """Handle window close"""
        # Save settings before closing
        self.save_settings()
        
        if self.video_thread and self.video_thread.isRunning():
            self.stop_processing()
        event.accept()
    
    def save_settings(self):
        """Save all settings to file"""
        settings = {
            # Model settings
            'model_choice': self.model_combo.currentText(),
            'best_model_choice': self.best_model_combo.currentText(),
            'tracker_choice': self.tracker_combo.currentText(),
            
            # Detection settings
            'confidence': self.confidence_slider.value(),
            
            # Performance settings
            'frame_skip': self.frame_skip_slider.value(),
            'resize_scale': self.resize_combo.currentText(),
            'smooth_mode': self.smooth_mode,
            
            # Stream settings
            'stream_quality': self.stream_quality_combo.currentText(),
            'video_type': self.video_type_combo.currentText(),
            
            # ROI settings
            'roi_threshold': self.roi_threshold_slider.value(),
            'roi_visible': self.roi_visible,
        }
        
        try:
            import json
            with open('app_settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            print("✅ Settings saved")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load all settings from file"""
        try:
            import json
            from pathlib import Path
            
            settings_file = Path('app_settings.json')
            if not settings_file.exists():
                print("No saved settings found, using defaults")
                return
            
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Model settings
            if 'model_choice' in settings:
                index = self.model_combo.findText(settings['model_choice'])
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
            
            if 'best_model_choice' in settings:
                index = self.best_model_combo.findText(settings['best_model_choice'])
                if index >= 0:
                    self.best_model_combo.setCurrentIndex(index)
            
            if 'tracker_choice' in settings:
                index = self.tracker_combo.findText(settings['tracker_choice'])
                if index >= 0:
                    self.tracker_combo.setCurrentIndex(index)
            
            # Detection settings
            if 'confidence' in settings:
                self.confidence_slider.setValue(settings['confidence'])
            
            # Performance settings
            if 'frame_skip' in settings:
                self.frame_skip_slider.setValue(settings['frame_skip'])
            
            if 'resize_scale' in settings:
                index = self.resize_combo.findText(settings['resize_scale'])
                if index >= 0:
                    self.resize_combo.setCurrentIndex(index)
            
            if 'smooth_mode' in settings:
                self.smooth_mode = settings['smooth_mode']
            
            # Stream settings
            if 'stream_quality' in settings:
                index = self.stream_quality_combo.findText(settings['stream_quality'])
                if index >= 0:
                    self.stream_quality_combo.setCurrentIndex(index)
            
            if 'video_type' in settings:
                index = self.video_type_combo.findText(settings['video_type'])
                if index >= 0:
                    self.video_type_combo.setCurrentIndex(index)
            
            # ROI settings
            if 'roi_threshold' in settings:
                self.roi_threshold_slider.setValue(settings['roi_threshold'])
            
            if 'roi_visible' in settings:
                self.roi_visible = settings['roi_visible']
                self.roi_manager.visible = self.roi_visible
                # Update button state
                if not self.roi_visible:
                    self.btn_toggle_roi.setChecked(True)
                    self.btn_toggle_roi.setText("👁️ Show")
            
            print("✅ Settings loaded")
            
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def load_stream_history(self):
        """Load stream URL history from file"""
        history_file = Path("stream_history.txt")
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f.readlines() if line.strip()]
                    # Add to combobox (most recent first)
                    for url in urls[:5]:  # Only keep 5 most recent
                        self.stream_input.addItem(url)
            except Exception as e:
                print(f"Error loading stream history: {e}")
    
    def save_stream_to_history(self, url: str):
        """Save stream URL to history"""
        if not url or url.strip() == "":
            return
        
        # Remove if already exists (to move to top)
        index = self.stream_input.findText(url)
        if index >= 0:
            self.stream_input.removeItem(index)
        
        # Add to top
        self.stream_input.insertItem(0, url)
        self.stream_input.setCurrentIndex(0)
        
        # Keep only 5 items
        while self.stream_input.count() > 5:
            self.stream_input.removeItem(self.stream_input.count() - 1)
        
        # Save to file
        try:
            history_file = Path("stream_history.txt")
            urls = [self.stream_input.itemText(i) for i in range(self.stream_input.count())]
            with open(history_file, 'w', encoding='utf-8') as f:
                for u in urls:
                    f.write(u + '\n')
        except Exception as e:
            print(f"Error saving stream history: {e}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

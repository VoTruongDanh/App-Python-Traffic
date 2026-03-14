"""
PyQt5 Desktop App - Real-time Object Tracking
High-performance alternative to Streamlit for local use
"""
import config
from model_loader import load_yolo_models, initialize_tracker

import os
import shutil
import subprocess
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QFileDialog, QLineEdit, QGroupBox, QGridLayout, QTextEdit,
                             QDialog, QProgressBar, QListWidget, QListWidgetItem, QToolButton,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from pathlib import Path

from video_processor import VideoProcessor
# from video_processor_optimized import VideoProcessorOptimized as VideoProcessor
from roi_manager import ROIManager

# Global state for model selection (replacement for st.session_state)
class AppState:
    model_choice = "YOLOv26n (Fastest)"
    tracker_choice = "SORT (Fast)"
    best_model_choice = "None (Use base YOLO only)"

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
        self._last_stats = {'total_objects': 0, 'class_counts': {}}
        
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

                # Apply frame-skip before expensive processing
                if self.frame_skip > 0:
                    skip_counter = (skip_counter + 1) % (self.frame_skip + 1)
                    if skip_counter != 0:
                        if self.smooth_mode and last_processed_frame is not None:
                            self.frame_ready.emit(last_processed_frame, self._last_stats)
                        else:
                            self.frame_ready.emit(frame, self._last_stats)
                        continue
                
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
                    self._last_stats = stats
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
        self.last_display_time = 0.0
        self.display_target_fps = 24
        self.display_frame_interval = 1.0 / self.display_target_fps
        self.last_auto_cleanup_time = 0.0
        self.auto_cleanup_cooldown = 20.0
        self.last_runtime_refresh_time = 0.0
        self.runtime_refresh_interval = 2.0
        self.last_stats_text = ""
        self.last_fps_color = ""
        self.video_source = None

        # Debounced restart state for heavy changes while stream is running.
        self.pending_restart = False
        self.pending_reload_models = False
        self.restart_message = ""
        self.restart_timer = QTimer(self)
        self.restart_timer.setSingleShot(True)
        self.restart_timer.timeout.connect(self._apply_pending_restart)
        
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
        central_widget.setObjectName("rootSurface")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(14)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        title_row = QHBoxLayout()
        self.video_title_label = QLabel("Live View")
        self.video_title_label.setObjectName("videoTitle")
        title_row.addWidget(self.video_title_label)
        title_row.addStretch()

        self.btn_toggle_sidebar = QPushButton("Hide Controls")
        self.btn_toggle_sidebar.setObjectName("secondaryAction")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        title_row.addWidget(self.btn_toggle_sidebar)
        left_panel.addLayout(title_row)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setObjectName("videoSurface")
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Load video or livestream to begin")
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.on_video_label_click
        left_panel.addWidget(self.video_label)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("fpsBadge")
        self.fps_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        left_panel.addWidget(self.fps_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_load_video = QPushButton("📁 Load Video")
        self.btn_load_video.setObjectName("secondaryAction")
        self.btn_load_video.clicked.connect(self.load_video_file)
        button_layout.addWidget(self.btn_load_video)
        
        self.btn_start = QPushButton("▶️ Start")
        self.btn_start.setObjectName("primaryAction")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        button_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setObjectName("dangerAction")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_stop)
        
        left_panel.addLayout(button_layout)
        
        # Right panel - Controls and stats
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        runtime_shell = QFrame()
        runtime_shell.setObjectName("panelShell")
        runtime_shell_layout = QVBoxLayout(runtime_shell)
        runtime_shell_layout.setContentsMargins(0, 0, 0, 0)
        runtime_shell_layout.setSpacing(8)

        runtime_header = QHBoxLayout()
        self.runtime_toggle_button = QToolButton()
        self.runtime_toggle_button.setText("Runtime Overview")
        self.runtime_toggle_button.setCheckable(True)
        self.runtime_toggle_button.setChecked(True)
        self.runtime_toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.runtime_toggle_button.setArrowType(Qt.DownArrow)
        self.runtime_toggle_button.clicked.connect(self.toggle_runtime_section)
        runtime_header.addWidget(self.runtime_toggle_button)
        runtime_header.addStretch()
        runtime_shell_layout.addLayout(runtime_header)

        self.runtime_panel = QFrame()
        self.runtime_panel.setObjectName("runtimeCard")
        runtime_layout = QVBoxLayout(self.runtime_panel)
        runtime_badge_grid = QGridLayout()
        runtime_badge_grid.setHorizontalSpacing(8)
        runtime_badge_grid.setVerticalSpacing(8)

        self.backend_badge = QLabel("Backend: Detecting")
        self.backend_badge.setObjectName("metricChip")
        runtime_badge_grid.addWidget(self.backend_badge, 0, 0)

        self.mode_badge = QLabel("Mode: Standard")
        self.mode_badge.setObjectName("metricChip")
        runtime_badge_grid.addWidget(self.mode_badge, 0, 1)

        self.tracker_badge = QLabel("Tracker: Simple")
        self.tracker_badge.setObjectName("metricChip")
        runtime_badge_grid.addWidget(self.tracker_badge, 1, 0)

        self.source_badge = QLabel("Source: Waiting")
        self.source_badge.setObjectName("metricChip")
        runtime_badge_grid.addWidget(self.source_badge, 1, 1)
        runtime_layout.addLayout(runtime_badge_grid)

        self.runtime_hint_label = QLabel("Balanced preset and 75% resize are the best starting point for stable live FPS.")
        self.runtime_hint_label.setObjectName("subtleInfo")
        self.runtime_hint_label.setWordWrap(True)
        runtime_layout.addWidget(self.runtime_hint_label)

        self.runtime_summary_label = QLabel("Waiting for models...")
        self.runtime_summary_label.setObjectName("summaryCard")
        self.runtime_summary_label.setWordWrap(True)
        runtime_layout.addWidget(self.runtime_summary_label)

        runtime_shell_layout.addWidget(self.runtime_panel)
        right_panel.addWidget(runtime_shell)
        
        # Model selection
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        self.custom_model_path = None
        
        model_layout.addWidget(QLabel("Detection Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "YOLOv26n (GPU Recommended)",
            "YOLOv8n",
            "YOLOv11n",
            "YOLOv11s (Fast)",
            "Train1 (Person model)",
            "Train2 (Multi-class)",
            "Custom model..."
        ])
        self.model_combo.setCurrentIndex(0)
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
        self.model_combo.setToolTip(
            "YOLOv26n: fastest GPU-friendly base model\n"
            "YOLOv8n / YOLOv11*: pretrained base models\n"
            "Train1: custom person model + YOLOv26n for vehicles\n"
            "Train2: one custom multi-class model\n"
            "Custom: use Browse to choose your own .pt or .onnx model"
        )
        model_layout.addWidget(self.model_combo)
        self.last_model_selection = self.model_combo.currentText()

        custom_row = QHBoxLayout()
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setReadOnly(True)
        self.custom_model_input.setPlaceholderText("No custom model selected")
        custom_row.addWidget(self.custom_model_input)

        self.btn_browse_custom_model = QPushButton("Browse...")
        self.btn_browse_custom_model.setObjectName("secondaryAction")
        self.btn_browse_custom_model.clicked.connect(self.browse_custom_model)
        custom_row.addWidget(self.btn_browse_custom_model)
        model_layout.addLayout(custom_row)
        
        model_layout.addWidget(QLabel("Tracking Mode:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["SORT (Fast)", "DeepSORT (Stable)"])
        self.tracker_combo.setCurrentIndex(0)
        self.tracker_combo.setToolTip(
            "SORT: faster tracking and lighter CPU/GPU load\n"
            "DeepSORT: more stable IDs, slower than SORT"
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
        self.btn_display_mode.setStyleSheet("background-color: #0f766e; color: white;")
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
        self.btn_trail_toggle.setStyleSheet("background-color: #b42318; color: white;")
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
        self.frame_skip_slider.valueChanged.connect(self.on_frame_skip_changed)
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
        self.resize_combo.setCurrentIndex(1)  # Default: 75% balanced
        self.resize_combo.setToolTip(
            "Resize inference resolution for speed:\n"
            "100%: Full 1920x1080 (Best quality, slower)\n"
            "75%: 1440x810 (Balanced - Recommended)\n"
            "50%: 960x540 (2-3x faster)\n"
            "25%: 480x270 (4-5x faster, lower quality)"
        )
        self.resize_combo.currentTextChanged.connect(self.on_resize_changed)
        self.resize_combo.setEnabled(True)  # ENABLED
        perf_layout.addWidget(self.resize_combo)
        
        perf_layout.addWidget(QLabel("Pipeline Mode:"))
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItem("Standard - full analytics", "standard")
        self.processing_mode_combo.addItem("Threaded - safer parallelism", "threaded")
        self.processing_mode_combo.addItem("Optimized - lighter overlays", "optimized")
        self.processing_mode_combo.addItem("Ultra - max FPS live mode", "ultra")
        self.processing_mode_combo.setToolTip(
            "Standard: richest labels and analytics\n"
            "Threaded: better overlap of work\n"
            "Optimized: lighter rendering for smoother playback\n"
            "Ultra: highest FPS / lowest latency"
        )
        self.processing_mode_combo.currentIndexChanged.connect(self.on_processing_mode_changed)
        perf_layout.addWidget(self.processing_mode_combo)

        self.processing_mode_hint = QLabel("Ultra is recommended for livestreams. Standard is best when you need richer labels.")
        self.processing_mode_hint.setObjectName("subtleInfo")
        self.processing_mode_hint.setWordWrap(True)
        perf_layout.addWidget(self.processing_mode_hint)
        
        # Manual cleanup button
        btn_cleanup = QPushButton("🧹 Clear Cache Now")
        btn_cleanup.setObjectName("warnButton")
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
        btn_balanced.setToolTip("Skip:1 + Resize:75%\nBest balance for most cameras")
        btn_balanced.clicked.connect(lambda: self.apply_preset(1, 75))
        preset_layout.addWidget(btn_balanced)
        
        btn_speed = QPushButton("Speed")
        btn_speed.setToolTip("Skip:2 + Resize:50%\nHigh FPS with acceptable detail")
        btn_speed.clicked.connect(lambda: self.apply_preset(2, 50))
        btn_speed.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold;")
        preset_layout.addWidget(btn_speed)
        
        perf_layout.addLayout(preset_layout)
        
        # Add smooth mode toggle
        self.smooth_mode = True  # Default: ON for smoother output
        self.btn_smooth_toggle = QPushButton("Smooth Mode: ON")
        self.btn_smooth_toggle.setCheckable(True)
        self.btn_smooth_toggle.setChecked(True)
        self.btn_smooth_toggle.setStyleSheet("background-color: #15803d; color: white;")
        self.btn_smooth_toggle.setToolTip(
            "ON: Display last frame when skipping (smoother but less accurate)\n"
            "OFF: Only show processed frames (accurate but may stutter)"
        )
        self.btn_smooth_toggle.clicked.connect(
            lambda checked: self.toggle_smooth_mode(checked, self.btn_smooth_toggle)
        )
        perf_layout.addWidget(self.btn_smooth_toggle)
        
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
        self.btn_start_stream.setObjectName("accentAction")
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
        self.roi_status_label.setObjectName("subtleInfo")
        roi_layout.addWidget(self.roi_status_label)
        
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        # Current model info
        self.model_info_label = QLabel("Model: Loading...")
        self.model_info_label.setObjectName("modelChip")
        self.model_info_label.setWordWrap(True)
        stats_layout.addWidget(self.model_info_label)
        
        self.stats_text = QTextEdit()
        self.stats_text.setObjectName("statsPanel")
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(140)
        self.stats_text.setPlainText(
            "No session data yet.\n\n"
            "Recommended workflow:\n"
            "- Start with 75% resize\n"
            "- Use Ultra mode for live streams\n"
            "- Keep tracker on Simple or SORT when tuning FPS"
        )
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLine")
        right_panel.addWidget(self.status_label)
        
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 3)
        
        # Wrap right panel in scroll area
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(right_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setMinimumWidth(350)
        self.scroll_area.setMaximumWidth(420)
        
        main_layout.addWidget(self.scroll_area, 1)
        self.apply_theme()
        self._refresh_custom_model_input()
        self._sync_processing_mode_buttons()

    def apply_theme(self):
        """Apply cohesive UI theme for better readability and hierarchy."""
        self.setStyleSheet("""
            QWidget {
                color: #1f2933;
                font-family: "Segoe UI", "Trebuchet MS", sans-serif;
                font-size: 12px;
            }
            QWidget#rootSurface {
                background: #ece6da;
            }
            QGroupBox {
                background: #fffdf8;
                border: 1px solid #d6cab8;
                border-radius: 16px;
                margin-top: 10px;
                padding: 10px 12px 12px 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #483628;
                background: #ece6da;
            }
            QLabel#videoTitle {
                color: #1b2733;
                font-size: 18px;
                font-weight: 700;
                padding: 2px 2px 2px 2px;
                letter-spacing: 0.4px;
            }
            QFrame#panelShell {
                background: transparent;
                border: none;
            }
            QFrame#runtimeCard {
                background: #fffdf8;
                border: 1px solid #d6cab8;
                border-radius: 16px;
                padding: 8px 10px;
            }
            QLabel#videoSurface {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #081018, stop:1 #152533);
                border: 1px solid #0f1b27;
                border-radius: 18px;
                color: #d8e5ef;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#fpsBadge {
                color: #1f2933;
                background: #fff7ed;
                border: 1px solid #f0b47a;
                border-radius: 10px;
                padding: 7px 12px;
            }
            QLabel#metricChip {
                background: #f5efe3;
                color: #334155;
                border: 1px solid #d8ccb7;
                border-radius: 12px;
                padding: 7px 12px;
                font-weight: 700;
            }
            QLabel#statusLine {
                color: #3a2a1d;
                background: #fff7ed;
                border: 1px solid #f3c892;
                border-radius: 12px;
                padding: 10px 12px;
                font-weight: 600;
            }
            QLabel#subtleInfo {
                color: #6a5543;
                padding: 4px 6px;
                background: transparent;
            }
            QToolButton {
                background: #fff7ed;
                color: #3a2a1d;
                border: 1px solid #e0c39a;
                border-radius: 12px;
                padding: 8px 10px;
                font-weight: 700;
                text-align: left;
            }
            QToolButton:hover {
                background: #ffedd5;
            }
            QLabel#modelChip {
                background: #1b2733;
                color: #f8d4a4;
                padding: 10px 12px;
                border-radius: 12px;
                font-weight: 700;
                font-size: 11px;
            }
            QLabel#summaryCard {
                background: #f4ede1;
                color: #433225;
                border: 1px solid #e4d7c1;
                border-radius: 12px;
                padding: 10px 12px;
                font-weight: 600;
                line-height: 1.4;
            }
            QTextEdit#statsPanel {
                background: #111c27;
                color: #f8fafc;
                border: 1px solid #293847;
                border-radius: 14px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
                padding: 10px;
            }
            QPushButton {
                background: #e8ddd0;
                color: #2a211a;
                border: 1px solid #d7cab9;
                border-radius: 12px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #ddcfbf;
            }
            QPushButton:pressed {
                background: #cdb9a4;
            }
            QPushButton:disabled {
                background: #efe7dd;
                color: #9a948e;
                border-color: #e5ddd4;
            }
            QPushButton#primaryAction {
                background: #0f766e;
                color: white;
                border-color: #115e59;
            }
            QPushButton#primaryAction:hover {
                background: #0d9488;
            }
            QPushButton#accentAction {
                background: #c96f16;
                color: white;
                border-color: #a65b12;
            }
            QPushButton#accentAction:hover {
                background: #dd7f1d;
            }
            QPushButton#dangerAction {
                background: #b42318;
                color: white;
                border-color: #912018;
            }
            QPushButton#dangerAction:hover {
                background: #d92d20;
            }
            QPushButton#warnButton {
                background: #f59e0b;
                color: white;
                border-color: #d97706;
            }
            QComboBox, QLineEdit, QTextEdit {
                background: #fffdf8;
                border: 1px solid #d7cab9;
                border-radius: 10px;
                padding: 7px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #fffdf8;
                selection-background-color: #d97706;
                selection-color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #dfd2c2;
                height: 6px;
                background: #efe3d4;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #c96f16;
                border: 1px solid #a65b12;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def _get_processing_mode(self) -> str:
        """Return currently selected processing mode."""
        if self.use_ultra_mode:
            return "ultra"
        if self.use_optimized_mode:
            return "optimized"
        if self.use_threaded_mode:
            return "threaded"
        return "standard"

    def _set_processing_mode_flags(self, mode: str):
        """Set mutually-exclusive processing mode flags."""
        mode = (mode or "standard").strip().lower()
        self.use_threaded_mode = mode == "threaded"
        self.use_optimized_mode = mode == "optimized"
        self.use_ultra_mode = mode == "ultra"

    def _sync_processing_mode_buttons(self):
        """Sync processing mode selector, badges, and helper text."""
        mode = self._get_processing_mode()
        descriptions = {
            "standard": "Standard keeps the richest labels and analytics.",
            "threaded": "Threaded overlaps work with safer parallelism.",
            "optimized": "Optimized trims overlay cost for steadier playback.",
            "ultra": "Ultra prioritizes lowest latency and highest live FPS.",
        }

        if hasattr(self, 'processing_mode_combo'):
            index = self.processing_mode_combo.findData(mode)
            if index >= 0:
                self.processing_mode_combo.blockSignals(True)
                self.processing_mode_combo.setCurrentIndex(index)
                self.processing_mode_combo.blockSignals(False)

        if hasattr(self, 'processing_mode_hint'):
            self.processing_mode_hint.setText(descriptions.get(mode, descriptions["standard"]))

        if hasattr(self, 'mode_badge'):
            self.mode_badge.setText(f"Mode: {mode.title()}")

    def _apply_processing_mode(self, mode: str, trigger_reload: bool = True):
        """
        Apply processing mode with strict mutual exclusivity.
        """
        self._set_processing_mode_flags(mode)
        self._sync_processing_mode_buttons()

        current_mode = self._get_processing_mode()
        if current_mode == "standard":
            self.status_label.setText("Standard mode enabled")
        else:
            self.status_label.setText(f"{current_mode.title()} mode enabled (reloading...)")

        if trigger_reload:
            if self.video_thread and self.video_thread.isRunning():
                self.pending_reload_models = True
                self.auto_reload_if_running(reason=f"Switching to {current_mode.title()} mode...")
            else:
                self.load_models()

    def _get_tracker_display_name(self, tracker) -> str:
        """Map tracker instance to UI label."""
        if tracker is None:
            return self.tracker_combo.currentText()

        name = tracker.__class__.__name__.lower()
        if "simple" in name:
            return "Simple (Ultra Fast)"
        if "sort" in name and "deepsort" not in name:
            return "SORT (Fast)"
        return "DeepSORT (Stable)"

    def _sync_tracker_combo_with_runtime(self):
        """Keep tracker combo aligned with actual tracker in use."""
        runtime_name = self._get_tracker_display_name(self.tracker)
        if self.tracker_combo.currentText() == runtime_name:
            return
        index = self.tracker_combo.findText(runtime_name)
        if index >= 0:
            self.tracker_combo.blockSignals(True)
            self.tracker_combo.setCurrentIndex(index)
            self.tracker_combo.blockSignals(False)
        app_state.tracker_choice = runtime_name

    def _resolve_model_selection(self, selection: str = None, prompt_for_custom: bool = False) -> bool:
        """Map the single model selector to the runtime model configuration."""
        selection = selection or self.model_combo.currentText()
        profile_map = {
            "YOLOv26n (GPU Recommended)": ("YOLOv26n (Fastest)", "None (Use base YOLO only)"),
            "YOLOv8n": ("YOLOv8n", "None (Use base YOLO only)"),
            "YOLOv11n": ("YOLOv11n", "None (Use base YOLO only)"),
            "YOLOv11s (Fast)": ("YOLOv11s (Fast)", "None (Use base YOLO only)"),
            "Train1 (Person model)": ("YOLOv26n (Fastest)", "Train1 (Person only) - ../best.pt"),
            "Train2 (Multi-class)": ("YOLOv26n (Fastest)", "Train2 (Multi-class) - ../Train2/best.pt"),
        }

        if selection == "Custom model...":
            if prompt_for_custom:
                return self.browse_custom_model()

            if not self.custom_model_path:
                return False

            app_state.model_choice = "YOLOv26n (Fastest)"
            app_state.best_model_choice = f"Custom: {self.custom_model_path}"
            self._refresh_custom_model_input()
            return True

        self.last_model_selection = selection
        base_model, best_model = profile_map.get(
            selection,
            ("YOLOv26n (Fastest)", "None (Use base YOLO only)")
        )
        self.custom_model_path = None
        app_state.model_choice = base_model
        app_state.best_model_choice = best_model
        self._refresh_custom_model_input()
        return True
        
    def load_models(self):
        """Load YOLO models and tracker"""
        self.status_label.setText("Loading models...")
        self._sync_processing_mode_buttons()
        QApplication.processEvents()
        
        try:
            # Set model choice in app state
            app_state.tracker_choice = self.tracker_combo.currentText()
            if not self._resolve_model_selection(self.model_combo.currentText(), prompt_for_custom=False):
                self.status_label.setText("Select a valid model first")
                return
            lowered_confidence = self._apply_recommended_confidence_for_model()
            
            self.model_person, self.model_vehicle = load_yolo_models()
            self.tracker = initialize_tracker(self.tracker_combo.currentText())
            self._sync_tracker_combo_with_runtime()
            app_state.tracker_choice = self._get_tracker_display_name(self.tracker)
            actual_backends = []
            for model in (self.model_person, self.model_vehicle):
                backend_name = self._identify_model_backend(model)
                if backend_name not in actual_backends:
                    actual_backends.append(backend_name)
            gpu_backend_active = any("CUDA" in backend for backend in actual_backends)
            cpu_only_mode = not gpu_backend_active
            selected_mode = self._get_processing_mode()
            
            # Create processor (ultra, optimized, threaded, or standard)
            if selected_mode == "ultra":
                from video_processor_ultra import UltraVideoProcessor
                self.processor = UltraVideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("Using Ultra Processor")
            elif selected_mode == "optimized":
                from video_processor_optimized import VideoProcessorOptimized
                self.processor = VideoProcessorOptimized(self.model_person, self.model_vehicle, self.tracker)
                print("Using Optimized Processor")
            elif selected_mode == "threaded":
                from video_processor_threaded import ThreadedVideoProcessor
                self.processor = ThreadedVideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("Using Threaded Processor")
            elif cpu_only_mode:
                from video_processor_ultra import UltraVideoProcessor
                self.processor = UltraVideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("Using Ultra Processor (CPU fallback)")
            else:
                self.processor = VideoProcessor(self.model_person, self.model_vehicle, self.tracker)
                print("Using Standard Processor")
                
            self.processor.set_confidence(self.confidence_slider.value() / 100.0)
            
            # Connect ROI manager to processor
            self.processor.set_roi_manager(self.roi_manager)
            
            # Update model info display
            self.update_model_info()
            
            backend_text = " + ".join(actual_backends) if actual_backends else "CPU"
            display_model_name = self._get_selected_model_display_name()

            if cpu_only_mode and all("CPU" in backend for backend in actual_backends):
                self.status_label.setText(f"Models loaded: {display_model_name} | Backend: CPU")
            else:
                self.status_label.setText(f"Models loaded: {display_model_name} | Backend: {backend_text}")
            if lowered_confidence is not None:
                self.status_label.setText(
                    f"Models loaded: {display_model_name} | Backend: {backend_text} | Conf auto -> 0.{lowered_confidence:02d}"
                )
            self.btn_load_video.setEnabled(True)
            self.btn_start_stream.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error loading models: {e}")

    def _has_torch_cuda(self) -> bool:
        """Check whether torch has CUDA support in current runtime."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _has_onnx_cuda_provider(self) -> bool:
        """Check whether ONNX Runtime CUDA provider is available."""
        try:
            import onnxruntime
            return 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        except Exception:
            return False

    def _describe_source(self) -> str:
        """Return a compact description of the active source."""
        if not self.video_source:
            return "Waiting"

        if isinstance(self.video_source, int):
            return f"Camera {self.video_source}"

        source_text = str(self.video_source).strip()
        if source_text.isdigit():
            return f"Camera {source_text}"
        if source_text.startswith(("rtsp://", "http://", "https://")):
            return "Live URL"
        return Path(source_text).name

    def toggle_sidebar(self):
        """Show or hide the right-side control panel."""
        visible = not self.scroll_area.isVisible()
        self.scroll_area.setVisible(visible)
        self.btn_toggle_sidebar.setText("Hide Controls" if visible else "Show Controls")

    def toggle_runtime_section(self):
        """Collapse or expand the runtime overview card."""
        expanded = self.runtime_toggle_button.isChecked()
        self.runtime_panel.setVisible(expanded)
        self.runtime_toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _open_file_dialog(self, title: str, file_filter: str):
        """Open a modal file dialog that stays above the app window."""
        dialog = QFileDialog(self, title, "", file_filter)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec_() == QDialog.Accepted:
            files = dialog.selectedFiles()
            if files:
                return files[0], dialog.selectedNameFilter()
        return "", ""

    def _refresh_custom_model_input(self):
        """Keep the custom model path field aligned with current selection."""
        if self.custom_model_path:
            path = Path(self.custom_model_path)
            self.custom_model_input.setText(path.name)
            self.custom_model_input.setToolTip(str(path))
            self.btn_browse_custom_model.setText("Change...")
        else:
            self.custom_model_input.clear()
            self.custom_model_input.setPlaceholderText("No custom model selected")
            self.custom_model_input.setToolTip("")
            self.btn_browse_custom_model.setText("Browse...")

    def browse_custom_model(self):
        """Select a custom detector file from disk."""
        previous_selection = getattr(self, 'last_model_selection', "YOLOv26n (GPU Recommended)")
        file_path, _ = self._open_file_dialog(
            "Select Custom Detection Model",
            "Model Files (*.pt *.onnx);;PyTorch Models (*.pt);;ONNX Models (*.onnx);;All Files (*.*)"
        )
        if not file_path:
            if not self.custom_model_path:
                index = self.model_combo.findText(previous_selection)
                if index >= 0:
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(index)
                    self.model_combo.blockSignals(False)
            self.status_label.setText("Custom model selection canceled")
            self._refresh_custom_model_input()
            return False

        self.custom_model_path = file_path
        custom_index = self.model_combo.findText("Custom model...")
        if custom_index >= 0 and self.model_combo.currentIndex() != custom_index:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(custom_index)
            self.model_combo.blockSignals(False)

        self._refresh_custom_model_input()
        if not self._resolve_model_selection("Custom model...", prompt_for_custom=False):
            self.status_label.setText("Unable to apply custom model")
            return False

        if self.video_thread and self.video_thread.isRunning():
            self.pending_reload_models = True
            self.auto_reload_if_running(reason=f"Switching model to {Path(file_path).name}...")
        else:
            self.load_models()
        return True

    def _get_selected_model_display_name(self) -> str:
        """Return the label the user expects to see in the UI."""
        selection = self.model_combo.currentText()
        if selection == "Custom model..." and self.custom_model_path:
            return f"Custom ({Path(self.custom_model_path).name})"
        return selection

    def _is_trained_model_selection(self) -> bool:
        """Return True for custom or trained detector profiles."""
        selection = self.model_combo.currentText()
        return selection in {
            "Train1 (Person model)",
            "Train2 (Multi-class)",
            "Custom model...",
        }

    def _apply_recommended_confidence_for_model(self):
        """
        Trained models in this project tend to emit lower confidence scores than
        the stock YOLO profiles. Lower the threshold automatically so the user
        does not end up with a blank frame after switching models.
        """
        if not self._is_trained_model_selection():
            return None

        recommended = 25
        current = self.confidence_slider.value()
        if current <= recommended:
            return None

        self.confidence_slider.setValue(recommended)
        return recommended

    def _get_loaded_model_name(self, model) -> str:
        """Return the actual loaded model filename when available."""
        loaded_name = getattr(model, '_loaded_model_name', None)
        if loaded_name:
            return loaded_name
        return self._get_selected_model_display_name()

    def _identify_model_backend(self, model) -> str:
        """Identify the backend used by a model instance."""
        if model is None:
            return "Not loaded"

        try:
            from onnx_model import ONNXModel
            if isinstance(model, ONNXModel):
                providers = model.session.get_providers()
                if providers and providers[0] == 'CUDAExecutionProvider':
                    return "ONNX CUDA"
                return "ONNX CPU"
        except Exception:
            pass

        try:
            if hasattr(model, 'model'):
                device = str(next(model.model.parameters()).device)
                return "Torch CUDA" if device.startswith("cuda") else "Torch CPU"
        except Exception:
            pass

        return "Unknown"

    def _refresh_runtime_overview(self):
        """Refresh runtime badges and summary card."""
        person_backend = self._identify_model_backend(self.model_person)
        vehicle_backend = self._identify_model_backend(self.model_vehicle)
        tracker_text = self._get_tracker_display_name(self.tracker)
        mode_text = self._get_processing_mode().title()
        source_text = self._describe_source()
        profile_text = self._get_selected_model_display_name()
        person_model_name = self._get_loaded_model_name(self.model_person)
        vehicle_model_name = self._get_loaded_model_name(self.model_vehicle)
        shared_detector = self.model_person is not None and self.model_person is self.model_vehicle

        backends = []
        for backend in (person_backend, vehicle_backend):
            if backend not in backends:
                backends.append(backend)
        backend_text = " + ".join(backends) if backends else "Detecting"

        if hasattr(self, 'backend_badge'):
            self.backend_badge.setText(f"Backend: {backend_text}")
        if hasattr(self, 'tracker_badge'):
            self.tracker_badge.setText(f"Tracker: {tracker_text}")
        if hasattr(self, 'source_badge'):
            self.source_badge.setText(f"Source: {source_text}")
        if hasattr(self, 'mode_badge'):
            self.mode_badge.setText(f"Mode: {mode_text}")

        if self._is_trained_model_selection():
            hint = "Custom/train models in this project usually work best around confidence 0.20-0.35. If the frame looks empty, lower the threshold first."
        elif "CPU" in backend_text and self._has_torch_cuda():
            hint = "GPU torch is ready, but at least one model still runs on CPU. Custom .pt models benefit most from CUDA or an ONNX export."
        elif "CPU" in backend_text and self._has_onnx_cuda_provider():
            hint = "ONNX CUDA is available. Models with sibling .onnx files will run much faster than raw .pt on CPU."
        elif "CUDA" in backend_text:
            hint = "GPU acceleration is active. Use Ultra mode + 75% resize for the best livestream responsiveness."
        else:
            hint = "Load models and source to inspect the active pipeline."

        if hasattr(self, 'runtime_hint_label'):
            self.runtime_hint_label.setText(hint)

        if hasattr(self, 'runtime_summary_label'):
            self.runtime_summary_label.setText(
                f"Profile: {profile_text}\n"
                f"Detector: {person_model_name}{' (shared)' if shared_detector else ''}\n"
                f"Person model: {person_backend}\n"
                f"Vehicle model: {vehicle_backend} [{vehicle_model_name}]\n"
                f"Tracker: {tracker_text}\n"
                f"Source: {source_text}"
            )

    def update_model_info(self):
        """Update the model info display."""
        try:
            using_onnx = False
            onnx_available = False
            onnx_cuda = False
            try:
                from onnx_model import ONNXModel
                import onnxruntime
                onnx_available = True
                using_onnx = isinstance(self.model_person, ONNXModel) or isinstance(self.model_vehicle, ONNXModel)
                onnx_cuda = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
            except Exception:
                pass

            selected_profile = self._get_selected_model_display_name()
            tracker_type = self._get_tracker_display_name(self.tracker)
            mode = self._get_processing_mode().title()
            actual_person_backend = self._identify_model_backend(self.model_person)
            actual_vehicle_backend = self._identify_model_backend(self.model_vehicle)

            info_lines = [f"Model: {selected_profile}"]
            info_lines.append(f"Person Backend: {actual_person_backend} [{self._get_loaded_model_name(self.model_person)}]")
            info_lines.append(f"Vehicle Backend: {actual_vehicle_backend} [{self._get_loaded_model_name(self.model_vehicle)}]")

            info_lines.append(f"Tracker: {tracker_type}")
            info_lines.append(f"Mode: {mode}")
            self.model_info_label.setText("\n".join(info_lines))
            self._refresh_runtime_overview()

        except Exception:
            self.model_info_label.setText(f"Model: {self.model_combo.currentText()}")
            self._refresh_runtime_overview()

    def on_model_changed(self, model_name):
        """Handle model selection change"""
        try:
            if model_name == "Custom model...":
                QTimer.singleShot(0, self.browse_custom_model)
                return
            if not self._resolve_model_selection(model_name, prompt_for_custom=True):
                self.status_label.setText("Model selection canceled")
                return
            if self.video_thread and self.video_thread.isRunning():
                self.pending_reload_models = True
                self.auto_reload_if_running(reason=f"Switching model to {model_name}...")
            else:
                self.load_models()
        except Exception as e:
            self.status_label.setText(f"Error loading {model_name}: {e}")
    
    def on_best_model_changed(self, best_model_name):
        """Legacy handler kept for compatibility with older code paths."""
        self.status_label.setText(f"Legacy model selector ignored: {best_model_name}")
        return

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
        
        try:
            if self.video_thread and self.video_thread.isRunning():
                self.pending_reload_models = True
                self.auto_reload_if_running(reason="Reloading trained model...")
            else:
                self.load_models()
                if "None" in best_model_name:
                    self.status_label.setText("Using base YOLO only")
                elif "Train2" in best_model_name:
                    self.status_label.setText("Using Train2 multi-class model")
                elif "Train1" in best_model_name:
                    self.status_label.setText("Using Train1 person model")
                elif "Custom" in best_model_name and self.custom_model_path:
                    self.status_label.setText(f"Using custom model: {Path(self.custom_model_path).name}")
        except Exception as e:
            self.status_label.setText(f"Error loading {best_model_name}: {e}")
        
    def on_tracker_changed(self, tracker_name):
        """Handle tracker selection change"""
        app_state.tracker_choice = tracker_name
        try:
            if self.video_thread and self.video_thread.isRunning():
                self.pending_reload_models = True
                self.auto_reload_if_running(reason=f"Switching tracker to {tracker_name}...")
            else:
                self.load_models()
                self.status_label.setText(f"Tracker changed to {self._get_tracker_display_name(self.tracker)}")
        except Exception as e:
            self.status_label.setText(f"Error changing tracker: {e}")

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
            self.btn_display_mode.setStyleSheet("background-color: #0f766e; color: white;")
        else:
            self.btn_display_mode.setText("📦 Display: Bounding Box")
            self.btn_display_mode.setStyleSheet("background-color: #475569; color: white;")
    
    def toggle_trail(self, checked):
        """Toggle trail drawing"""
        import config
        if checked:
            config.TRAIL_LENGTH = 3  # Enable trail
            self.btn_trail_toggle.setText("🟢 Trail: ON")
            self.btn_trail_toggle.setStyleSheet("background-color: #15803d; color: white;")
        else:
            config.TRAIL_LENGTH = 0  # Disable trail
            self.btn_trail_toggle.setText("🔴 Trail: OFF")
            self.btn_trail_toggle.setStyleSheet("background-color: #b42318; color: white;")
    
    def manual_cleanup(self):
        """Manual memory cleanup"""
        self.status_label.setText("Cleaning cache...")
        QApplication.processEvents()
        
        try:
            # Clear processor cache
            if self.processor:
                trails = getattr(self.processor, 'trails', None)
                if isinstance(trails, dict):
                    trails.clear()
                if hasattr(self.processor, 'tracker') and hasattr(self.processor.tracker, 'tracks'):
                    self.processor.tracker.tracks.clear()
                if hasattr(self.processor, 'frame_counter'):
                    self.processor.frame_counter = 0
            
            # Clear Python garbage
            import gc
            gc.collect()
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("CUDA cache cleared")
            except Exception:
                pass
            
            self.status_label.setText("Cache cleared")
            QTimer.singleShot(
                1500,
                lambda: self.status_label.setText("Processing..." if self.video_thread and self.video_thread.isRunning() else "Ready")
            )
        except Exception as e:
            self.status_label.setText(f"Cleanup error: {e}")

    def toggle_threaded_mode(self, checked):
        """Toggle threaded processing mode."""
        self._apply_processing_mode("threaded" if checked else "standard")

    def on_processing_mode_changed(self, index):
        """Handle processing mode selection from combo box."""
        mode = self.processing_mode_combo.itemData(index)
        if mode:
            self._apply_processing_mode(mode)

    def on_max_det_changed(self, value):
        """Handle max detections change"""
        self.max_det = value
        self.max_det_label.setText(f"{value} objects")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.max_det = value
    
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

    def on_frame_skip_changed(self, value):
        """Apply frame skip live without restarting the stream."""
        self.frame_skip_label.setText(f"Skip: {value} frames")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.frame_skip = value

    def on_resize_changed(self, text):
        """Apply resize scale live when possible."""
        resize_scale = int(text.split()[0].replace('%', ''))
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.resize_scale = resize_scale

    def auto_reload_if_running(self, reason="Applying changes..."):
        """Debounce a heavy restart only when the current source is running."""
        if self.video_thread and self.video_thread.isRunning():
            self.pending_restart = True
            self.restart_message = reason
            self.status_label.setText(reason)
            self.restart_timer.start(180)

    def _apply_pending_restart(self):
        """Commit the queued restart after debounce window."""
        if not self.pending_restart:
            return

        if self.video_thread and self.video_thread.isRunning():
            self.stop_processing(preserve_pending=True)
            return

        reload_models = self.pending_reload_models
        self.pending_restart = False
        self.pending_reload_models = False
        self.restart_message = ""
        if reload_models:
            self.load_models()
        if self.video_source:
            self.start_processing()
    
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
        """Toggle optimized processor mode."""
        self._apply_processing_mode("optimized" if checked else "standard")

    def toggle_ultra_mode(self, checked):
        """Toggle ultra processor mode (async double-buffer pipeline)."""
        self._apply_processing_mode("ultra" if checked else "standard")

    def toggle_smooth_mode(self, checked, button):
        """Toggle smooth mode for frame interpolation"""
        self.smooth_mode = checked
        if checked:
            button.setText("🎬 Smooth Mode: ON")
            button.setStyleSheet("background-color: #15803d; color: white;")
            self.status_label.setText("Smooth mode enabled")
        else:
            button.setText("🎬 Smooth Mode: OFF")
            button.setStyleSheet("")
            self.status_label.setText("Smooth mode disabled")
        
        # Update thread if running
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.smooth_mode = self.smooth_mode
    
    def toggle_roi_drawing(self, checked):
        """Toggle ROI drawing mode"""
        self.roi_drawing_mode = checked
        if checked:
            self.btn_draw_roi.setText("✏️ Drawing... (Click to add points)")
            self.btn_draw_roi.setStyleSheet("background-color: #c96f16; color: white;")
            self.status_label.setText("Click on video to draw ROI polygon. Right-click to finish.")
            self.roi_temp_points = []
        else:
            self.btn_draw_roi.setText("✏️ Draw ROI")
            self.btn_draw_roi.setStyleSheet("")
            if len(self.roi_temp_points) >= 3:
                # Finalize ROI
                self.roi_manager.set_points(self.roi_temp_points)
                self.update_roi_status()
                self.status_label.setText(f"ROI set with {len(self.roi_temp_points)} points")
            else:
                self.status_label.setText("ROI needs at least 3 points")
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

        self.video_label.update()
    
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
        self.video_label.update()
    
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
            self._refresh_runtime_overview()
            
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
                self._refresh_runtime_overview()
                
        self.start_processing()
        
    def start_processing(self):
        """Start video processing"""
        if not self.processor:
            self.status_label.setText("⚠️ Models not loaded")
            return
        if not self.video_source:
            self.status_label.setText("Select a video file, stream URL, or camera first")
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
        self.video_thread.smooth_mode = self.smooth_mode
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.finished.connect(self.on_processing_finished)
        self.video_thread.start()
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_start_stream.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Reset FPS counter
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = []  # Clear FPS history
        self.last_display_time = 0.0
        self._refresh_runtime_overview()
        
    def stop_processing(self, preserve_pending=False):
        """Stop video processing"""
        if not preserve_pending:
            self.pending_restart = False
            self.pending_reload_models = False
            self.restart_timer.stop()
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            
    def on_processing_finished(self):
        """Handle processing finished"""
        if self.pending_restart:
            self.pending_restart = False
            reload_models = self.pending_reload_models
            self.pending_reload_models = False
            self.restart_message = ""
            if reload_models:
                self.load_models()
            self.start_processing()
            return

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load_video.setEnabled(True)
        self.btn_start_stream.setEnabled(True)
        self.status_label.setText("Stopped")
        self._refresh_runtime_overview()
        
    def update_frame(self, frame, stats):
        """Update video display with processed frame - OPTIMIZED"""
        # Render throttle to reduce UI thread overhead
        now = time.perf_counter()
        if (now - self.last_display_time) < self.display_frame_interval:
            return
        self.last_display_time = now

        # Always render on a writable copy so UI overlays behave consistently
        # across Standard / Threaded / Optimized / Ultra processor modes.
        frame = frame.copy()

        # Store original frame size for ROI coordinate mapping
        original_h, original_w = frame.shape[:2]
        self.current_frame_size = (original_w, original_h)

        # Draw finalized ROI overlay in the UI layer so it remains visible even
        # when the active processor mode skips ROI drawing for performance.
        if self.roi_manager and self.roi_manager.is_active():
            self.roi_manager.draw_roi(frame)
        
        # Draw temporary ROI points if in drawing mode (on original frame)
        if self.roi_drawing_mode and len(self.roi_temp_points) > 0:
            for i, point in enumerate(self.roi_temp_points):
                cv2.circle(frame, point, 8, (0, 255, 255), -1)
                cv2.circle(frame, point, 10, (0, 0, 0), 2)
                cv2.putText(frame, str(i + 1), (point[0] + 12, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if i > 0:
                    cv2.line(frame, self.roi_temp_points[i - 1], point, (0, 255, 255), 3)

            if len(self.roi_temp_points) >= 3:
                p1 = self.roi_temp_points[-1]
                p2 = self.roi_temp_points[0]
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

            text_overlay = f"ROI: {len(self.roi_temp_points)} points (Right-click to finish)"
            cv2.putText(frame, text_overlay, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Use BGR888 directly to avoid cvtColor overhead
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        if hasattr(QImage, "Format_BGR888"):
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
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

            # AUTO CLEANUP if FPS drops significantly (with cooldown)
            if len(self.fps_history) >= 5:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                if self.current_fps < avg_fps * 0.7:
                    since_last_cleanup = time.time() - self.last_auto_cleanup_time
                    if since_last_cleanup >= self.auto_cleanup_cooldown:
                        self.last_auto_cleanup_time = time.time()
                        QTimer.singleShot(0, self.manual_cleanup)

            if len(self.fps_history) > 0:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.fps_label.setText(f"FPS: {self.current_fps:.1f} (avg: {avg_fps:.1f})")
            else:
                self.fps_label.setText(f"FPS: {self.current_fps:.1f}")

            if self.current_fps >= 20:
                color = '#4ade80'
            elif self.current_fps >= 10:
                color = '#fbbf24'
            else:
                color = '#ef4444'
            if color != self.last_fps_color:
                self.fps_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
                self.last_fps_color = color

            self.fps_counter = 0
            self.fps_start_time = time.time()

            # Update stats only once per second
            if stats['total_objects'] > 0:
                stats_text = f"Total Objects: {stats['total_objects']}\n\n"
                stats_text += "Detections:\n"
                for cls_name, count in stats['class_counts'].items():
                    stats_text += f"  {cls_name}: {count}\n"
            else:
                stats_text = "No active detections.\n\nTips:\n- Use Ultra mode for livestreams\n- Lower resize to 75% or 50%\n- Keep tracker on Simple or SORT for higher FPS"

            if stats_text != self.last_stats_text:
                self.stats_text.setPlainText(stats_text)
                self.last_stats_text = stats_text

            if (now - self.last_runtime_refresh_time) >= self.runtime_refresh_interval:
                self._refresh_runtime_overview()
                self.last_runtime_refresh_time = now

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
            'custom_model_path': self.custom_model_path,
            'tracker_choice': self.tracker_combo.currentText(),
            
            # Detection settings
            'confidence': self.confidence_slider.value(),
            
            # Performance settings
            'frame_skip': self.frame_skip_slider.value(),
            'resize_scale': self.resize_combo.currentText(),
            'smooth_mode': self.smooth_mode,
            'processing_mode': self._get_processing_mode(),
            
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
            saved_model_choice = settings.get('model_choice')
            model_choice_loaded = False
            if 'model_choice' in settings:
                index = self.model_combo.findText(settings['model_choice'])
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                    model_choice_loaded = True
                elif 'YOLOv26n' in settings['model_choice']:
                    index = self.model_combo.findText("YOLOv26n (GPU Recommended)")
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        model_choice_loaded = True
                elif 'YOLOv11s' in settings['model_choice']:
                    index = self.model_combo.findText("YOLOv11s (Fast)")
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        model_choice_loaded = True
            if 'custom_model_path' in settings:
                self.custom_model_path = settings['custom_model_path']
            elif 'best_model_choice' in settings and (
                not model_choice_loaded or saved_model_choice == "Custom model..."
            ):
                saved_best_model = settings['best_model_choice']
                if isinstance(saved_best_model, str) and saved_best_model.startswith("Custom: "):
                    self.custom_model_path = saved_best_model.replace("Custom: ", "").strip()
                    index = self.model_combo.findText("Custom model...")
                    if index >= 0:
                        self.model_combo.blockSignals(True)
                        self.model_combo.setCurrentIndex(index)
                        self.model_combo.blockSignals(False)
                elif 'Train2' in str(saved_best_model):
                    index = self.model_combo.findText("Train2 (Multi-class)")
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                elif 'Train1' in str(saved_best_model):
                    index = self.model_combo.findText("Train1 (Person model)")
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
            
            if 'tracker_choice' in settings:
                index = self.tracker_combo.findText(settings['tracker_choice'])
                if index >= 0:
                    self.tracker_combo.setCurrentIndex(index)
                elif 'SORT' in settings['tracker_choice']:
                    sort_index = self.tracker_combo.findText("SORT (Fast)")
                    if sort_index >= 0:
                        self.tracker_combo.setCurrentIndex(sort_index)
                elif 'DeepSort' in settings['tracker_choice']:
                    deep_index = self.tracker_combo.findText("DeepSORT (Stable)")
                    if deep_index >= 0:
                        self.tracker_combo.setCurrentIndex(deep_index)
            
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
                if hasattr(self, 'btn_smooth_toggle'):
                    self.btn_smooth_toggle.blockSignals(True)
                    self.btn_smooth_toggle.setChecked(self.smooth_mode)
                    self.btn_smooth_toggle.blockSignals(False)
                    self.toggle_smooth_mode(self.smooth_mode, self.btn_smooth_toggle)

            # Mutually-exclusive processing mode
            saved_mode = settings.get('processing_mode', 'standard')
            self._apply_processing_mode(saved_mode, trigger_reload=False)
            
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
            
            self._refresh_custom_model_input()
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


def _ensure_python310_runtime() -> bool:
    """
    Relaunch with Python 3.10 when the user starts the app via `py pyqt_app.py`
    and the launcher resolves to another interpreter such as Python 3.13.
    """
    if sys.version_info[:2] == (3, 10):
        return True

    if os.environ.get("PYQT_APP_PY310_RELAUNCHED") == "1":
        return True

    launcher = shutil.which("py")
    if not launcher:
        print(
            f"[WARN] Running on Python {sys.version_info.major}.{sys.version_info.minor}. "
            "GPU runtime is validated on Python 3.10."
        )
        return True

    env = dict(os.environ)
    env["PYQT_APP_PY310_RELAUNCHED"] = "1"
    try:
        subprocess.Popen(
            [launcher, "-3.10", str(Path(__file__).resolve())],
            cwd=str(Path(__file__).resolve().parent),
            env=env,
        )
        print("Re-launching with Python 3.10 for the CUDA-enabled runtime...")
        return False
    except Exception as exc:
        print(f"[WARN] Could not relaunch with Python 3.10: {exc}")
        return True


def main():
    """Main entry point"""
    if not _ensure_python310_runtime():
        return

    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

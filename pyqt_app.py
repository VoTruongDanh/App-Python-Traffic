"""
PyQt5 Desktop App - Real-time Object Tracking
High-performance alternative to Streamlit for local use
"""
from src.core import config
from src.inference.model_loader import load_yolo_models, initialize_tracker

import os
import shutil
import subprocess
import sys
import threading
import cv2
import numpy as np
import urllib.request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox,
                             QFileDialog, QLineEdit, QGroupBox, QGridLayout, QTextEdit,
                             QDialog, QProgressBar, QListWidget, QListWidgetItem, QToolButton,
                             QScrollArea, QFrame, QSizePolicy, QOpenGLWidget, QStackedWidget,
                             QMenu, QInputDialog, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPainter, QColor,
    QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QIcon
)
import time
from pathlib import Path

from src.processing.video_processor import VideoProcessor
from src.processing.latest_frame_buffer import LatestFrameBuffer
# from src.processing.video_processor_optimized import VideoProcessorOptimized as VideoProcessor
from src.tracking.roi_manager import ROIManager

# Global state for model selection (replacement for st.session_state)
class AppState:
    model_choice = "YOLOv26n (Fastest)"
    tracker_choice = "SORT (Fast)"
    best_model_choice = "None (Use base YOLO only)"

app_state = AppState()


class SharedFrameStore:
    """Thread-safe holder for the newest processed frame and metadata."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._stats = {'total_objects': 0, 'class_counts': {}}
        self._version = 0

    def publish(self, frame, stats):
        with self._lock:
            self._frame = frame
            self._stats = stats
            self._version += 1
            return self._version

    def read(self):
        with self._lock:
            return self._version, self._frame, self._stats

    def clear(self):
        with self._lock:
            self._frame = None
            self._stats = {'total_objects': 0, 'class_counts': {}}
            self._version = 0


class VideoOpenGLWidget(QOpenGLWidget):
    """Render frames with OpenGL shader; fallback to painter if GL path fails."""

    _GL_TRIANGLE_STRIP = 0x0005
    _GL_COLOR_BUFFER_BIT = 0x00004000

    def __init__(self):
        super().__init__()
        self._frame = None
        self._message = "Load video or livestream to begin"
        self._shader_program = None
        self._texture = None
        self._texture_size = (0, 0)
        self._gl_ready = False
        self._gl_failed = False
        self._vertices = np.array(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            dtype=np.float32
        )
        self._tex_coords = np.array(
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            dtype=np.float32
        )
        self._display_rect = QRect()

    def setText(self, text):
        self._message = text or ""
        self.update()

    def clear_frame(self):
        self._frame = None
        self.update()

    def has_frame(self):
        return self._frame is not None

    def content_rect(self, frame_size=None):
        if frame_size is None and self._frame is not None:
            frame_size = (self._frame.shape[1], self._frame.shape[0])
        if not frame_size:
            return QRect(0, 0, self.width(), self.height())

        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0 or self.width() <= 0 or self.height() <= 0:
            return QRect(0, 0, self.width(), self.height())

        scale = min(self.width() / frame_w, self.height() / frame_h)
        draw_w = int(frame_w * scale)
        draw_h = int(frame_h * scale)
        off_x = (self.width() - draw_w) // 2
        off_y = (self.height() - draw_h) // 2
        return QRect(off_x, off_y, draw_w, draw_h)

    def set_frame(self, frame):
        self._frame = frame
        if frame is not None:
            self._message = ""
        self.update()

    def initializeGL(self):
        try:
            self._init_shader_program()
            self._gl_ready = True
        except Exception:
            self._gl_failed = True
            self._gl_ready = False

    def _init_shader_program(self):
        vertex_shader = """
            attribute vec2 position;
            attribute vec2 texCoord;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                vTexCoord = texCoord;
            }
        """
        fragment_shader = """
            varying vec2 vTexCoord;
            uniform sampler2D frameTex;
            void main() {
                vec4 c = texture2D(frameTex, vTexCoord);
                gl_FragColor = vec4(c.b, c.g, c.r, 1.0);
            }
        """

        program = QOpenGLShaderProgram(self.context())
        if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader):
            raise RuntimeError(program.log())
        if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader):
            raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        self._shader_program = program

    def _ensure_texture(self, width, height):
        if self._texture and self._texture_size == (width, height):
            return

        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.setFormat(QOpenGLTexture.RGB8_UNorm)
        texture.setSize(width, height)
        texture.setWrapMode(QOpenGLTexture.ClampToEdge)
        texture.setMinificationFilter(QOpenGLTexture.Linear)
        texture.setMagnificationFilter(QOpenGLTexture.Linear)
        texture.allocateStorage(QOpenGLTexture.RGB, QOpenGLTexture.UInt8)
        self._texture = texture
        self._texture_size = (width, height)

    def _upload_texture(self, frame):
        try:
            self._texture.setData(QOpenGLTexture.RGB, QOpenGLTexture.UInt8, frame)
        except TypeError:
            self._texture.setData(QOpenGLTexture.RGB, QOpenGLTexture.UInt8, frame.data)

    def _draw_with_shader(self, frame):
        if frame is None:
            return False
        if not self._gl_ready or self._gl_failed or self._shader_program is None:
            return False
        if frame.ndim != 3 or frame.shape[2] != 3:
            return False

        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return False

        try:
            self._ensure_texture(w, h)
            self._upload_texture(frame)
            functions = self.context().functions()

            functions.glClearColor(0.03, 0.06, 0.09, 1.0)
            functions.glClear(self._GL_COLOR_BUFFER_BIT)

            draw_rect = self.content_rect((w, h))
            self._display_rect = draw_rect
            viewport_y = max(0, self.height() - draw_rect.y() - draw_rect.height())
            functions.glViewport(draw_rect.x(), viewport_y, draw_rect.width(), draw_rect.height())

            self._shader_program.bind()
            self._texture.bind(0)
            self._shader_program.setUniformValue("frameTex", 0)

            pos_loc = self._shader_program.attributeLocation("position")
            tex_loc = self._shader_program.attributeLocation("texCoord")
            self._shader_program.enableAttributeArray(pos_loc)
            self._shader_program.enableAttributeArray(tex_loc)
            self._shader_program.setAttributeArray(pos_loc, self._vertices, 2)
            self._shader_program.setAttributeArray(tex_loc, self._tex_coords, 2)

            functions.glDrawArrays(self._GL_TRIANGLE_STRIP, 0, 4)

            self._shader_program.disableAttributeArray(pos_loc)
            self._shader_program.disableAttributeArray(tex_loc)
            self._texture.release()
            self._shader_program.release()

            functions.glViewport(0, 0, self.width(), self.height())
            return True
        except Exception:
            self._gl_failed = True
            return False

    def _draw_with_painter(self, frame):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 16, 24))
        if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            draw_rect = self.content_rect((w, h))
            self._display_rect = draw_rect
            painter.drawImage(draw_rect, image)
        elif self._message:
            painter.setPen(QColor(216, 229, 239))
            painter.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
            painter.drawText(self.rect(), Qt.AlignCenter, self._message)
        painter.end()

    def paintGL(self):
        frame = self._frame
        rendered = self._draw_with_shader(frame)
        if not rendered:
            self._draw_with_painter(frame)


class ModelLoadThread(QThread):
    """Background loader for heavy model + tracker initialization."""

    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, tracker_choice: str):
        super().__init__()
        self.tracker_choice = tracker_choice

    def run(self):
        try:
            self.progress.emit("Loading detection models...")
            model_person, model_vehicle = load_yolo_models()
            self.progress.emit("Initializing tracker...")
            tracker = initialize_tracker(self.tracker_choice)
            self.loaded.emit({
                'model_person': model_person,
                'model_vehicle': model_vehicle,
                'tracker': tracker,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class StreamResolveThread(QThread):
    """Background resolver for YouTube sources to keep UI responsive."""

    progress = pyqtSignal(str)
    resolved = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, stream_url: str, quality_text: str):
        super().__init__()
        self.stream_url = stream_url
        self.quality_text = quality_text

    @staticmethod
    def _quality_to_profile(quality_text: str):
        text = (quality_text or "").lower()
        if "1080" in text:
            return "1080p", 1080
        if "720" in text:
            return "720p", 720
        if "480" in text:
            return "480p", 480
        return "360p", 360

    @staticmethod
    def _yt_logger():
        class _QuietLogger:
            def debug(self, msg):
                return

            def warning(self, msg):
                return

            def error(self, msg):
                return

        return _QuietLogger()

    @staticmethod
    def _js_runtime_options():
        return {
            'quiet': True,
            'no_warnings': True,
            'logger': StreamResolveThread._yt_logger(),
        }

    def run(self):
        try:
            import yt_dlp
            import random
            import string

            quality_name, height = self._quality_to_profile(self.quality_text)
            self.progress.emit("🔄 Checking YouTube URL...")

            ydl_opts_check = self._js_runtime_options()
            with yt_dlp.YoutubeDL(ydl_opts_check) as ydl:
                info = ydl.extract_info(self.stream_url, download=False)
                title = info.get('title', 'Unknown')[:30]
                is_live = info.get('is_live', False)

            if is_live:
                self.progress.emit(f"🔴 Live: {title}... (Getting URL)")
                ydl_opts = {
                    'format': f'best[height<={height}]',
                }
                ydl_opts.update(self._js_runtime_options())
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.stream_url, download=False)
                    resolved_url = info['url']
                self.resolved.emit(resolved_url, f"✅ 🔴 Live {quality_name}: {title}...")
                return

            self.progress.emit(f"⏬ Downloading: {title}...")
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            temp_file = temp_dir / f"downloaded_{random_id}.mp4"

            ydl_opts = {
                'format': f'best[height<={height}][ext=mp4]/best[height<={height}]',
                'outtmpl': str(temp_file),
            }
            ydl_opts.update(self._js_runtime_options())
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.stream_url])

            self.resolved.emit(str(temp_file), f"✅ 🎥 Video {quality_name}: {title}...")
        except Exception as exc:
            self.failed.emit(str(exc))


def _reset_tracker_instance(tracker):
    """Clear stale tracker state across SORT, SimpleTracker, and DeepSort."""
    if tracker is None:
        return

    for attr_name in ("tracks", "trackers"):
        track_list = getattr(tracker, attr_name, None)
        if isinstance(track_list, list):
            track_list.clear()

    if hasattr(tracker, 'frame_count'):
        tracker.frame_count = 0
    if hasattr(tracker, 'next_id'):
        tracker.next_id = 1
    if hasattr(tracker, '_last_ts'):
        tracker._last_ts = time.monotonic()


def _reset_processor_tracking_state(processor):
    if processor is None:
        return
    _reset_tracker_instance(getattr(processor, 'tracker', None))
    trails = getattr(processor, 'trails', None)
    if isinstance(trails, dict):
        trails.clear()
    if hasattr(processor, 'last_detections'):
        processor.last_detections = []
    if hasattr(processor, 'detection_buffer'):
        try:
            processor.detection_buffer.clear()
        except Exception:
            pass
    if hasattr(processor, 'detection_version'):
        processor.detection_version = 0
    if hasattr(processor, 'last_consumed_detection_version'):
        processor.last_consumed_detection_version = -1
    if hasattr(processor, 'pending_frame'):
        try:
            lock = getattr(processor, 'frame_lock', None)
            if lock is not None:
                with lock:
                    processor.pending_frame = None
                    if hasattr(processor, 'pending_params'):
                        processor.pending_params = None
            else:
                processor.pending_frame = None
                if hasattr(processor, 'pending_params'):
                    processor.pending_params = None
        except Exception:
            pass


class VideoThread(QThread):
    """Thread for processing video/livestream - SIMPLE & STABLE"""
    frame_ready = pyqtSignal(int)
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
        self.output_fps_limit = 30
        
        self.frame_buffer = []  # DISABLED: Buffer causes 2-3 frame delay
        self.max_buffer_size = 1  # Keep only 1 frame (effectively disabled)
        self.frame_buf = None
        self._last_stats = {'total_objects': 0, 'class_counts': {}}
        self.shared_frame_store = None
        self._last_signal_emit_ts = 0.0
        
    def set_source(self, source):
        self.source = source
        
    def set_processor(self, processor):
        self.processor = processor
        
    def set_params(self, frame_skip, resize_scale, output_fps_limit=None):
        # Strict realtime: never skip frames in processing thread.
        self.frame_skip = 0
        self.resize_scale = resize_scale
        if output_fps_limit is not None:
            self.output_fps_limit = max(5, int(output_fps_limit))

    def set_shared_frame_store(self, shared_frame_store):
        self.shared_frame_store = shared_frame_store

    def _publish_frame(self, frame, stats):
        if self.shared_frame_store is None:
            return
        version = self.shared_frame_store.publish(frame, stats)
        now = time.perf_counter()
        # Coalesce cross-thread notifications to prevent Qt event-queue backlog.
        if (now - self._last_signal_emit_ts) >= (1.0 / 60.0):
            self._last_signal_emit_ts = now
            self.frame_ready.emit(version)
        
    def stop(self):
        self.running = False
        
    def run(self):
        """Main processing loop — realtime-first, no pacing sleep in inference thread."""
        if not self.source or not self.processor:
            return

        self.running = True

        is_rtsp = isinstance(self.source, str) and self.source.startswith('rtsp://')
        is_http = isinstance(self.source, str) and ('http' in self.source or 'https' in self.source)
        is_livestream = is_rtsp or is_http
        is_file = not is_livestream and isinstance(self.source, str)
        is_webcam = isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())

        if is_rtsp:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.source)
            if is_file:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            else:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"Error: Cannot open source {self.source}")
            self.finished.emit()
            return

        # Strict timeline mode: read frames sequentially to avoid perceived fast-forward.
        self.frame_buf = None

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 240:
            source_fps = 30.0

        # Strict realtime: disable skip/catch-up branches entirely.
        self.frame_skip = 0
        skip_counter = 0
        last_processed_frame = None
        last_success_time = time.time()
        consecutive_failures = 0
        frames_rendered = 0
        target_display_interval = 1.0 / max(5.0, float(self.output_fps_limit))
        forced_tracking_frames = 0
        last_input_timestamp = 0.0

        # Keep local-file playback at source speed and avoid catch-up fast-forward.
        file_frame_interval = (1.0 / source_fps) if is_file and source_fps > 0 else 0.0
        next_file_frame_due = time.perf_counter()

        processor_supports_async = hasattr(self.processor, 'process_frame_threaded')
        supports_tracking_only = hasattr(self.processor, 'process_frame_tracking_only')

        try:
            while self.running:
                if file_frame_interval > 0:
                    now = time.perf_counter()
                    if now < next_file_frame_due:
                        time.sleep(min(0.01, next_file_frame_due - now))
                        continue

                frame_ts = 0.0
                ret, frame = cap.read()
                if ret:
                    frame_ts = time.monotonic()

                if not ret:
                    consecutive_failures += 1
                    time_since_last = time.time() - last_success_time

                    if (is_rtsp or is_livestream) and consecutive_failures == 1:
                        _reset_processor_tracking_state(self.processor)

                    if is_rtsp and consecutive_failures > 5 and time_since_last > 0.3:
                        print("RTSP reconnecting...")
                        cap.release()

                        time.sleep(0.2)
                        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
                        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        consecutive_failures = 0
                        _reset_processor_tracking_state(self.processor)
                        continue

                    if not is_rtsp and time_since_last > 2.0:
                        break

                    time.sleep(0.005)
                    continue

                consecutive_failures = 0
                last_success_time = time.time()
                last_input_timestamp = frame_ts

                use_tracking_only = False
                if self.frame_skip > 0 and not processor_supports_async:
                    skip_counter = (skip_counter + 1) % (self.frame_skip + 1)
                    if skip_counter != 0:
                        use_tracking_only = supports_tracking_only
                        if not use_tracking_only:
                            if last_processed_frame is None:
                                self._publish_frame(frame, self._last_stats)
                            continue

                if (
                    not use_tracking_only
                    and not processor_supports_async
                    and supports_tracking_only
                    and forced_tracking_frames > 0
                ):
                    use_tracking_only = True
                    forced_tracking_frames -= 1

                process_started = time.perf_counter()
                try:
                    if use_tracking_only:
                        processed_frame, stats = self.processor.process_frame_tracking_only(
                            frame,
                            frame_timestamp=frame_ts
                        )
                    elif processor_supports_async:
                        processed_frame, stats = self.processor.process_frame_threaded(
                            frame,
                            self.resize_scale,
                            self.max_det,
                            frame_timestamp=frame_ts
                        )
                    else:
                        processed_frame, stats = self.processor.process_frame(
                            frame,
                            self.resize_scale,
                            self.max_det,
                            frame_timestamp=frame_ts
                        )
                    
                    last_processed_frame = processed_frame
                    self._last_stats = stats
                    frames_rendered += 1
                    self._publish_frame(processed_frame, stats)
                    process_duration = time.perf_counter() - process_started
                    if not processor_supports_async and supports_tracking_only and not use_tracking_only:
                        if process_duration > (target_display_interval * 1.25):
                            extra_tracking = int(process_duration / target_display_interval) - 1
                            forced_tracking_frames = max(
                                forced_tracking_frames,
                                min(6, max(1, extra_tracking))
                            )
                    if file_frame_interval > 0:
                        # Strict pacing: do not "catch up" by bursting frames when behind.
                        next_file_frame_due = time.perf_counter() + file_frame_interval
                except TypeError:
                    if use_tracking_only:
                        processed_frame, stats = self.processor.process_frame_tracking_only(frame)
                    elif processor_supports_async:
                        processed_frame, stats = self.processor.process_frame_threaded(
                            frame,
                            self.resize_scale,
                            self.max_det
                        )
                    else:
                        processed_frame, stats = self.processor.process_frame(
                            frame,
                            self.resize_scale,
                            self.max_det
                        )

                    last_processed_frame = processed_frame
                    self._last_stats = stats
                    frames_rendered += 1
                    self._publish_frame(processed_frame, stats)
                    process_duration = time.perf_counter() - process_started
                    if not processor_supports_async and supports_tracking_only and not use_tracking_only:
                        if process_duration > (target_display_interval * 1.25):
                            extra_tracking = int(process_duration / target_display_interval) - 1
                            forced_tracking_frames = max(
                                forced_tracking_frames,
                                min(6, max(1, extra_tracking))
                            )
                    if file_frame_interval > 0:
                        # Strict pacing: do not "catch up" by bursting frames when behind.
                        next_file_frame_due = time.perf_counter() + file_frame_interval
                except Exception as e:
                    print(f"Inference error: {e}")
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
        self.model_loader_thread = None
        self.stream_resolver_thread = None
        self.pending_youtube_source = None
        self.processor = None
        self.model_person = None
        self.model_vehicle = None
        self.tracker = None
        self.fps_counter = 0
        self.fps_start_time = time.perf_counter()
        self.current_fps = 0
        self.render_fps_counter = 0
        self.render_fps_start_time = time.perf_counter()
        self.current_render_fps = 0.0
        self.current_detect_fps = 0.0
        self.fps_sample_interval = 0.5
        self.fps_smoothing = 0.25
        self.latest_active_objects = 0
        self.latest_total_objects = 0
        self.last_display_time = 0.0
        self.display_target_fps = 30
        self.display_frame_interval = 1.0 / self.display_target_fps
        self.last_auto_cleanup_time = 0.0
        self.auto_cleanup_cooldown = 20.0
        self.last_runtime_refresh_time = 0.0
        self.runtime_refresh_interval = 2.0
        self.last_stats_text = ""
        self.last_fps_color = ""
        self.last_alert_color = ""
        self.vehicle_alert_limit = 20
        self.vehicle_alert_scope = "ROI"
        self.alert_yellow_ratio = 0.7
        self.video_source = None
        self.source_profiles = []
        self.source_profile_counter = 1
        self.active_source_profile_id = None
        self.source_preview_cache = {}
        self.source_preview_meta = {}
        self.preview_retry_after = {}
        self.preview_width = 256
        self.preview_height = 144
        self.source_grid_columns = 5
        self.source_tile_text_size = "Medium"
        self.pin_active_source_first = True
        self.admin_row_profile_ids = []
        self.preview_load_queue = []
        self.preview_load_pending = set()
        self.preview_loader_timer = QTimer(self)
        self.preview_loader_timer.setSingleShot(True)
        self.preview_loader_timer.timeout.connect(self._process_preview_queue)
        self.last_preview_capture_ts = 0.0
        self.preview_capture_interval = 1.2
        self.shared_frame_store = SharedFrameStore()
        self.pending_frame_version = 0
        self.metrics_last_version = 0
        self.pending_display_frame = None
        self.pending_display_stats = {'total_objects': 0, 'class_counts': {}}
        self.pending_frame_dirty = False
        self.settings_loading = True
        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.timeout.connect(self.save_settings)
        self.adaptive_fps = False
        self.last_adaptive_tune_time = 0.0
        self.adaptive_tune_cooldown = 6.0
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(33)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.timeout.connect(self._render_latest_frame)

        # Debounced restart state for heavy changes while stream is running.
        self.pending_restart = False
        self.pending_reload_models = False
        self.pending_start_after_model_load = False
        self.models_loading = False
        self.model_load_lowered_confidence = None
        self.restart_message = ""
        self.restart_timer = QTimer(self)
        self.restart_timer.setSingleShot(True)
        self.restart_timer.timeout.connect(self._apply_pending_restart)
        
        # Detection settings
        self.max_det = 20  # Default max detections
        self.tracker_max_age = 1  # Default tracker max age
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
        self.settings_loading = False
        
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

        self.btn_toggle_sidebar = QPushButton("Mo Cau Hinh")
        self.btn_toggle_sidebar.setObjectName("secondaryAction")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        title_row.addWidget(self.btn_toggle_sidebar)

        self.btn_show_admin = QPushButton("Quan Ly Nguon")
        self.btn_show_admin.setObjectName("secondaryAction")
        self.btn_show_admin.clicked.connect(self.show_admin_dashboard)
        title_row.addWidget(self.btn_show_admin)
        left_panel.addLayout(title_row)
        
        # Main content stack: admin dashboard (default) + large player view
        self.main_view_stack = QStackedWidget()
        self.main_view_stack.setObjectName("mainViewStack")

        admin_page = QWidget()
        admin_layout = QVBoxLayout(admin_page)
        admin_layout.setContentsMargins(10, 10, 10, 10)
        admin_layout.setSpacing(10)

        self.admin_heading_label = QLabel("Quan Ly Nguon Camera")
        self.admin_heading_label.setObjectName("videoTitle")
        admin_layout.addWidget(self.admin_heading_label)

        self.admin_subtitle_label = QLabel("Dat ten de phan loai nguon nhanh hon. Chon tile de mo player lon.")
        self.admin_subtitle_label.setObjectName("subtleInfo")
        self.admin_subtitle_label.setWordWrap(True)
        admin_layout.addWidget(self.admin_subtitle_label)

        admin_name_row = QHBoxLayout()
        self.admin_name_input = QLineEdit()
        self.admin_name_input.setPlaceholderText("Ten hien thi (vi du: Cong 1, Tang 2)")
        admin_name_row.addWidget(self.admin_name_input)

        self.btn_admin_rename_source = QPushButton("Luu Ten")
        self.btn_admin_rename_source.setObjectName("secondaryAction")
        self.btn_admin_rename_source.clicked.connect(self.rename_selected_admin_source)
        self.btn_admin_rename_source.setEnabled(False)
        admin_name_row.addWidget(self.btn_admin_rename_source)
        admin_layout.addLayout(admin_name_row)

        admin_input_row = QHBoxLayout()
        self.admin_source_input = QLineEdit()
        self.admin_source_input.setPlaceholderText("rtsp://..., https://..., youtube url or camera id")
        admin_input_row.addWidget(self.admin_source_input)

        self.btn_admin_add_url = QPushButton("Them URL/Cam")
        self.btn_admin_add_url.clicked.connect(self.add_admin_source_from_input)
        admin_input_row.addWidget(self.btn_admin_add_url)

        self.btn_admin_add_file = QPushButton("Them File")
        self.btn_admin_add_file.setObjectName("secondaryAction")
        self.btn_admin_add_file.clicked.connect(self.add_admin_source_from_file)
        admin_input_row.addWidget(self.btn_admin_add_file)
        admin_layout.addLayout(admin_input_row)

        admin_grid_row = QHBoxLayout()
        admin_grid_row.addWidget(QLabel("Nguon / hang:"))
        self.source_grid_combo = QComboBox()
        self.source_grid_combo.addItems(["2", "3", "4", "5", "6", "7"])
        self.source_grid_combo.setCurrentText(str(self.source_grid_columns))
        self.source_grid_combo.currentTextChanged.connect(self.on_source_grid_columns_changed)
        admin_grid_row.addWidget(self.source_grid_combo)

        admin_grid_row.addWidget(QLabel("Chu:"))
        self.source_text_size_combo = QComboBox()
        self.source_text_size_combo.addItems(["Small", "Medium", "Large"])
        self.source_text_size_combo.setCurrentText(self.source_tile_text_size)
        self.source_text_size_combo.currentTextChanged.connect(self.on_source_text_size_changed)
        admin_grid_row.addWidget(self.source_text_size_combo)

        self.pin_active_checkbox = QCheckBox("Ghim ACTIVE dau hang")
        self.pin_active_checkbox.setChecked(self.pin_active_source_first)
        self.pin_active_checkbox.toggled.connect(self.on_pin_active_source_changed)
        admin_grid_row.addWidget(self.pin_active_checkbox)
        admin_grid_row.addStretch()
        admin_layout.addLayout(admin_grid_row)

        self.admin_source_list = QListWidget()
        self.admin_source_list.setObjectName("adminSourceList")
        self.admin_source_list.setMinimumHeight(320)
        self.admin_source_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.admin_source_list.setViewMode(QListWidget.IconMode)
        self.admin_source_list.setIconSize(QSize(self.preview_width, self.preview_height))
        self.admin_source_list.setGridSize(QSize(self.preview_width + 42, self.preview_height + 104))
        self.admin_source_list.setResizeMode(QListWidget.Adjust)
        self.admin_source_list.setMovement(QListWidget.Static)
        self.admin_source_list.setUniformItemSizes(True)
        self.admin_source_list.setSpacing(10)
        self.admin_source_list.setWrapping(True)
        self.admin_source_list.setWordWrap(False)
        self.admin_source_list.setTextElideMode(Qt.ElideRight)
        self.admin_source_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.admin_source_list.currentRowChanged.connect(self.on_admin_source_selected)
        self.admin_source_list.customContextMenuRequested.connect(self.show_admin_source_context_menu)
        self.admin_source_list.itemDoubleClicked.connect(lambda _: self.open_selected_admin_source())
        self.admin_source_list.itemActivated.connect(lambda _: self.open_selected_admin_source())
        admin_layout.addWidget(self.admin_source_list, 1)

        admin_btn_row = QHBoxLayout()
        self.btn_admin_open_source = QPushButton("Mo Man Hinh Lon")
        self.btn_admin_open_source.setObjectName("primaryAction")
        self.btn_admin_open_source.clicked.connect(self.open_selected_admin_source)
        self.btn_admin_open_source.setEnabled(False)
        admin_btn_row.addWidget(self.btn_admin_open_source)

        self.btn_admin_refresh_sources = QPushButton("Lam Moi Preview")
        self.btn_admin_refresh_sources.setObjectName("secondaryAction")
        self.btn_admin_refresh_sources.clicked.connect(self.refresh_admin_source_previews)
        admin_btn_row.addWidget(self.btn_admin_refresh_sources)
        admin_layout.addLayout(admin_btn_row)

        self._apply_admin_source_grid_columns(refresh_cards=False)

        player_page = QWidget()
        player_layout = QVBoxLayout(player_page)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(0)

        # Video display label
        self.video_label = VideoOpenGLWidget()
        self.video_label.setObjectName("videoSurface")
        self.video_label.setMinimumWidth(0)
        self.video_label.setMinimumHeight(320)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if hasattr(self.video_label, 'setAlignment'):
            self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Load video or livestream to begin")
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.on_video_label_click
        player_layout.addWidget(self.video_label)

        self.main_view_stack.addWidget(admin_page)
        self.main_view_stack.addWidget(player_page)
        left_panel.addWidget(self.main_view_stack, 1)
        
        # New Consolidated Status Dashboard
        self._setup_status_panel(left_panel)
        
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
        # Keep live video area dominant; status row remains compact.
        left_panel.setStretch(0, 12)
        left_panel.setStretch(1, 0)
        left_panel.setStretch(2, 0)
        
        # Right panel - Controls and stats
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        runtime_shell = QFrame()
        self.runtime_shell = runtime_shell
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
        self.model_group = model_group
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
        custom_row.setSpacing(6)
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setReadOnly(True)
        self.custom_model_input.setPlaceholderText("No custom model selected")
        self.custom_model_input.setMinimumWidth(0)
        self.custom_model_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        custom_row.addWidget(self.custom_model_input)

        self.btn_browse_custom_model = QPushButton("Browse...")
        self.btn_browse_custom_model.setObjectName("secondaryAction")
        self.btn_browse_custom_model.setMinimumWidth(0)
        self.btn_browse_custom_model.setMaximumWidth(110)
        self.btn_browse_custom_model.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_browse_custom_model.clicked.connect(self.browse_custom_model)
        custom_row.addWidget(self.btn_browse_custom_model)
        model_layout.addLayout(custom_row)
        
        model_layout.addWidget(QLabel("Tracking Mode:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["SORT (Fast)", "ByteTrack (Balanced)", "DeepSORT (Stable)"])
        self.tracker_combo.setCurrentIndex(0)
        self.tracker_combo.setToolTip(
            "SORT: faster tracking and lighter CPU/GPU load\n"
            "ByteTrack: keeps IDs stable in crowded scenes with moderate cost\n"
            "DeepSORT: more stable IDs, slower than SORT"
        )
        self.tracker_combo.currentTextChanged.connect(self.on_tracker_changed)
        model_layout.addWidget(self.tracker_combo)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        self.detection_group = detection_group
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

        detection_layout.addWidget(QLabel("Duplicate Box IOU Threshold:"))
        self.duplicate_iou_slider = QSlider(Qt.Horizontal)
        self.duplicate_iou_slider.setMinimum(30)
        self.duplicate_iou_slider.setMaximum(90)
        self.duplicate_iou_slider.setValue(55)
        self.duplicate_iou_slider.setToolTip(
            "Lower value removes more overlapping boxes (aggressive)\n"
            "Higher value keeps more boxes (conservative)"
        )
        self.duplicate_iou_slider.valueChanged.connect(self.on_duplicate_iou_changed)
        detection_layout.addWidget(self.duplicate_iou_slider)

        self.duplicate_iou_label = QLabel("0.55")
        detection_layout.addWidget(self.duplicate_iou_label)
        
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
        self.tracker_age_slider.setValue(1)  # Default: 1
        self.tracker_age_slider.valueChanged.connect(self.on_tracker_age_changed)
        self.tracker_age_slider.setToolTip(
            "How long to keep track without detection\n"
            "Higher = more stable ID (better for lag)\n"
            "Lower = remove ghost tracks faster"
        )
        detection_layout.addWidget(self.tracker_age_slider)
        
        self.tracker_age_label = QLabel("1 frame")
        detection_layout.addWidget(self.tracker_age_label)

        detection_layout.addWidget(QLabel("Vehicle Alert Scope (Auto):"))
        self.vehicle_alert_scope_combo = QComboBox()
        self.vehicle_alert_scope_combo.addItems(["Auto (ROI if available)"])
        self.vehicle_alert_scope_combo.setCurrentIndex(0)
        self.vehicle_alert_scope_combo.setEnabled(False)
        self.vehicle_alert_scope_combo.setToolTip("Tu dong: neu co ROI active thi canh bao theo ROI, neu khong thi theo toan bo khung hinh")
        detection_layout.addWidget(self.vehicle_alert_scope_combo)

        detection_layout.addWidget(QLabel("Vehicle Alert Limit:"))
        self.vehicle_alert_limit_slider = QSlider(Qt.Horizontal)
        self.vehicle_alert_limit_slider.setMinimum(1)
        self.vehicle_alert_limit_slider.setMaximum(200)
        self.vehicle_alert_limit_slider.setValue(self.vehicle_alert_limit)
        self.vehicle_alert_limit_slider.setToolTip("Nguong canh bao: >100% do, >=70% vang, thap hon la xanh")
        self.vehicle_alert_limit_slider.valueChanged.connect(self.on_vehicle_alert_limit_changed)
        detection_layout.addWidget(self.vehicle_alert_limit_slider)

        self.vehicle_alert_limit_label = QLabel(f"{self.vehicle_alert_limit} vehicles")
        detection_layout.addWidget(self.vehicle_alert_limit_label)
        
        detection_group.setLayout(detection_layout)
        right_panel.addWidget(detection_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance Optimization")
        self.perf_group = perf_group
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
        self.processing_mode_combo.setCurrentIndex(0)
        self.processing_mode_combo.setEnabled(False)
        perf_layout.addWidget(self.processing_mode_combo)

        self.processing_mode_hint = QLabel("Strict realtime mode: Standard processor is locked to avoid async burst/stutter artifacts.")
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
        preset_layout.setSpacing(6)
        
        btn_quality = QPushButton("Quality")
        btn_quality.setMinimumWidth(0)
        btn_quality.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_quality.setToolTip("Skip:0 (~12-15 FPS)\nBest quality, slow")
        btn_quality.clicked.connect(lambda: self.apply_preset(0, 100))
        preset_layout.addWidget(btn_quality)
        
        btn_balanced = QPushButton("Balanced")
        btn_balanced.setMinimumWidth(0)
        btn_balanced.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_balanced.setToolTip("Skip:1 + Resize:75%\nBest balance for most cameras")
        btn_balanced.clicked.connect(lambda: self.apply_preset(1, 75))
        preset_layout.addWidget(btn_balanced)
        
        btn_speed = QPushButton("Speed")
        btn_speed.setMinimumWidth(0)
        btn_speed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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

        self.btn_adaptive_fps = QPushButton("Adaptive FPS Boost: OFF")
        self.btn_adaptive_fps.setCheckable(True)
        self.btn_adaptive_fps.setChecked(False)
        self.btn_adaptive_fps.setStyleSheet("background-color: #64748b; color: white;")
        self.btn_adaptive_fps.setToolTip(
            "When live FPS drops, the app will step down frame skip and resize\n"
            "to keep latency low and output smooth."
        )
        self.btn_adaptive_fps.clicked.connect(self.toggle_adaptive_fps)
        perf_layout.addWidget(self.btn_adaptive_fps)
        
        perf_group.setLayout(perf_layout)
        right_panel.addWidget(perf_group)
        
        # Video source
        stream_group = QGroupBox("Video Source")
        self.stream_group = stream_group
        self.stream_group = stream_group
        stream_layout = QVBoxLayout()

        source_hint = QLabel("Open a local file or paste a YouTube, RTSP, HTTP stream, or webcam ID.")
        source_hint.setObjectName("subtleInfo")
        source_hint.setWordWrap(True)
        stream_layout.addWidget(source_hint)

        self.btn_open_source_file = QPushButton("Open Video File...")
        self.btn_open_source_file.setObjectName("secondaryAction")
        self.btn_open_source_file.clicked.connect(self.load_video_file)
        stream_layout.addWidget(self.btn_open_source_file)
        
        stream_layout.addWidget(QLabel("YouTube/RTSP URL or Webcam ID:"))
        self.stream_input = QComboBox()
        self.stream_input.setEditable(True)
        self.stream_input.setPlaceholderText("https://youtube.com/... or 0 for webcam")
        self.stream_input.setMaxCount(5)  # Keep only 5 items
        self.stream_input.setInsertPolicy(QComboBox.InsertAtTop)
        self.stream_input.currentTextChanged.connect(lambda _: self.schedule_settings_save())
        
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
        self.stream_quality_combo.currentTextChanged.connect(lambda _: self.schedule_settings_save())
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
        self.video_type_combo.currentTextChanged.connect(lambda _: self.schedule_settings_save())
        self.video_type_combo.setToolTip(
            "Regular Video: Stream video thường (không tải hết)\n"
            "Livestream: Video đang phát trực tiếp"
        )
        stream_layout.addWidget(self.video_type_combo)
        
        self.btn_start_stream = QPushButton("Start Source")
        self.btn_start_stream.setObjectName("accentAction")
        self.btn_start_stream.clicked.connect(self.start_livestream)
        stream_layout.addWidget(self.btn_start_stream)
        
        stream_group.setLayout(stream_layout)
        right_panel.addWidget(stream_group)

        # Source admin hub
        source_hub_group = QGroupBox("Source Admin Hub")
        self.source_hub_group = source_hub_group
        source_hub_layout = QVBoxLayout()

        hub_hint = QLabel("Manage multiple sources. Select one source and open it in the large player.")
        hub_hint.setObjectName("subtleInfo")
        hub_hint.setWordWrap(True)
        source_hub_layout.addWidget(hub_hint)

        self.source_list_widget = QListWidget()
        self.source_list_widget.setMinimumHeight(140)
        self.source_list_widget.currentRowChanged.connect(self.on_source_profile_selected)
        source_hub_layout.addWidget(self.source_list_widget)

        hub_btn_row_1 = QHBoxLayout()
        self.btn_add_current_source = QPushButton("Add Current")
        self.btn_add_current_source.clicked.connect(self.add_current_source_profile)
        hub_btn_row_1.addWidget(self.btn_add_current_source)

        self.btn_open_selected_source = QPushButton("Open Selected")
        self.btn_open_selected_source.setObjectName("primaryAction")
        self.btn_open_selected_source.clicked.connect(self.open_selected_source_profile)
        hub_btn_row_1.addWidget(self.btn_open_selected_source)
        source_hub_layout.addLayout(hub_btn_row_1)

        hub_btn_row_2 = QHBoxLayout()
        self.btn_update_source_profile = QPushButton("Update Profile")
        self.btn_update_source_profile.clicked.connect(self.update_selected_source_profile)
        hub_btn_row_2.addWidget(self.btn_update_source_profile)

        self.btn_remove_source_profile = QPushButton("Remove")
        self.btn_remove_source_profile.setObjectName("dangerAction")
        self.btn_remove_source_profile.clicked.connect(self.remove_selected_source_profile)
        hub_btn_row_2.addWidget(self.btn_remove_source_profile)
        source_hub_layout.addLayout(hub_btn_row_2)

        self.source_profile_status = QLabel("No source profiles yet")
        self.source_profile_status.setObjectName("subtleInfo")
        source_hub_layout.addWidget(self.source_profile_status)

        source_hub_group.setLayout(source_hub_layout)
        right_panel.addWidget(source_hub_group)
        
        # ROI Settings
        roi_group = QGroupBox("ROI (Region of Interest)")
        self.roi_group = roi_group
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
        roi_button_layout.setSpacing(6)
        
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
        roi_save_layout.setSpacing(6)
        
        self.btn_save_roi = QPushButton("💾 Save")
        self.btn_save_roi.setMinimumWidth(0)
        self.btn_save_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_save_roi.clicked.connect(self.save_roi)
        roi_save_layout.addWidget(self.btn_save_roi)
        
        self.btn_load_roi = QPushButton("📂 Load")
        self.btn_load_roi.setMinimumWidth(0)
        self.btn_load_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.stats_group = stats_group
        stats_layout = QVBoxLayout()
        
        # Current model info
        self.model_info_label = QLabel("Model: Loading...")
        self.model_info_label.setObjectName("modelChip")
        self.model_info_label.setWordWrap(True)
        stats_layout.addWidget(self.model_info_label)
        
        self.stats_text = QTextEdit()
        self.stats_text.setObjectName("statsPanel")
        self.stats_text.setReadOnly(True)
        self.stats_text.setLineWrapMode(QTextEdit.NoWrap)
        self.stats_text.setMinimumHeight(240)
        self.stats_text.setMaximumHeight(320)
        self.stats_text.setPlainText(
            "SESSION TOTAL: 0\n"
            "ACTIVE NOW:    0\n\n"
            "No session data yet.\n\n"
            "Tips:\n"
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

        ordered_widgets = [
            self.stream_group,
            self.source_hub_group,
            self.runtime_shell,
            self.model_group,
            self.perf_group,
            self.detection_group,
            self.roi_group,
            self.stats_group,
            self.status_label,
        ]
        for index, widget in enumerate(ordered_widgets):
            right_panel.removeWidget(widget)
            right_panel.insertWidget(index, widget)

        compact_combos = [
            self.model_combo,
            self.tracker_combo,
            self.stream_input,
            self.stream_quality_combo,
            self.video_type_combo,
            self.resize_combo,
            self.processing_mode_combo,
        ]
        for combo in compact_combos:
            combo.setMinimumWidth(0)
            combo.setMinimumContentsLength(1)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        compact_buttons = [
            self.btn_open_source_file,
            self.btn_start_stream,
            self.btn_smooth_toggle,
            self.btn_adaptive_fps,
            self.btn_clear_roi,
            self.btn_toggle_roi,
            self.btn_toggle_sidebar,
        ]
        for button in compact_buttons:
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 3)
        
        # Wrap right panel in scroll area
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMinimumWidth(0)
        right_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(right_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setMinimumWidth(300)
        self.scroll_area.setMaximumWidth(480)
        self.scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        main_layout.addWidget(self.scroll_area, 1)
        self.apply_theme()
        self._refresh_custom_model_input()
        self._sync_processing_mode_buttons()
        self.refresh_admin_source_cards()
        self.show_admin_dashboard()

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
            QWidget#videoSurface {
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
                border-radius: 14px;
                padding: 12px 16px;
                font-size: 15px;
                line-height: 1.35;
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
                font-size: 16px;
                font-weight: 600;
                padding: 18px;
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
            QListWidget#adminSourceList {
                background: #f7f1e7;
                border: 1px solid #d7cab9;
                border-radius: 14px;
                padding: 8px;
                outline: none;
            }
            QListWidget#adminSourceList::item {
                background: #fffdf8;
                border: 1px solid #d9ccb8;
                border-radius: 12px;
                padding: 6px;
                margin: 4px;
            }
            QListWidget#adminSourceList::item:selected {
                background: #e6f4ff;
                border: 2px solid #0284c7;
                color: #0c4a6e;
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
        # Strict realtime: lock to standard mode to avoid async queue bursts.
        self.use_threaded_mode = False
        self.use_optimized_mode = False
        self.use_ultra_mode = False

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
        self.schedule_settings_save()

        current_mode = "standard"
        self.status_label.setText("Standard mode enabled (strict realtime)")

        if trigger_reload:
            if self.video_thread and self.video_thread.isRunning():
                self.pending_reload_models = True
                self.auto_reload_if_running(reason="Applying strict realtime standard mode...")
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

    def _shutdown_processor_workers(self):
        """Stop background processor workers before replacing or idling a processor."""
        if not self.processor:
            return

        try:
            if hasattr(self.processor, 'stop_async_inference'):
                self.processor.stop_async_inference()
            if hasattr(self.processor, 'stop_threads'):
                self.processor.stop_threads()
        except Exception as exc:
            print(f"[WARN] Processor worker shutdown skipped: {exc}")

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
        """Load YOLO models + tracker in background so GUI stays responsive."""
        if self.models_loading:
            self.status_label.setText("Models are already loading...")
            return

        self.status_label.setText("Loading models...")
        self._sync_processing_mode_buttons()

        try:
            self._shutdown_processor_workers()
            app_state.tracker_choice = self.tracker_combo.currentText()
            if not self._resolve_model_selection(self.model_combo.currentText(), prompt_for_custom=False):
                self.status_label.setText("Select a valid model first")
                return
            self.model_load_lowered_confidence = self._apply_recommended_confidence_for_model()
        except Exception as exc:
            self.status_label.setText(f"Error loading models: {exc}")
            return

        self.models_loading = True
        self._set_model_loading_ui(True)

        self.model_loader_thread = ModelLoadThread(self.tracker_combo.currentText())
        self.model_loader_thread.progress.connect(self._on_model_load_progress)
        self.model_loader_thread.loaded.connect(self._on_model_load_success)
        self.model_loader_thread.failed.connect(self._on_model_load_error)
        self.model_loader_thread.finished.connect(self._on_model_load_finished)
        self.model_loader_thread.start()

    def _set_model_loading_ui(self, loading: bool):
        for widget in (
            self.model_combo,
            self.tracker_combo,
            self.processing_mode_combo,
            self.btn_load_video,
            self.btn_open_source_file,
            self.btn_start_stream,
        ):
            if widget:
                widget.setEnabled(not loading)

        if loading:
            self.btn_start.setEnabled(False)

    def _on_model_load_progress(self, message: str):
        if message:
            self.status_label.setText(message)

    def _on_model_load_success(self, payload):
        try:
            self.model_person = payload.get('model_person')
            self.model_vehicle = payload.get('model_vehicle')
            self.tracker = payload.get('tracker')

            if hasattr(self.tracker, 'max_age'):
                self.tracker_max_age = int(getattr(self.tracker, 'max_age', 1))
                config.TRACKER_MAX_AGE = self.tracker_max_age
                self.tracker_age_slider.blockSignals(True)
                self.tracker_age_slider.setValue(self.tracker_max_age)
                self.tracker_age_slider.blockSignals(False)
                self.tracker_age_label.setText(
                    f"{self.tracker_max_age} frame" if self.tracker_max_age == 1 else f"{self.tracker_max_age} frames"
                )

            self._sync_tracker_combo_with_runtime()
            app_state.tracker_choice = self._get_tracker_display_name(self.tracker)

            actual_backends = []
            for model in (self.model_person, self.model_vehicle):
                backend_name = self._identify_model_backend(model)
                if backend_name not in actual_backends:
                    actual_backends.append(backend_name)

            gpu_backend_active = any("CUDA" in backend for backend in actual_backends)
            cpu_only_mode = not gpu_backend_active
            self.processor = VideoProcessor(self.model_person, self.model_vehicle, self.tracker)

            self.processor.set_confidence(self.confidence_slider.value() / 100.0)
            if hasattr(self.processor, 'set_duplicate_iou_threshold'):
                self.processor.set_duplicate_iou_threshold(self.duplicate_iou_slider.value() / 100.0)
            if hasattr(self.processor, 'set_box_thickness'):
                self.processor.set_box_thickness(self.box_thickness_slider.value())
            if hasattr(self.processor, 'set_font_size'):
                self.processor.set_font_size(self.font_size_slider.value())
            if hasattr(self.processor, 'set_font_thickness'):
                self.processor.set_font_thickness(self.font_thickness_slider.value())
            if hasattr(self.processor, 'set_point_mode'):
                self.processor.set_point_mode(self.btn_display_mode.isChecked())
            if hasattr(self.processor, 'set_frame_skip'):
                self.processor.set_frame_skip(self.frame_skip_slider.value())
            config.TRAIL_LENGTH = 3 if self.btn_trail_toggle.isChecked() else 0
            self.processor.set_roi_manager(self.roi_manager)
            self.update_model_info()

            backend_text = " + ".join(actual_backends) if actual_backends else "CPU"
            display_model_name = self._get_selected_model_display_name()
            lowered_confidence = self.model_load_lowered_confidence

            if cpu_only_mode and all("CPU" in backend for backend in actual_backends):
                status_text = f"Models loaded: {display_model_name} | Backend: CPU"
            else:
                status_text = f"Models loaded: {display_model_name} | Backend: {backend_text}"
            if lowered_confidence is not None:
                status_text = (
                    f"Models loaded: {display_model_name} | Backend: {backend_text} | "
                    f"Conf auto -> 0.{lowered_confidence:02d}"
                )
            self.status_label.setText(status_text)

            self.btn_start.setEnabled(bool(self.video_source))
            self.btn_load_video.setEnabled(True)
            self.btn_open_source_file.setEnabled(True)
            self.btn_start_stream.setEnabled(True)

            if self.pending_start_after_model_load and self.video_source:
                self.pending_start_after_model_load = False
                QTimer.singleShot(0, self.start_processing)
            else:
                self.pending_start_after_model_load = False
        except Exception as exc:
            self.status_label.setText(f"Error loading models: {exc}")

    def _on_model_load_error(self, error_text: str):
        self.status_label.setText(f"Error loading models: {error_text}")
        self.btn_start.setEnabled(False)

    def _on_model_load_finished(self):
        self.models_loading = False
        self.model_load_lowered_confidence = None
        self._set_model_loading_ui(False)
        if self.model_loader_thread:
            self.model_loader_thread.deleteLater()
            self.model_loader_thread = None

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

    def _is_live_source(self) -> bool:
        """Return True for live URL/webcam style sources."""
        if isinstance(self.video_source, int):
            return True

        if not self.video_source:
            return False

        source_text = str(self.video_source).strip()
        if source_text.isdigit():
            return True
        return source_text.startswith(("rtsp://", "http://", "https://"))

    def _current_resize_scale(self) -> int:
        """Read the active resize percentage from the combo."""
        resize_text = self.resize_combo.currentText().split()[0]
        return int(resize_text.replace('%', ''))

    def _set_resize_scale(self, resize_scale: int):
        """Apply a supported resize preset."""
        resize_map = {100: 0, 75: 1, 50: 2, 25: 3}
        index = resize_map.get(resize_scale)
        if index is not None:
            self.resize_combo.setCurrentIndex(index)

    def schedule_settings_save(self):
        """Debounce settings writes so panel interactions stay responsive."""
        if getattr(self, 'settings_loading', False):
            return
        self.settings_save_timer.start(450)

    def toggle_adaptive_fps(self, checked):
        """Enable or disable live FPS auto-tuning."""
        self.adaptive_fps = checked
        if checked:
            self.btn_adaptive_fps.setText("Adaptive FPS Boost: ON")
            self.btn_adaptive_fps.setStyleSheet("background-color: #1d4ed8; color: white;")
            self.status_label.setText("Adaptive FPS boost enabled")
        else:
            self.btn_adaptive_fps.setText("Adaptive FPS Boost: OFF")
            self.btn_adaptive_fps.setStyleSheet("background-color: #64748b; color: white;")
            self.status_label.setText("Adaptive FPS boost disabled")
        self.schedule_settings_save()

    def _maybe_adaptive_tune(self):
        """Step down live processing load when sustained FPS becomes too low."""
        if not self.adaptive_fps or not self._is_live_source():
            return

        if not (self.video_thread and self.video_thread.isRunning()):
            return

        now = time.time()
        if (now - self.last_adaptive_tune_time) < self.adaptive_tune_cooldown:
            return

        profile_ladder = [
            (0, 100),
            (0, 75),
            (1, 75),
            (2, 50),
            (3, 50),
            (4, 25),
        ]
        current_profile = (self.frame_skip_slider.value(), self._current_resize_scale())
        try:
            current_index = profile_ladder.index(current_profile)
        except ValueError:
            current_index = 0

        target_index = current_index
        if self.current_fps < 6.0:
            target_index = min(len(profile_ladder) - 1, current_index + 2)
        elif self.current_fps < 10.0:
            target_index = min(len(profile_ladder) - 1, current_index + 1)

        if target_index == current_index:
            return

        target_skip, target_resize = profile_ladder[target_index]
        self.last_adaptive_tune_time = now
        self.frame_skip_slider.setValue(target_skip)
        self._set_resize_scale(target_resize)
        self.status_label.setText(
            f"Adaptive FPS boost: frame skip {target_skip}, resize {target_resize}%"
        )

    def toggle_sidebar(self):
        """Show or hide the right-side control panel."""
        visible = not self.scroll_area.isVisible()
        self.scroll_area.setVisible(visible)
        self.btn_toggle_sidebar.setText("An Cau Hinh" if visible else "Mo Cau Hinh")
        self.schedule_settings_save()

    def toggle_runtime_section(self):
        """Collapse or expand the runtime overview card."""
        expanded = self.runtime_toggle_button.isChecked()
        self.runtime_panel.setVisible(expanded)
        self.runtime_toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.schedule_settings_save()

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
        self.schedule_settings_save()

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
            from src.inference.onnx_model import ONNXModel
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
                from src.inference.onnx_model import ONNXModel
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
        if getattr(self, 'settings_loading', False):
            # Ignore startup restore events to avoid auto-opening custom model dialog.
            self.last_model_selection = model_name
            self._refresh_custom_model_input()
            return

        try:
            if model_name == "Custom model...":
                QTimer.singleShot(0, self.browse_custom_model)
                return
            if not self._resolve_model_selection(model_name, prompt_for_custom=True):
                self.status_label.setText("Model selection canceled")
                return
            self.schedule_settings_save()
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
        self.schedule_settings_save()
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
        self.schedule_settings_save()

    def on_duplicate_iou_changed(self, value):
        """Handle duplicate box IOU threshold change."""
        iou_threshold = value / 100.0
        self.duplicate_iou_label.setText(f"{iou_threshold:.2f}")
        if self.processor and hasattr(self.processor, 'set_duplicate_iou_threshold'):
            self.processor.set_duplicate_iou_threshold(iou_threshold)
        self.schedule_settings_save()
    
    def on_box_thickness_changed(self, value):
        """Handle box thickness change"""
        self.box_thickness_label.setText(f"{value} px")
        if self.processor:
            self.processor.set_box_thickness(value)
        self.schedule_settings_save()
    
    def on_font_size_changed(self, value):
        """Handle font size change"""
        self.font_size_label.setText(f"{value} pt")
        if self.processor:
            self.processor.set_font_size(value)
        self.schedule_settings_save()
    
    def on_font_thickness_changed(self, value):
        """Handle font thickness change"""
        self.font_thickness_label.setText(f"{value}")
        if self.processor and hasattr(self.processor, 'set_font_thickness'):
            self.processor.set_font_thickness(value)
        self.schedule_settings_save()
    
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
        self.schedule_settings_save()
    
    def toggle_trail(self, checked):
        """Toggle trail drawing"""
        from src.core import config
        if checked:
            config.TRAIL_LENGTH = 3  # Enable trail
            self.btn_trail_toggle.setText("🟢 Trail: ON")
            self.btn_trail_toggle.setStyleSheet("background-color: #15803d; color: white;")
        else:
            config.TRAIL_LENGTH = 0  # Disable trail
            self.btn_trail_toggle.setText("🔴 Trail: OFF")
            self.btn_trail_toggle.setStyleSheet("background-color: #b42318; color: white;")
        self.schedule_settings_save()
    
    def manual_cleanup(self):
        """Manual memory cleanup"""
        self.status_label.setText("Cleaning cache...")
        QApplication.processEvents()
        
        try:
            # Clear processor cache
            if self.processor:
                _reset_processor_tracking_state(self.processor)
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
        # Ignore user selection changes while strict realtime lock is active.
        self._apply_processing_mode("standard")

    def on_max_det_changed(self, value):
        """Handle max detections change"""
        self.max_det = value
        self.max_det_label.setText(f"{value} objects")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.max_det = value
        self.schedule_settings_save()
    
    def on_tracker_age_changed(self, value):
        """Handle tracker max age change"""
        self.tracker_max_age = value
        self.tracker_age_label.setText(f"{value} frames")
        # Update config
        from src.core import config
        config.TRACKER_MAX_AGE = value
        # Update tracker if exists
        if self.tracker:
            if hasattr(self.tracker, 'max_age'):
                self.tracker.max_age = value
        self.schedule_settings_save()

    def on_frame_skip_changed(self, value):
        """Apply frame skip live without restarting the stream."""
        self.frame_skip_label.setText(f"Skip: {value} frames")
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.frame_skip = value
        if self.processor and hasattr(self.processor, 'set_frame_skip'):
            self.processor.set_frame_skip(value)
        self.schedule_settings_save()

    def on_resize_changed(self, text):
        """Apply resize scale live when possible."""
        resize_scale = int(text.split()[0].replace('%', ''))
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.resize_scale = resize_scale
        self.schedule_settings_save()

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
            self.pending_start_after_model_load = bool(self.video_source)
            self.load_models()
            return
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
        self.schedule_settings_save()
    
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
                self.schedule_settings_save()
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
        
        # Get actual frame size (from current_frame_size, not pixmap)
        frame_width, frame_height = self.current_frame_size

        if hasattr(self.video_label, 'content_rect'):
            draw_rect = self.video_label.content_rect((frame_width, frame_height))
            offset_x = draw_rect.x()
            offset_y = draw_rect.y()
            display_width = draw_rect.width()
            display_height = draw_rect.height()
            if display_width <= 0 or display_height <= 0:
                return
            scale = display_width / frame_width
        else:
            # QLabel fallback path
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            scale_w = label_width / frame_width
            scale_h = label_height / frame_height
            scale = min(scale_w, scale_h)
            display_width = int(frame_width * scale)
            display_height = int(frame_height * scale)
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
        self.schedule_settings_save()
    
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
        self.schedule_settings_save()
    
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
        self.schedule_settings_save()
    
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
                self.schedule_settings_save()
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
        if hasattr(self, 'main_view_stack') and self.main_view_stack.currentIndex() != 0:
            self.status_label.setText("Open Source Page to add new sources")
            self.show_admin_dashboard()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        
        if file_path:
            self._cancel_pending_stream_resolve()
            self.video_source = file_path
            self._upsert_source_profile(file_path, source_type="file", auto_select=True)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.btn_start.setEnabled(True)
            self._refresh_runtime_overview()
            self.schedule_settings_save()
            
    def start_livestream(self):
        """Start livestream processing"""
        if hasattr(self, 'main_view_stack') and self.main_view_stack.currentIndex() != 0:
            self.status_label.setText("Open Source Page to add new sources")
            self.show_admin_dashboard()
            return

        stream_url = self.stream_input.currentText().strip()
        
        if not stream_url:
            self.status_label.setText("⚠️ Please enter stream URL or webcam ID")
            return
        
        # Save to history
        self.save_stream_to_history(stream_url)
        self.schedule_settings_save()
            
        # Check if webcam ID
        if stream_url.isdigit():
            self._cancel_pending_stream_resolve()
            self.video_source = int(stream_url)
            self._upsert_source_profile(self.video_source, source_type="camera", auto_select=True)
        else:
            # Handle YouTube URL  
            if 'youtube.com' in stream_url or 'youtu.be' in stream_url:
                self._upsert_source_profile(stream_url, source_type="youtube", auto_select=True)
                self._start_youtube_resolve(stream_url)
                return
            else:
                self._cancel_pending_stream_resolve()
                self.video_source = stream_url
                source_type = "rtsp" if stream_url.lower().startswith("rtsp://") else "stream"
                self._upsert_source_profile(self.video_source, source_type=source_type, auto_select=True)
                self._refresh_runtime_overview()
                
        self.start_processing()

    def _cancel_pending_stream_resolve(self):
        """Ignore stale async YouTube resolve callbacks after source changed."""
        self.pending_youtube_source = None

    def _on_stream_resolve_progress(self, message: str):
        if not self.pending_youtube_source:
            return
        if message:
            self.status_label.setText(message)

    def _on_stream_resolved(self, resolved_source, status_text: str):
        if not self.pending_youtube_source:
            return
        requested_source = self.pending_youtube_source
        self.video_source = resolved_source
        self._upsert_source_profile(requested_source, source_type="youtube", auto_select=True)
        self.status_label.setText(status_text)
        self._refresh_runtime_overview()
        self.start_processing()

    def _on_stream_resolve_failed(self, error_text: str):
        if not self.pending_youtube_source:
            return
        self.status_label.setText(f"❌ Error: {error_text[:120]}")
        print(f"Full error: {error_text}")

    def _on_stream_resolve_finished(self):
        self.pending_youtube_source = None
        is_processing = bool(self.video_thread and self.video_thread.isRunning())
        self.btn_start_stream.setEnabled((not self.models_loading) and (not is_processing))
        self.btn_start.setEnabled((not self.models_loading) and bool(self.video_source) and bool(self.processor) and (not is_processing))
        if self.stream_resolver_thread:
            self.stream_resolver_thread.deleteLater()
            self.stream_resolver_thread = None

    def _start_youtube_resolve(self, youtube_url: str):
        if self.stream_resolver_thread and self.stream_resolver_thread.isRunning():
            self.status_label.setText("YouTube source is already being resolved...")
            return False

        self.pending_youtube_source = youtube_url
        self.btn_start_stream.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.status_label.setText("🔄 Checking YouTube URL...")
        self.stream_resolver_thread = StreamResolveThread(
            youtube_url,
            self.stream_quality_combo.currentText()
        )
        self.stream_resolver_thread.progress.connect(self._on_stream_resolve_progress)
        self.stream_resolver_thread.resolved.connect(self._on_stream_resolved)
        self.stream_resolver_thread.failed.connect(self._on_stream_resolve_failed)
        self.stream_resolver_thread.finished.connect(self._on_stream_resolve_finished)
        self.stream_resolver_thread.start()
        return True

    def _reset_runtime_frame_state(self, clear_widget=False, widget_message=None):
        """Reset frame handoff state so old-source frames cannot leak into new source."""
        self.shared_frame_store = SharedFrameStore()
        self.pending_frame_version = 0
        self.metrics_last_version = 0
        self.pending_display_frame = None
        self.pending_display_stats = {'total_objects': 0, 'class_counts': {}}
        self.pending_frame_dirty = False
        self.last_preview_capture_ts = 0.0
        if clear_widget and hasattr(self, 'video_label'):
            if hasattr(self.video_label, 'clear_frame'):
                self.video_label.clear_frame()
            if widget_message is not None and hasattr(self.video_label, 'setText'):
                self.video_label.setText(widget_message)
        
    def start_processing(self):
        """Start video processing"""
        if self.models_loading:
            self.pending_start_after_model_load = True
            self.status_label.setText("Models are loading... stream will start automatically.")
            return
        if not self.processor:
            self.status_label.setText("⚠️ Models not loaded")
            return
        if not self.video_source:
            self.status_label.setText("Select a video file, stream URL, or camera first")
            return

        if isinstance(self.video_source, str):
            source_text = self.video_source.strip().lower()
            if ('youtube.com' in source_text) or ('youtu.be' in source_text):
                self._start_youtube_resolve(self.video_source)
                return
        self._cancel_pending_stream_resolve()

        self.show_player_view()

        source_text = str(self.video_source).strip().lower()
        is_live_source = (
            source_text.startswith(("rtsp://", "http://", "https://"))
            or source_text.isdigit()
            or isinstance(self.video_source, int)
        )

        # Respect user tracker choice (including DeepSORT) on all sources.

        # Auto high-FPS preset for live sources (without reducing resolution).
        if is_live_source:
            if self.max_det_slider.value() > 12:
                self.max_det_slider.blockSignals(True)
                self.max_det_slider.setValue(12)
                self.max_det_slider.blockSignals(False)
                self.max_det = 12
                self.max_det_label.setText("12 objects")
            
        # Stop existing thread if running
        if self.video_thread and self.video_thread.isRunning():
            self.stop_processing(clear_display=True)
            
        # Get parameters
        frame_skip = 0
        if self.frame_skip_slider.value() != 0:
            self.frame_skip_slider.blockSignals(True)
            self.frame_skip_slider.setValue(0)
            self.frame_skip_slider.blockSignals(False)
            self.frame_skip_label.setText("Skip: 0 frames")
        resize_text = self.resize_combo.currentText().split()[0]  # Get "100%" from "100% (Full)"
        resize_scale = int(resize_text.replace('%', ''))
        
        # Reset processor statistics
        self.processor.reset_statistics()
        self._reset_runtime_frame_state(clear_widget=True)
        
        # Create and start thread
        self.video_thread = VideoThread()
        self.video_thread.set_source(self.video_source)
        self.video_thread.set_processor(self.processor)
        self.video_thread.set_params(frame_skip, resize_scale, self.display_target_fps)
        self.video_thread.set_shared_frame_store(self.shared_frame_store)
        self.video_thread.max_det = self.max_det  # Pass max_det
        self.video_thread.smooth_mode = self.smooth_mode
        self.video_thread.finished.connect(self.on_processing_finished)
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_open_source_file.setEnabled(False)
        self.btn_start_stream.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Reset FPS counter
        self.fps_counter = 0
        self.fps_start_time = time.perf_counter()
        self.render_fps_counter = 0
        self.render_fps_start_time = time.perf_counter()
        self.current_fps = 0.0
        self.current_detect_fps = 0.0
        self.current_render_fps = 0.0
        self.fps_history = []  # Clear FPS history
        self.last_display_time = 0.0
        self.last_adaptive_tune_time = 0.0
        self.last_fps_color = ""
        self.last_stats_text = ""
        self.video_thread.start()
        self.render_timer.start()
        self._update_fps_summary(self.pending_display_stats)
        self._update_stats_panel(self.pending_display_stats)
        self._update_vehicle_alert_ui(self.pending_display_stats)
        self._refresh_runtime_overview()
        
    def stop_processing(self, preserve_pending=False, clear_display=False):
        """Stop video processing"""
        if not preserve_pending:
            self.pending_restart = False
            self.pending_reload_models = False
            self.pending_start_after_model_load = False
            self.restart_timer.stop()
        if clear_display:
            self.render_timer.stop()
            self._reset_runtime_frame_state(clear_widget=True, widget_message="Dang chuyen nguon...")
        if self.video_thread:
            self.video_thread.stop()
            if self.video_thread.isRunning():
                self.status_label.setText("Stopping stream...")
            while self.video_thread.isRunning():
                self.video_thread.wait(30)
                QApplication.processEvents()
        self._shutdown_processor_workers()
            
    def on_processing_finished(self):
        """Handle processing finished"""
        if self.pending_restart:
            self.pending_restart = False
            reload_models = self.pending_reload_models
            self.pending_reload_models = False
            self.restart_message = ""
            if reload_models:
                self.pending_start_after_model_load = bool(self.video_source)
                self.load_models()
                return
            self.start_processing()
            return

        self.btn_start.setEnabled((not self.models_loading) and bool(self.video_source) and bool(self.processor))
        self.btn_stop.setEnabled(False)
        self.btn_load_video.setEnabled(not self.models_loading)
        self.btn_open_source_file.setEnabled(not self.models_loading)
        self.btn_start_stream.setEnabled(not self.models_loading)
        self.status_label.setText("Stopped")
        self.render_timer.stop()
        self.pending_frame_version = 0
        self.metrics_last_version = 0
        self._update_vehicle_alert_ui({'active_class_counts': {}, 'class_counts': {}})
        self._refresh_runtime_overview()

    def _smooth_fps_value(self, current_value, sample_value):
        """Blend FPS samples so short spikes do not thrash the UI."""
        if sample_value <= 0:
            return current_value
        if current_value <= 0:
            return sample_value
        return current_value + (sample_value - current_value) * self.fps_smoothing

    def _format_count_summary(self, counts, empty_text="none"):
        """Render class counts on a single line for the compact FPS card."""
        if not counts:
            return empty_text
        return " | ".join(f"{cls_name}:{count}" for cls_name, count in sorted(counts.items()))

    def _format_count_block(self, title, counts):
        """Render aligned count rows for the detailed stats panel."""
        if not counts:
            return ""

        name_width = max(12, max(len(cls_name) for cls_name in counts))
        lines = [title]
        for cls_name, count in sorted(counts.items()):
            lines.append(f"  {cls_name:<{name_width}} {count:>4}")
        return "\n".join(lines)

    def _setup_status_panel(self, parent_layout):
        """Realtime status panel with a single-row 8-card layout."""
        panel = QFrame()
        self.status_panel = panel
        panel.setObjectName("statusPanel")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        panel.setMinimumHeight(120)
        panel.setMaximumHeight(210)
        panel.setStyleSheet(
            "QFrame#statusPanel {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8fafc, stop:1 #f1f5f9);"
            "  border: 2px solid #cbd5e1;"
            "  border-radius: 16px;"
            "}"
        )

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 8, 12, 8)
        panel_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        title = QLabel("📊 Realtime Status")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #1e293b;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.status_level_badge = QLabel("AN TOAN")
        self.status_level_badge.setAlignment(Qt.AlignCenter)
        self.status_level_badge.setMinimumWidth(220)
        self.status_level_badge.setStyleSheet(
            "background: #22c55e;"
            "color: #ffffff;"
            "border-radius: 12px;"
            "padding: 8px 16px;"
            "font-weight: 700;"
            "font-size: 12px;"
            "letter-spacing: 0.5px;"
        )
        header_row.addWidget(self.status_level_badge)
        panel_layout.addLayout(header_row)

        metrics_container = QFrame()
        metrics_container.setStyleSheet(
            "background: #ffffff;"
            "border: 1px solid #e2e8f0;"
            "border-radius: 10px;"
            "padding: 6px;"
        )
        metrics_layout = QGridLayout(metrics_container)
        metrics_layout.setContentsMargins(4, 4, 4, 4)
        metrics_layout.setHorizontalSpacing(10)
        metrics_layout.setVerticalSpacing(6)

        def add_metric_card(title_text, value_attr, sub_attr, value_text, sub_text, row, col, value_color="#0f172a"):
            card = QFrame()
            card.setStyleSheet(
                "background: #f8fafc;"
                "border: 1px solid #d6deea;"
                "border-radius: 10px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 4, 6, 4)
            card_layout.setSpacing(2)

            title_label = QLabel(title_text)
            title_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
            title_label.setStyleSheet("color: #64748b; border: none;")
            title_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(title_label)

            value_label = QLabel(value_text)
            value_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
            value_label.setStyleSheet(f"color: {value_color}; border: none;")
            value_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(value_label)

            sub_label = QLabel(sub_text)
            sub_label.setFont(QFont("Segoe UI", 8, QFont.Normal))
            sub_label.setStyleSheet("color: #64748b; border: none;")
            sub_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(sub_label)

            setattr(self, value_attr, value_label)
            setattr(self, sub_attr, sub_label)
            metrics_layout.addWidget(card, row, col)

        add_metric_card("⚡ FPS", "status_main_fps", "status_fps_subtitle", "0.0", "Display", 0, 0, "#22c55e")
        add_metric_card("🚶 Ped", "status_pedestrian_now", "status_pedestrian_total", "0", "Total: 0", 0, 1)
        add_metric_card("🚲 Bike", "status_bicycle_now", "status_bicycle_total", "0", "Total: 0", 0, 2)
        add_metric_card("🏍️ Moto", "status_moto_now", "status_moto_total", "0", "Total: 0", 0, 3)
        add_metric_card("🚗 Car", "status_car_now", "status_car_total", "0", "Total: 0", 0, 4)
        add_metric_card("🚚 Truck", "status_truck_now", "status_truck_total", "0", "Total: 0", 0, 5)
        add_metric_card("🚌 Bus", "status_bus_now", "status_bus_total", "0", "Total: 0", 0, 6)
        add_metric_card("📦 Cont", "status_container_now", "status_container_total", "0", "Total: 0", 0, 7)
        add_metric_card("🔴 Now", "status_total_current_now", "status_total_current_sub", "0", "All class", 0, 8, "#dc2626")
        add_metric_card("📊 Total", "status_total_session_now", "status_total_session_sub", "0", "Session", 0, 9, "#1d4ed8")

        for col in range(10):
            metrics_layout.setColumnStretch(col, 1)
        metrics_layout.setRowStretch(0, 1)

        panel_layout.addWidget(metrics_container)
        parent_layout.addWidget(panel)

    def _normalize_vehicle_counts(self, counts):
        """Normalize all 7 object classes for legacy compact grid."""
        normalized = {
            "pedestrian": 0,
            "bicycle": 0,
            "motorbike": 0,
            "car": 0,
            "truck": 0,
            "container truck": 0,
            "bus": 0,
        }
        if not counts:
            return normalized
        for cls_name, count in counts.items():
            key = str(cls_name).strip().lower()
            if key in ["pedestrian", "person", "people"]:
                normalized["pedestrian"] += int(count)
            elif key in ["bicycle", "bike", "cycle"]:
                normalized["bicycle"] += int(count)
            elif key in ["motorbike", "motorcycle", "moto"]:
                normalized["motorbike"] += int(count)
            elif key in ["car", "sedan"]:
                normalized["car"] += int(count)
            elif key in ["truck"]:
                normalized["truck"] += int(count)
            elif key in ["container truck", "container", "semi"]:
                normalized["container truck"] += int(count)
            elif key in ["bus"]:
                normalized["bus"] += int(count)
        return normalized

    def _format_vehicle_counts(self, counts):
        order = [
            ("car", "Car"),
            ("truck", "Truck"),
            ("bus", "Bus"),
            ("motorcycle", "Moto"),
        ]
        parts = [f"{label}: {counts.get(key, 0)}" for key, label in order]
        return " | ".join(parts)

    def _set_fps_badge_color(self, fps_value):
        """Legacy helper kept for compatibility."""
        if fps_value >= 20:
            color = '#22c55e'
        elif fps_value >= 10:
            color = '#f59e0b'
        else:
            color = '#ef4444'

        if color == self.last_fps_color:
            return

        if hasattr(self, 'status_main_fps'):
            self.status_main_fps.setStyleSheet(
                "color: {color};"
                "font-weight: 700;"
                .format(color=color)
            )
        self.last_fps_color = color

    def _update_fps_summary(self, stats):
        """Update FPS in single compact column (Display FPS)."""
        main_fps = self.current_render_fps
        if hasattr(self, 'status_main_fps'):
            self.status_main_fps.setText(f"{main_fps:.1f}")
            if main_fps >= 20:
                color = '#22c55e'
            elif main_fps >= 10:
                color = '#f59e0b'
            else:
                color = '#ef4444'
            self.status_main_fps.setStyleSheet(f"color: {color}; font-weight: 700;")

    def _extract_vehicle_count(self, counts):
        """Return active vehicle count from class-count payload."""
        normalized = self._normalize_vehicle_counts(counts)
        return sum(normalized.values())

    def _update_vehicle_alert_ui(self, stats):
        """Update compact legacy grid + alert badge + video border color."""
        active_counts = stats.get('active_class_counts', {}) if isinstance(stats, dict) else {}
        total_counts = stats.get('class_counts', {}) if isinstance(stats, dict) else {}
        active_all_counts = self._normalize_vehicle_counts(active_counts)
        total_all_counts = self._normalize_vehicle_counts(total_counts)

        roi_active = False
        try:
            roi_active = bool(self.roi_manager and self.roi_manager.is_active())
        except Exception:
            roi_active = False

        if roi_active:
            current_vehicles = sum(active_all_counts.values())
        else:
            current_vehicles = sum(total_all_counts.values())

        limit = max(1, int(self.vehicle_alert_limit))
        ratio = current_vehicles / float(limit)

        if ratio > 1.0:
            color = "#ef4444"
            message = "VUOT NGUONG"
        elif ratio >= self.alert_yellow_ratio:
            color = "#f59e0b"
            message = "GAN NGUONG"
        else:
            color = "#22c55e"
            message = "AN TOAN"

        ratio_percent = min(999, int(ratio * 100))
        if hasattr(self, 'status_level_badge'):
            badge_text = f"{message} | {current_vehicles}/{limit} ({ratio_percent}%)"
            self.status_level_badge.setText(badge_text)
            self.status_level_badge.setStyleSheet(
                f"background: {color};"
                "color: #ffffff;"
                "border-radius: 12px;"
                "padding: 8px 16px;"
                "font-weight: 700;"
                "font-size: 12px;"
                "letter-spacing: 0.5px;"
            )

        if hasattr(self, 'status_pedestrian_now'):
            self.status_pedestrian_now.setText(str(active_all_counts.get('pedestrian', 0)))
        if hasattr(self, 'status_pedestrian_total'):
            self.status_pedestrian_total.setText(f"Total: {total_all_counts.get('pedestrian', 0)}")

        if hasattr(self, 'status_bicycle_now'):
            self.status_bicycle_now.setText(str(active_all_counts.get('bicycle', 0)))
        if hasattr(self, 'status_bicycle_total'):
            self.status_bicycle_total.setText(f"Total: {total_all_counts.get('bicycle', 0)}")

        if hasattr(self, 'status_moto_now'):
            self.status_moto_now.setText(str(active_all_counts.get('motorbike', 0)))
        if hasattr(self, 'status_moto_total'):
            self.status_moto_total.setText(f"Total: {total_all_counts.get('motorbike', 0)}")

        if hasattr(self, 'status_car_now'):
            self.status_car_now.setText(str(active_all_counts.get('car', 0)))
        if hasattr(self, 'status_car_total'):
            self.status_car_total.setText(f"Total: {total_all_counts.get('car', 0)}")

        if hasattr(self, 'status_truck_now'):
            self.status_truck_now.setText(str(active_all_counts.get('truck', 0)))
        if hasattr(self, 'status_truck_total'):
            self.status_truck_total.setText(f"Total: {total_all_counts.get('truck', 0)}")

        if hasattr(self, 'status_container_now'):
            self.status_container_now.setText(str(active_all_counts.get('container truck', 0)))
        if hasattr(self, 'status_container_total'):
            self.status_container_total.setText(f"Total: {total_all_counts.get('container truck', 0)}")

        if hasattr(self, 'status_bus_now'):
            self.status_bus_now.setText(str(active_all_counts.get('bus', 0)))
        if hasattr(self, 'status_bus_total'):
            self.status_bus_total.setText(f"Total: {total_all_counts.get('bus', 0)}")

        total_now = sum(active_all_counts.values())
        total_session = sum(total_all_counts.values())
        if hasattr(self, 'status_total_current_now'):
            self.status_total_current_now.setText(str(total_now))
        if hasattr(self, 'status_total_session_now'):
            self.status_total_session_now.setText(str(total_session))

        if hasattr(self, 'video_label'):
            self.video_label.setStyleSheet(
                f"border: 4px solid {color};"
                "border-radius: 10px;"
                "background: #081018;"
            )
        self.last_alert_color = color

    def _update_stats_panel(self, stats):
        """Render the detailed stats panel with larger, aligned text."""
        active_objects = stats.get('active_objects', stats.get('total_objects', 0))
        total_objects = stats.get('total_objects', 0)
        active_class_counts = stats.get('active_class_counts', stats.get('class_counts', {}))
        total_class_counts = stats.get('class_counts', {})

        lines = [
            f"SESSION TOTAL: {total_objects}",
            f"ACTIVE NOW:    {active_objects}",
            "",
        ]

        session_block = self._format_count_block("Session Total", total_class_counts)
        active_block = self._format_count_block("Now", active_class_counts)

        if session_block:
            lines.append(session_block)
            lines.append("")

        if active_block:
            lines.append(active_block)
        elif total_objects <= 0 and active_objects <= 0:
            lines.extend([
                "No active detections.",
                "",
                "Tips:",
                "- Use Ultra mode for livestreams",
                "- Lower resize to 75% or 50%",
                "- Keep tracker on Simple or SORT for higher FPS",
            ])

        stats_text = "\n".join(lines).strip()
        if stats_text != self.last_stats_text:
            self.stats_text.setPlainText(stats_text)
            self.last_stats_text = stats_text

    def on_source_grid_columns_changed(self, text):
        try:
            columns = int(str(text).strip())
        except Exception:
            columns = 5
        self.source_grid_columns = max(2, min(7, columns))
        self._apply_admin_source_grid_columns(refresh_cards=True)
        self.schedule_settings_save()

    def on_source_text_size_changed(self, text):
        allowed = {"Small", "Medium", "Large"}
        value = str(text).strip()
        self.source_tile_text_size = value if value in allowed else "Medium"
        self._apply_admin_source_grid_columns(refresh_cards=True)
        self.schedule_settings_save()

    def on_pin_active_source_changed(self, checked):
        self.pin_active_source_first = bool(checked)
        self.refresh_admin_source_cards()
        self.schedule_settings_save()

    def _apply_admin_source_grid_columns(self, refresh_cards=False):
        if not hasattr(self, 'admin_source_list'):
            return

        columns = max(2, min(7, int(getattr(self, 'source_grid_columns', 5))))
        list_width = max(320, self.admin_source_list.viewport().width())
        usable_width = max(280, list_width - 8)

        min_spacing = 6
        max_spacing = 20
        spacing = 10
        card_w = max(170, (usable_width - spacing * (columns + 1)) // columns)

        if card_w < 170:
            spacing = min_spacing
            card_w = max(160, (usable_width - spacing * (columns + 1)) // columns)

        card_w = min(440, card_w)
        distributed = usable_width - (card_w * columns)
        spacing = max(min_spacing, min(max_spacing, distributed // (columns + 1)))
        self.admin_source_list.setSpacing(spacing)

        preview_w = max(156, card_w - 18)
        preview_h = max(88, int(preview_w * 9 / 16))
        text_size_map = {"Small": 9, "Medium": 10, "Large": 11}
        font_pt = text_size_map.get(self.source_tile_text_size, 10)
        list_font = self.admin_source_list.font()
        list_font.setPointSize(font_pt)
        self.admin_source_list.setFont(list_font)

        label_block_h = {"Small": 48, "Medium": 54, "Large": 62}.get(self.source_tile_text_size, 54)
        card_h = preview_h + label_block_h

        size_changed = (preview_w != self.preview_width) or (preview_h != self.preview_height)
        self.preview_width = preview_w
        self.preview_height = preview_h
        self.admin_source_list.setIconSize(QSize(preview_w, preview_h))
        self.admin_source_list.setGridSize(QSize(card_w, card_h))

        if refresh_cards:
            if size_changed:
                self.source_preview_cache.clear()
            self.refresh_admin_source_cards()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'admin_source_list'):
            self._apply_admin_source_grid_columns(refresh_cards=False)
        
    def update_frame(self, frame_version):
        """Receive the latest processed frame without doing heavy GUI work."""
        if frame_version <= 0:
            return

        version, frame, stats = self.shared_frame_store.read()
        if frame is None or version <= 0:
            return
        if version < self.pending_frame_version:
            return

        self.pending_frame_version = version
        self.pending_display_frame = frame
        self.pending_display_stats = stats
        self.pending_frame_dirty = True

    def _render_latest_frame(self):
        """Render the newest frame at a stable UI cadence."""
        self.render_fps_counter += 1
        version, latest_frame, latest_stats = self.shared_frame_store.read()
        if version > self.pending_frame_version and latest_frame is not None:
            self.pending_frame_version = version
            self.pending_display_frame = latest_frame
            self.pending_display_stats = latest_stats
            self.pending_frame_dirty = True

        if version > self.metrics_last_version:
            self.fps_counter += (version - self.metrics_last_version)
            self.metrics_last_version = version

        elapsed = time.perf_counter() - self.fps_start_time
        if elapsed >= self.fps_sample_interval:
            now = time.perf_counter()
            self.current_detect_fps = self.fps_counter / elapsed
            self.current_fps = self._smooth_fps_value(self.current_fps, self.current_detect_fps)

            if self.current_fps >= 5.0:
                self.fps_history.append(self.current_fps)
                if len(self.fps_history) > self.fps_history_max:
                    self.fps_history.pop(0)

            if self.adaptive_fps:
                self._maybe_adaptive_tune()
            self._update_fps_summary(self.pending_display_stats)
            self._update_stats_panel(self.pending_display_stats)
            self._update_vehicle_alert_ui(self.pending_display_stats)

            self.fps_counter = 0
            self.fps_start_time = time.perf_counter()

            if (now - self.last_runtime_refresh_time) >= self.runtime_refresh_interval:
                self._refresh_runtime_overview()
                self.last_runtime_refresh_time = now

        if self.pending_display_frame is None:
            return

        if not self.pending_frame_dirty and not self.roi_drawing_mode:
            return

        frame = self.pending_display_frame
        self.pending_frame_dirty = False

        original_h, original_w = frame.shape[:2]
        self.current_frame_size = (original_w, original_h)

        needs_overlay_copy = (
            (self.roi_manager and self.roi_manager.is_active())
            or (self.roi_drawing_mode and len(self.roi_temp_points) > 0)
        )
        if needs_overlay_copy:
            frame = frame.copy()

        if self.roi_manager and self.roi_manager.is_active():
            self.roi_manager.draw_roi(frame)

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

        if hasattr(self.video_label, 'set_frame'):
            self.video_label.set_frame(frame)
        else:
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

        self._capture_runtime_preview_for_active_source(frame)

        render_elapsed = time.perf_counter() - self.render_fps_start_time
        if render_elapsed >= self.fps_sample_interval:
            render_sample = self.render_fps_counter / render_elapsed
            self.current_render_fps = self._smooth_fps_value(self.current_render_fps, render_sample)
            self.render_fps_counter = 0
            self.render_fps_start_time = time.perf_counter()
            self._update_fps_summary(self.pending_display_stats)
            self._update_vehicle_alert_ui(self.pending_display_stats)

    def on_vehicle_alert_scope_changed(self, text):
        # Scope is now automatic: ROI when active, otherwise Full Frame.
        self.vehicle_alert_scope = "AUTO"
        self.schedule_settings_save()

    def on_vehicle_alert_limit_changed(self, value):
        self.vehicle_alert_limit = max(1, int(value))
        self.vehicle_alert_limit_label.setText(f"{self.vehicle_alert_limit} vehicles")
        self.schedule_settings_save()

    def closeEvent(self, event):
        """Handle window close"""
        # Save settings before closing
        self.save_settings()
        
        if self.video_thread and self.video_thread.isRunning():
            self.stop_processing()
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.wait(500)
        if self.stream_resolver_thread and self.stream_resolver_thread.isRunning():
            self.stream_resolver_thread.wait(500)
        event.accept()
    
    def save_settings(self):
        """Save all settings to file"""
        last_video_file_path = None
        if isinstance(self.video_source, str):
            source_text = self.video_source.strip()
            if source_text and not source_text.startswith(("rtsp://", "http://", "https://")) and Path(source_text).exists():
                last_video_file_path = source_text

        settings = {
            # Model settings
            'model_choice': self.model_combo.currentText(),
            'custom_model_path': self.custom_model_path,
            'tracker_choice': self.tracker_combo.currentText(),
            
            # Detection settings
            'confidence': self.confidence_slider.value(),
            'duplicate_iou_threshold': self.duplicate_iou_slider.value(),
            'box_thickness': self.box_thickness_slider.value(),
            'font_size': self.font_size_slider.value(),
            'font_thickness': self.font_thickness_slider.value(),
            'display_point_mode': self.btn_display_mode.isChecked(),
            'trail_enabled': self.btn_trail_toggle.isChecked(),
            'max_det': self.max_det_slider.value(),
            'tracker_age': self.tracker_age_slider.value(),
            
            # Performance settings
            'frame_skip': self.frame_skip_slider.value(),
            'resize_scale': self.resize_combo.currentText(),
            'smooth_mode': self.smooth_mode,
            'processing_mode': self._get_processing_mode(),
            'adaptive_fps': self.adaptive_fps,
            
            # Stream settings
            'stream_input': self.stream_input.currentText().strip(),
            'stream_quality': self.stream_quality_combo.currentText(),
            'video_type': self.video_type_combo.currentText(),
            'last_video_file_path': last_video_file_path,

            # Source admin hub
            'source_profiles': self.source_profiles,
            'source_profile_counter': self.source_profile_counter,
            'active_source_profile_id': self.active_source_profile_id,
            'source_grid_columns': self.source_grid_columns,
            'source_tile_text_size': self.source_tile_text_size,
            'pin_active_source_first': self.pin_active_source_first,
            
            # ROI settings
            'roi_threshold': self.roi_threshold_slider.value(),
            'roi_visible': self.roi_visible,
            'roi_config': self.roi_manager.get_config() if self.roi_manager.is_active() else None,
            'vehicle_alert_limit': self.vehicle_alert_limit,
            'vehicle_alert_scope': 'AUTO',

            # Panel state
            'sidebar_visible': self.scroll_area.isVisible(),
            'runtime_expanded': self.runtime_toggle_button.isChecked(),
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
            if 'duplicate_iou_threshold' in settings and hasattr(self, 'duplicate_iou_slider'):
                self.duplicate_iou_slider.setValue(settings['duplicate_iou_threshold'])
            if 'box_thickness' in settings:
                self.box_thickness_slider.setValue(settings['box_thickness'])
            if 'font_size' in settings:
                self.font_size_slider.setValue(settings['font_size'])
            if 'font_thickness' in settings:
                self.font_thickness_slider.setValue(settings['font_thickness'])
            if 'display_point_mode' in settings:
                checked = bool(settings['display_point_mode'])
                self.btn_display_mode.blockSignals(True)
                self.btn_display_mode.setChecked(checked)
                self.btn_display_mode.blockSignals(False)
                self.toggle_display_mode(checked)
            if 'trail_enabled' in settings:
                checked = bool(settings['trail_enabled'])
                self.btn_trail_toggle.blockSignals(True)
                self.btn_trail_toggle.setChecked(checked)
                self.btn_trail_toggle.blockSignals(False)
                self.toggle_trail(checked)
            if 'max_det' in settings:
                self.max_det_slider.setValue(settings['max_det'])
            if 'tracker_age' in settings:
                self.tracker_age_slider.setValue(settings['tracker_age'])
            
            # Performance settings
            if 'frame_skip' in settings:
                self.frame_skip_slider.setValue(0)
            
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
            if 'adaptive_fps' in settings and hasattr(self, 'btn_adaptive_fps'):
                # Force default OFF for strict real-time stability.
                checked = False
                self.btn_adaptive_fps.blockSignals(True)
                self.btn_adaptive_fps.setChecked(checked)
                self.btn_adaptive_fps.blockSignals(False)
                self.toggle_adaptive_fps(checked)

            # Mutually-exclusive processing mode
            self._apply_processing_mode('standard', trigger_reload=False)
            
            # Stream settings
            if 'stream_input' in settings and settings['stream_input']:
                self.stream_input.setEditText(settings['stream_input'])
            if 'stream_quality' in settings:
                index = self.stream_quality_combo.findText(settings['stream_quality'])
                if index >= 0:
                    self.stream_quality_combo.setCurrentIndex(index)
            
            if 'video_type' in settings:
                index = self.video_type_combo.findText(settings['video_type'])
                if index >= 0:
                    self.video_type_combo.setCurrentIndex(index)
            last_video_file_path = settings.get('last_video_file_path')
            if isinstance(last_video_file_path, str) and Path(last_video_file_path).exists():
                self.video_source = last_video_file_path
                self.btn_start.setEnabled(True)

            id_alias = {}
            loaded_profiles = settings.get('source_profiles', [])
            if isinstance(loaded_profiles, list):
                self.source_profiles = []
                seen_keys = {}
                used_ids = set()
                fallback_id = 1

                for raw_profile in loaded_profiles:
                    if not isinstance(raw_profile, dict):
                        continue
                    source = raw_profile.get('source')
                    if source is None:
                        continue

                    source_key = self._source_to_key(source)
                    source_type = raw_profile.get('source_type', 'source')
                    name = raw_profile.get('name', self._display_name_for_source(source))
                    profile_settings = raw_profile.get('settings', {})
                    if not isinstance(profile_settings, dict):
                        profile_settings = {}

                    raw_id = raw_profile.get('id', 0)
                    try:
                        profile_id = int(raw_id)
                    except Exception:
                        profile_id = 0
                    if profile_id <= 0 or profile_id in used_ids:
                        while fallback_id in used_ids:
                            fallback_id += 1
                        if profile_id > 0:
                            id_alias[profile_id] = fallback_id
                        profile_id = fallback_id
                        fallback_id += 1

                    normalized = {
                        'id': profile_id,
                        'source': source,
                        'source_key': source_key,
                        'source_type': source_type,
                        'name': name,
                        'settings': profile_settings,
                    }

                    existing_idx = seen_keys.get(source_key)
                    if existing_idx is not None:
                        existing = self.source_profiles[existing_idx]
                        id_alias[profile_id] = existing.get('id')

                        existing_name = str(existing.get('name') or "").strip()
                        incoming_name = str(name or "").strip()
                        default_existing_name = self._display_name_for_source(existing.get('source'))
                        if incoming_name and (not existing_name or existing_name == default_existing_name):
                            existing['name'] = incoming_name

                        if existing.get('source_type', 'source') in ("", "source", None) and source_type not in ("", "source", None):
                            existing['source_type'] = source_type
                        if profile_settings:
                            existing['settings'] = profile_settings
                        continue

                    self.source_profiles.append(normalized)
                    seen_keys[source_key] = len(self.source_profiles) - 1
                    used_ids.add(profile_id)

            self.source_profile_counter = int(settings.get('source_profile_counter', max(1, len(self.source_profiles) + 1)))
            if self.source_profiles:
                max_id = max(int(p.get('id', 0)) for p in self.source_profiles)
                self.source_profile_counter = max(self.source_profile_counter, max_id + 1)
            raw_active_id = settings.get('active_source_profile_id')
            try:
                active_id = int(raw_active_id) if raw_active_id is not None else None
            except Exception:
                active_id = None
            if active_id in id_alias:
                active_id = id_alias[active_id]
            valid_ids = {int(p.get('id', 0)) for p in self.source_profiles}
            if active_id not in valid_ids:
                active_id = self.source_profiles[0].get('id') if self.source_profiles else None
            self.active_source_profile_id = active_id
            loaded_columns = int(settings.get('source_grid_columns', self.source_grid_columns))
            self.source_grid_columns = max(2, min(7, loaded_columns))
            loaded_text_size = str(settings.get('source_tile_text_size', self.source_tile_text_size))
            if loaded_text_size not in {"Small", "Medium", "Large"}:
                loaded_text_size = "Medium"
            self.source_tile_text_size = loaded_text_size
            self.pin_active_source_first = bool(settings.get('pin_active_source_first', self.pin_active_source_first))
            if hasattr(self, 'source_grid_combo'):
                self.source_grid_combo.blockSignals(True)
                self.source_grid_combo.setCurrentText(str(self.source_grid_columns))
                self.source_grid_combo.blockSignals(False)
            if hasattr(self, 'source_text_size_combo'):
                self.source_text_size_combo.blockSignals(True)
                self.source_text_size_combo.setCurrentText(self.source_tile_text_size)
                self.source_text_size_combo.blockSignals(False)
            if hasattr(self, 'pin_active_checkbox'):
                self.pin_active_checkbox.blockSignals(True)
                self.pin_active_checkbox.setChecked(self.pin_active_source_first)
                self.pin_active_checkbox.blockSignals(False)
            self._apply_admin_source_grid_columns(refresh_cards=False)
            self._refresh_source_profiles_ui()
            
            # ROI settings
            if 'roi_threshold' in settings:
                self.roi_threshold_slider.setValue(settings['roi_threshold'])

            if 'vehicle_alert_limit' in settings and hasattr(self, 'vehicle_alert_limit_slider'):
                self.vehicle_alert_limit_slider.setValue(int(settings['vehicle_alert_limit']))
            if hasattr(self, 'vehicle_alert_scope_combo'):
                self.vehicle_alert_scope = 'AUTO'
                self.vehicle_alert_scope_combo.blockSignals(True)
                self.vehicle_alert_scope_combo.setCurrentIndex(0)
                self.vehicle_alert_scope_combo.blockSignals(False)
            
            if 'roi_visible' in settings:
                self.roi_visible = settings['roi_visible']
                self.roi_manager.visible = self.roi_visible
                # Update button state
                if not self.roi_visible:
                    self.btn_toggle_roi.setChecked(True)
                    self.btn_toggle_roi.setText("👁️ Show")
            if settings.get('roi_config'):
                self.roi_manager.load_config(settings['roi_config'])
                self.roi_visible = self.roi_manager.visible
                self.btn_toggle_roi.blockSignals(True)
                self.btn_toggle_roi.setChecked(not self.roi_visible)
                self.btn_toggle_roi.blockSignals(False)
                self.btn_toggle_roi.setText("Hide" if self.roi_visible else "Show")
                self.update_roi_status()

            if settings.get('sidebar_visible') is False:
                self.scroll_area.setVisible(False)
                self.btn_toggle_sidebar.setText("Mo Cau Hinh")
            else:
                self.scroll_area.setVisible(True)
                self.btn_toggle_sidebar.setText("An Cau Hinh")
            runtime_expanded = settings.get('runtime_expanded', True)
            self.runtime_toggle_button.blockSignals(True)
            self.runtime_toggle_button.setChecked(runtime_expanded)
            self.runtime_toggle_button.blockSignals(False)
            self.toggle_runtime_section()

            # Enforce page-specific visibility after settings restore.
            if hasattr(self, 'main_view_stack'):
                if self.main_view_stack.currentIndex() == 0:
                    self.show_admin_dashboard()
                else:
                    self.show_player_view()
            
            self._refresh_custom_model_input()
            self._refresh_runtime_overview()
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

    def _capture_source_profile_settings(self):
        """Capture current control values into a source profile payload."""
        return {
            'model_choice': self.model_combo.currentText(),
            'tracker_choice': self.tracker_combo.currentText(),
            'confidence': self.confidence_slider.value(),
            'duplicate_iou_threshold': self.duplicate_iou_slider.value(),
            'max_det': self.max_det_slider.value(),
            'tracker_age': self.tracker_age_slider.value(),
            'resize_scale': self.resize_combo.currentText(),
            'display_point_mode': self.btn_display_mode.isChecked(),
            'trail_enabled': self.btn_trail_toggle.isChecked(),
        }

    def _apply_source_profile_settings(self, settings):
        """Apply source-specific settings to the control panel."""
        if not settings:
            return

        try:
            model_choice = settings.get('model_choice')
            if model_choice:
                index = self.model_combo.findText(model_choice)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

            tracker_choice = settings.get('tracker_choice')
            if tracker_choice:
                index = self.tracker_combo.findText(tracker_choice)
                if index >= 0:
                    self.tracker_combo.setCurrentIndex(index)

            if 'confidence' in settings:
                self.confidence_slider.setValue(int(settings['confidence']))
            if 'duplicate_iou_threshold' in settings:
                self.duplicate_iou_slider.setValue(int(settings['duplicate_iou_threshold']))
            if 'max_det' in settings:
                self.max_det_slider.setValue(int(settings['max_det']))
            if 'tracker_age' in settings:
                self.tracker_age_slider.setValue(int(settings['tracker_age']))

            resize_text = settings.get('resize_scale')
            if resize_text:
                index = self.resize_combo.findText(resize_text)
                if index >= 0:
                    self.resize_combo.setCurrentIndex(index)

            if 'display_point_mode' in settings:
                checked = bool(settings['display_point_mode'])
                self.btn_display_mode.setChecked(checked)
                self.toggle_display_mode(checked)

            if 'trail_enabled' in settings:
                checked = bool(settings['trail_enabled'])
                self.btn_trail_toggle.setChecked(checked)
                self.toggle_trail(checked)
        except Exception as exc:
            print(f"[WARN] Apply source profile settings failed: {exc}")

    def _source_to_key(self, source):
        if isinstance(source, int):
            return f"cam:{source}"

        text = str(source).strip()
        if text.isdigit():
            return f"cam:{int(text)}"

        if Path(text).exists():
            try:
                return f"file:{str(Path(text).resolve()).lower()}"
            except Exception:
                return f"file:{text.lower()}"

        try:
            parts = urlsplit(text)
            scheme = parts.scheme.lower()
            if scheme in ("http", "https", "rtsp", "rtsps") and parts.netloc:
                netloc = parts.netloc.lower()
                path = parts.path or ""
                while len(path) > 1 and path.endswith("/"):
                    path = path[:-1]
                query_items = sorted(parse_qsl(parts.query, keep_blank_values=True))
                query = urlencode(query_items, doseq=True)
                normalized_url = urlunsplit((scheme, netloc, path, query, ""))
                return f"url:{normalized_url}"
        except Exception:
            pass

        return text.casefold()

    def _display_name_for_source(self, source):
        if isinstance(source, int):
            return f"Camera {source}"
        text = str(source).strip()
        if text.startswith("rtsp://"):
            return "RTSP Source"
        if text.startswith(("http://", "https://")):
            return "Web Stream"
        p = Path(text)
        if p.exists():
            return p.name
        return text[:48]

    def _find_source_profile_index(self, source):
        if source is None:
            return -1
        key = self._source_to_key(source)
        for i, profile in enumerate(self.source_profiles):
            profile_source = profile.get('source')
            profile_key = None
            if profile_source is not None:
                try:
                    profile_key = self._source_to_key(profile_source)
                except Exception:
                    profile_key = None
            if not profile_key:
                profile_key = profile.get('source_key')
            if profile_key and profile.get('source_key') != profile_key:
                # Normalize stale keys from old settings format.
                profile['source_key'] = profile_key
            if profile_key and profile_key == key:
                return i
        return -1

    def _create_placeholder_preview(self, title="NO PREVIEW"):
        pixmap = QPixmap(self.preview_width, self.preview_height)
        pixmap.fill(QColor("#1b2733"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QColor("#f8d4a4"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, title)
        painter.end()
        return pixmap

    def _preview_pixmap_from_bgr_frame(self, frame):
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        target_ratio = self.preview_width / max(1, self.preview_height)
        frame_ratio = w / max(1, h)

        if frame_ratio > target_ratio:
            crop_w = int(h * target_ratio)
            start_x = max(0, (w - crop_w) // 2)
            crop = frame[:, start_x:start_x + crop_w]
        else:
            crop_h = int(w / max(target_ratio, 1e-6))
            start_y = max(0, (h - crop_h) // 2)
            crop = frame[start_y:start_y + crop_h, :]

        thumb = cv2.resize(crop, (self.preview_width, self.preview_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            self.preview_width,
            self.preview_height,
            self.preview_width * 3,
            QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimage.copy())

    def _youtube_thumbnail_pixmap(self, source_url):
        try:
            import yt_dlp

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(str(source_url), download=False)

            thumb_url = info.get('thumbnail')
            if not thumb_url:
                thumbnails = info.get('thumbnails') or []
                if thumbnails:
                    thumb_url = thumbnails[-1].get('url')

            if thumb_url:
                with urllib.request.urlopen(thumb_url, timeout=3.0) as response:
                    data = response.read()
                arr = np.frombuffer(data, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is not None:
                    return self._preview_pixmap_from_bgr_frame(image)
        except Exception:
            return None
        return None

    def _source_preview_pixmap(self, source, max_wait_s=None):
        key = self._source_to_key(source)
        cached = self.source_preview_cache.get(key)
        if cached is not None:
            return cached

        source_text = str(source).strip().lower()
        is_network = source_text.startswith(("rtsp://", "rtsps://", "http://", "https://"))

        if ("youtube.com" in source_text) or ("youtu.be" in source_text):
            yt_pixmap = self._youtube_thumbnail_pixmap(source)
            if yt_pixmap is not None:
                self.source_preview_cache[key] = yt_pixmap
                self.source_preview_meta[key] = {'kind': 'real', 'ts': time.time(), 'via': 'youtube-thumb'}
                self.preview_retry_after.pop(key, None)
                return yt_pixmap

        if max_wait_s is None:
            max_wait_s = 2.2 if is_network else 0.9

        pixmap = None
        cap = None
        try:
            if isinstance(source, int):
                cap = cv2.VideoCapture(source)
            else:
                text = str(source).strip()
                if text.lower().startswith("rtsp://"):
                    cap = cv2.VideoCapture(text, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    cap = cv2.VideoCapture(text)

            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2500)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2500)

                # Many HEVC streams fail on first packets; warm-up grabs improve first decodable frame odds.
                warmup_grabs = 12 if is_network else 4
                for _ in range(warmup_grabs):
                    cap.grab()

                frame = None
                deadline = time.time() + max_wait_s
                attempts = 0
                while time.time() < deadline:
                    attempts += 1
                    ret, candidate = cap.read()
                    if ret and candidate is not None and candidate.size > 0:
                        frame = candidate
                        break
                    cap.grab()
                    time.sleep(0.025)
                    if attempts >= 80:
                        break

                if frame is not None:
                    pixmap = self._preview_pixmap_from_bgr_frame(frame)
        except Exception:
            pixmap = None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        if pixmap is None:
            pixmap = self._create_placeholder_preview("NO SIGNAL")
            self.source_preview_meta[key] = {'kind': 'placeholder', 'ts': time.time(), 'via': 'fallback'}
            self.preview_retry_after[key] = time.time() + 20.0
        else:
            self.source_preview_meta[key] = {'kind': 'real', 'ts': time.time(), 'via': 'capture'}
            self.preview_retry_after.pop(key, None)

        self.source_preview_cache[key] = pixmap
        return pixmap

    def _enqueue_preview_load(self, source, force=False):
        key = self._source_to_key(source)
        if key in self.source_preview_cache or key in self.preview_load_pending:
            return

        if not force:
            retry_at = self.preview_retry_after.get(key, 0.0)
            if retry_at > time.time():
                return

        self.preview_load_queue.append(source)
        self.preview_load_pending.add(key)
        if not self.preview_loader_timer.isActive():
            self.preview_loader_timer.start(15)

    def _process_preview_queue(self):
        if not self.preview_load_queue:
            return

        source = self.preview_load_queue.pop(0)
        key = self._source_to_key(source)
        try:
            pixmap = self._source_preview_pixmap(source, max_wait_s=0.35)
            self._update_admin_card_preview_icon(key, pixmap)
        finally:
            self.preview_load_pending.discard(key)

        if self.preview_load_queue:
            self.preview_loader_timer.start(15)

    def _update_admin_card_preview_icon(self, source_key, pixmap):
        if not hasattr(self, 'admin_source_list'):
            return

        profile_id = None
        for profile in self.source_profiles:
            if self._source_to_key(profile.get('source')) == source_key:
                profile_id = profile.get('id')
                break
        if profile_id is None:
            return

        row_map = getattr(self, 'admin_row_profile_ids', [])
        if not isinstance(row_map, list):
            return
        try:
            row = row_map.index(profile_id)
        except ValueError:
            return
        if row < 0 or row >= self.admin_source_list.count():
            return

        item = self.admin_source_list.item(row)
        if item is not None:
            item.setIcon(QIcon(pixmap))

    def add_admin_source_from_input(self):
        raw = self.admin_source_input.text().strip() if hasattr(self, 'admin_source_input') else ""
        if not raw:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Enter source URL or camera id")
            return
        source = int(raw) if raw.isdigit() else raw
        existed_before = self._find_source_profile_index(source) >= 0
        display_name = self.admin_name_input.text().strip() if hasattr(self, 'admin_name_input') else ""
        src_key = self._source_to_key(source)
        self.source_preview_cache.pop(src_key, None)
        self.source_preview_meta.pop(src_key, None)
        self.preview_retry_after.pop(src_key, None)
        self._upsert_source_profile(
            source,
            source_type=self._infer_source_type(source),
            auto_select=True,
            display_name=display_name or None
        )
        self.schedule_settings_save()
        self.refresh_admin_source_cards()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText("Nguon da ton tai: da cap nhat" if existed_before else "Da them nguon")

    def add_admin_source_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if not file_path:
            return
        existed_before = self._find_source_profile_index(file_path) >= 0
        display_name = self.admin_name_input.text().strip() if hasattr(self, 'admin_name_input') else ""
        src_key = self._source_to_key(file_path)
        self.source_preview_cache.pop(src_key, None)
        self.source_preview_meta.pop(src_key, None)
        self.preview_retry_after.pop(src_key, None)
        self._upsert_source_profile(
            file_path,
            source_type="file",
            auto_select=True,
            display_name=display_name or None
        )
        self.schedule_settings_save()
        self.refresh_admin_source_cards()
        if hasattr(self, 'source_profile_status'):
            if existed_before:
                self.source_profile_status.setText("File da ton tai: da cap nhat")
            else:
                self.source_profile_status.setText(f"Da them file: {Path(file_path).name}")

    def refresh_admin_source_previews(self):
        self.preview_load_queue.clear()
        self.preview_load_pending.clear()
        if self.preview_loader_timer.isActive():
            self.preview_loader_timer.stop()
        self.source_preview_cache.clear()
        self.source_preview_meta.clear()
        self.preview_retry_after.clear()
        self.refresh_admin_source_cards()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText("Previews refreshed")

    def _infer_source_type(self, source):
        if isinstance(source, int):
            return "camera"
        text = str(source).strip().lower()
        if Path(str(source)).exists():
            return "file"
        if "youtube" in text or "youtu.be" in text:
            return "youtube"
        if text.startswith("rtsp://"):
            return "rtsp"
        if text.startswith(("http://", "https://")):
            return "stream"
        return "source"

    def open_source_manager_dialog(self):
        """Open dedicated source-management page as a separate dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Source Manager")
        dialog.resize(760, 520)

        root = QVBoxLayout(dialog)
        title = QLabel("Camera and Stream Sources")
        title.setObjectName("videoTitle")
        root.addWidget(title)

        subtitle = QLabel("Select one source profile and open it in the main player.")
        subtitle.setObjectName("subtleInfo")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        source_input_row = QHBoxLayout()
        source_input = QLineEdit()
        source_input.setPlaceholderText("rtsp://..., https://..., or camera id like 0")
        source_input_row.addWidget(source_input)

        btn_add_source = QPushButton("Add Source")
        source_input_row.addWidget(btn_add_source)
        root.addLayout(source_input_row)

        list_widget = QListWidget()
        list_widget.setMinimumHeight(280)
        root.addWidget(list_widget)

        status = QLabel("Ready")
        status.setObjectName("subtleInfo")
        root.addWidget(status)

        btn_row = QHBoxLayout()
        btn_add_current = QPushButton("Add Current")
        btn_row.addWidget(btn_add_current)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.setObjectName("dangerAction")
        btn_row.addWidget(btn_remove)

        btn_open = QPushButton("Open Selected")
        btn_open.setObjectName("primaryAction")
        btn_row.addWidget(btn_open)

        btn_close = QPushButton("Close")
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        if hasattr(self, 'stream_input'):
            source_input.setText(self.stream_input.currentText().strip())

        def refresh_list():
            list_widget.clear()
            active_row = -1
            for idx, profile in enumerate(self.source_profiles):
                name = profile.get('name') or self._display_name_for_source(profile.get('source'))
                stype = profile.get('source_type', 'source')
                src = profile.get('source')
                item = QListWidgetItem(f"{name} [{stype}]\n{src}")
                item.setData(Qt.UserRole, profile.get('id'))
                list_widget.addItem(item)
                if profile.get('id') == self.active_source_profile_id:
                    active_row = idx
            if active_row >= 0:
                list_widget.setCurrentRow(active_row)

        def add_manual_source():
            raw = source_input.text().strip()
            if not raw:
                status.setText("Enter a source first")
                return
            source = int(raw) if raw.isdigit() else raw
            self._upsert_source_profile(source, source_type=self._infer_source_type(source), auto_select=True)
            refresh_list()
            self.schedule_settings_save()
            status.setText("Source added")

        def add_current_source():
            if self.video_source is None:
                raw = source_input.text().strip()
                if not raw:
                    status.setText("No current source to add")
                    return
                source = int(raw) if raw.isdigit() else raw
            else:
                source = self.video_source
            self._upsert_source_profile(source, source_type=self._infer_source_type(source), auto_select=True)
            refresh_list()
            self.schedule_settings_save()
            status.setText("Current source added")

        def remove_selected_source():
            row = list_widget.currentRow()
            if row < 0 or row >= len(self.source_profiles):
                status.setText("Select a source first")
                return
            removed = self.source_profiles.pop(row)
            if removed.get('id') == self.active_source_profile_id:
                self.active_source_profile_id = None
            refresh_list()
            self._refresh_source_profiles_ui()
            self.schedule_settings_save()
            status.setText("Source removed")

        def open_selected_source():
            row = list_widget.currentRow()
            if row < 0 or row >= len(self.source_profiles):
                status.setText("Select a source first")
                return
            profile = self.source_profiles[row]
            self.active_source_profile_id = profile.get('id')
            self.video_source = profile.get('source')
            self._apply_source_profile_settings(profile.get('settings', {}))
            self._refresh_runtime_overview()
            self.btn_start.setEnabled(True)
            self.show_player_view()
            dialog.accept()
            QTimer.singleShot(0, self.start_processing)

        btn_add_source.clicked.connect(add_manual_source)
        btn_add_current.clicked.connect(add_current_source)
        btn_remove.clicked.connect(remove_selected_source)
        btn_open.clicked.connect(open_selected_source)
        btn_close.clicked.connect(dialog.reject)

        refresh_list()
        dialog.exec_()

    def show_admin_dashboard(self):
        if hasattr(self, 'main_view_stack'):
            self.main_view_stack.setCurrentIndex(0)
        self.video_title_label.setText("Camera Admin Dashboard")
        self.refresh_admin_source_cards()
        if hasattr(self, 'stream_group'):
            self.stream_group.setVisible(True)
        if hasattr(self, 'source_hub_group'):
            self.source_hub_group.setVisible(True)
        if hasattr(self, 'status_panel'):
            self.status_panel.setVisible(False)
        if hasattr(self, 'btn_load_video'):
            self.btn_load_video.setVisible(False)
        if hasattr(self, 'btn_start'):
            self.btn_start.setVisible(False)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.setVisible(False)
        if hasattr(self, 'btn_toggle_sidebar'):
            self.btn_toggle_sidebar.setVisible(False)
        if hasattr(self, 'scroll_area'):
            self.scroll_area.setVisible(False)

    def show_player_view(self):
        if hasattr(self, 'main_view_stack'):
            self.main_view_stack.setCurrentIndex(1)
        self.video_title_label.setText("Live View")
        # Keep player focused: source creation/management is only on Source Page.
        if hasattr(self, 'stream_group'):
            self.stream_group.setVisible(False)
        if hasattr(self, 'source_hub_group'):
            self.source_hub_group.setVisible(False)
        if hasattr(self, 'status_panel'):
            self.status_panel.setVisible(True)
        if hasattr(self, 'btn_load_video'):
            self.btn_load_video.setVisible(True)
        if hasattr(self, 'btn_start'):
            self.btn_start.setVisible(True)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.setVisible(True)
        if hasattr(self, 'btn_toggle_sidebar'):
            self.btn_toggle_sidebar.setVisible(True)
            self.btn_toggle_sidebar.setText("Mo Cau Hinh")
        if hasattr(self, 'scroll_area'):
            self.scroll_area.setVisible(False)

    def refresh_admin_source_cards(self):
        if not hasattr(self, 'admin_source_list'):
            return

        self.admin_source_list.blockSignals(True)
        self.admin_source_list.clear()
        self.admin_row_profile_ids = []
        active_row = -1

        display_profiles = list(self.source_profiles)
        if self.pin_active_source_first and self.active_source_profile_id is not None:
            display_profiles.sort(key=lambda p: 0 if p.get('id') == self.active_source_profile_id else 1)

        for idx, profile in enumerate(display_profiles):
            title = profile.get('name') or self._display_name_for_source(profile.get('source'))
            source_type = profile.get('source_type', 'source')
            source = profile.get('source')
            source_key = self._source_to_key(source)
            is_active = profile.get('id') == self.active_source_profile_id
            compact_title = " ".join(str(title).split())
            size_factor = {"Small": 6.8, "Medium": 6.2, "Large": 5.8}.get(self.source_tile_text_size, 6.2)
            max_chars = max(18, min(46, int(self.preview_width / size_factor)))
            if len(compact_title) > max_chars:
                compact_title = compact_title[:max_chars - 3] + "..."
            item_title = f"ACTIVE | {compact_title}" if is_active else compact_title
            item = QListWidgetItem(f"{item_title}\n{source_type.upper()}")
            preview = self.source_preview_cache.get(source_key)
            if preview is None:
                preview = self._create_placeholder_preview("LOADING")
                self._enqueue_preview_load(source)
            item.setIcon(QIcon(preview))
            item.setData(Qt.UserRole, profile.get('id'))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setToolTip(f"{title}\nType: {source_type}\nSource: {source}")
            if is_active:
                item.setBackground(QColor("#dcfce7"))
                item.setForeground(QColor("#14532d"))
            else:
                item.setBackground(QColor("#fffdf8"))
                item.setForeground(QColor("#374151"))
            self.admin_source_list.addItem(item)
            self.admin_row_profile_ids.append(profile.get('id'))
            if is_active:
                active_row = idx

        if active_row >= 0:
            self.admin_source_list.setCurrentRow(active_row)
        self.admin_source_list.blockSignals(False)
        self._update_admin_open_button_state()

        if self.preview_load_queue and not self.preview_loader_timer.isActive():
            self.preview_loader_timer.start(15)

    def _update_admin_open_button_state(self):
        if not hasattr(self, 'btn_admin_open_source') or not hasattr(self, 'admin_source_list'):
            return
        row = self.admin_source_list.currentRow()
        has_valid_selection = 0 <= row < len(getattr(self, 'admin_row_profile_ids', []))
        self.btn_admin_open_source.setEnabled(has_valid_selection)
        if hasattr(self, 'btn_admin_rename_source'):
            self.btn_admin_rename_source.setEnabled(has_valid_selection)

    def on_admin_source_selected(self, row):
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            if hasattr(self, 'admin_name_input'):
                self.admin_name_input.setText("")
            self._update_admin_open_button_state()
            return

        profile_id = self.admin_row_profile_ids[row]
        profile = next((p for p in self.source_profiles if p.get('id') == profile_id), None)
        if profile is None:
            self._update_admin_open_button_state()
            return

        self.active_source_profile_id = profile.get('id')
        if hasattr(self, 'admin_name_input'):
            self.admin_name_input.setText(profile.get('name') or "")
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText(f"Selected: {profile.get('name')}")
        self._update_admin_open_button_state()

    def _capture_runtime_preview_for_active_source(self, frame):
        if frame is None or self.video_source is None:
            return
        now = time.perf_counter()

        source_key = self._source_to_key(self.video_source)
        meta = self.source_preview_meta.get(source_key, {})
        if meta.get('kind') == 'real':
            return
        if (now - self.last_preview_capture_ts) < self.preview_capture_interval:
            return

        pixmap = self._preview_pixmap_from_bgr_frame(frame)
        if pixmap is None:
            return

        self.source_preview_cache[source_key] = pixmap
        self.source_preview_meta[source_key] = {'kind': 'real', 'ts': time.time(), 'via': 'runtime'}
        self.preview_retry_after.pop(source_key, None)
        self.last_preview_capture_ts = now
        self._update_admin_card_preview_icon(source_key, pixmap)

    def show_admin_source_context_menu(self, pos):
        if not hasattr(self, 'admin_source_list'):
            return

        item = self.admin_source_list.itemAt(pos)
        if item is None:
            return

        row = self.admin_source_list.row(item)
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            return

        self.admin_source_list.setCurrentRow(row)

        menu = QMenu(self)
        rename_action = menu.addAction("Doi ten")
        change_source_action = menu.addAction("Doi nguon")
        delete_action = menu.addAction("Xoa nguon")
        selected_action = menu.exec_(self.admin_source_list.mapToGlobal(pos))

        if selected_action == rename_action:
            self.rename_admin_source_from_menu(row)
        elif selected_action == change_source_action:
            self.change_admin_source_from_menu(row)
        elif selected_action == delete_action:
            self.remove_admin_source_from_menu(row)

    def rename_admin_source_from_menu(self, row):
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            return

        profile_id = self.admin_row_profile_ids[row]
        source_row = next((idx for idx, p in enumerate(self.source_profiles) if p.get('id') == profile_id), -1)
        if source_row < 0:
            return

        profile = self.source_profiles[source_row]
        current_name = profile.get('name') or self._display_name_for_source(profile.get('source'))
        new_name, ok = QInputDialog.getText(self, "Doi ten nguon", "Ten moi:", text=str(current_name))
        if not ok:
            return

        new_name = (new_name or "").strip()
        if not new_name:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Ten khong duoc de trong")
            return

        self.source_profiles[source_row]['name'] = new_name
        self._refresh_source_profiles_ui()
        self.schedule_settings_save()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText(f"Da doi ten: {new_name}")

    def change_admin_source_from_menu(self, row):
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            return

        profile_id = self.admin_row_profile_ids[row]
        source_row = next((idx for idx, p in enumerate(self.source_profiles) if p.get('id') == profile_id), -1)
        if source_row < 0:
            return

        profile = self.source_profiles[source_row]
        old_source = profile.get('source')
        current_value = str(old_source)
        new_raw, ok = QInputDialog.getText(self, "Doi nguon", "Nguon moi (URL/camera id/file):", text=current_value)
        if not ok:
            return

        new_raw = (new_raw or "").strip()
        if not new_raw:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Nguon moi khong hop le")
            return

        new_source = int(new_raw) if new_raw.isdigit() else new_raw
        dup_idx = self._find_source_profile_index(new_source)
        if dup_idx >= 0 and dup_idx != source_row:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Trung nguon: da ton tai profile khac")
            return

        old_key = self._source_to_key(old_source)
        new_key = self._source_to_key(new_source)

        current_name = profile.get('name') or ""
        auto_name_old = self._display_name_for_source(old_source)

        profile['source'] = new_source
        profile['source_key'] = new_key
        profile['source_type'] = self._infer_source_type(new_source)
        if not current_name or current_name == auto_name_old:
            profile['name'] = self._display_name_for_source(new_source)

        self.source_preview_cache.pop(old_key, None)
        self.source_preview_cache.pop(new_key, None)

        self._refresh_source_profiles_ui()
        self.admin_source_list.setCurrentRow(row)
        self.schedule_settings_save()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText("Da cap nhat nguon")

    def remove_admin_source_from_menu(self, row):
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            return

        profile_id = self.admin_row_profile_ids[row]
        source_row = next((idx for idx, p in enumerate(self.source_profiles) if p.get('id') == profile_id), -1)
        if source_row < 0:
            return

        removed = self.source_profiles.pop(source_row)
        removed_source = removed.get('source')
        removed_key = self._source_to_key(removed_source)

        self.source_preview_cache.pop(removed_key, None)
        self.source_preview_meta.pop(removed_key, None)
        self.preview_retry_after.pop(removed_key, None)

        was_active = removed.get('id') == self.active_source_profile_id
        if was_active:
            self.active_source_profile_id = None

        is_current_source = False
        if self.video_source is not None:
            try:
                is_current_source = self._source_to_key(self.video_source) == removed_key
            except Exception:
                is_current_source = False

        if is_current_source:
            if self.video_thread and self.video_thread.isRunning():
                self.stop_processing()
            self.video_source = None
            if hasattr(self, 'btn_start'):
                self.btn_start.setEnabled(False)

        self._refresh_source_profiles_ui()
        self.refresh_admin_source_cards()
        self._refresh_runtime_overview()
        self.schedule_settings_save()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText("Da xoa nguon")

    def open_selected_admin_source(self):
        if not hasattr(self, 'admin_source_list'):
            return
        row = self.admin_source_list.currentRow()
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Select a source profile first")
            return

        profile_id = self.admin_row_profile_ids[row]
        source_row = next((idx for idx, p in enumerate(self.source_profiles) if p.get('id') == profile_id), -1)
        if source_row < 0:
            return

        if hasattr(self, 'source_list_widget'):
            self.source_list_widget.setCurrentRow(source_row)
        self.open_selected_source_profile()

    def _refresh_source_profiles_ui(self):
        self.source_list_widget.blockSignals(True)
        self.source_list_widget.clear()

        active_row = -1
        for idx, profile in enumerate(self.source_profiles):
            title = profile.get('name') or self._display_name_for_source(profile.get('source'))
            source_type = profile.get('source_type', 'source')
            item = QListWidgetItem(f"{title}  [{source_type}]")
            item.setData(Qt.UserRole, profile.get('id'))
            self.source_list_widget.addItem(item)
            if profile.get('id') == self.active_source_profile_id:
                active_row = idx

        if active_row >= 0:
            self.source_list_widget.setCurrentRow(active_row)

        self.source_list_widget.blockSignals(False)

        if self.source_profiles:
            self.source_profile_status.setText(f"{len(self.source_profiles)} sources ready")
        else:
            self.source_profile_status.setText("No source profiles yet")
        self.refresh_admin_source_cards()

    def _upsert_source_profile(self, source, source_type="source", auto_select=True, display_name=None):
        if source is None or str(source).strip() == "":
            return

        # Compute source key for duplicate detection
        source_key = self._source_to_key(source)
        
        # Find existing profile - use computed key for safety
        idx = -1
        for i, profile in enumerate(self.source_profiles):
            profile_source = profile.get('source')
            profile_key = None
            if profile_source is not None:
                try:
                    profile_key = self._source_to_key(profile_source)
                except Exception:
                    profile_key = None
            if not profile_key:
                profile_key = profile.get('source_key')
            if profile_key and profile.get('source_key') != profile_key:
                profile['source_key'] = profile_key
            if profile_key and profile_key == source_key:
                idx = i
                break
        
        payload = self._capture_source_profile_settings()

        if idx >= 0:
            # Update existing profile
            self.source_profiles[idx]['source'] = source
            self.source_profiles[idx]['source_key'] = source_key
            self.source_profiles[idx]['settings'] = payload
            self.source_profiles[idx]['source_type'] = source_type
            if display_name is not None:
                self.source_profiles[idx]['name'] = display_name
            profile_id = self.source_profiles[idx]['id']
        else:
            # Create new profile
            profile_id = self.source_profile_counter
            self.source_profile_counter += 1
            self.source_profiles.append({
                'id': profile_id,
                'source': source,
                'source_key': source_key,
                'source_type': source_type,
                'name': display_name if display_name else self._display_name_for_source(source),
                'settings': payload,
            })

        if auto_select:
            self.active_source_profile_id = profile_id
        self._refresh_source_profiles_ui()

    def rename_selected_admin_source(self):
        if not hasattr(self, 'admin_source_list') or not hasattr(self, 'admin_name_input'):
            return
        row = self.admin_source_list.currentRow()
        if row < 0 or row >= len(getattr(self, 'admin_row_profile_ids', [])):
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Chon nguon truoc khi doi ten")
            return

        profile_id = self.admin_row_profile_ids[row]
        source_row = next((idx for idx, p in enumerate(self.source_profiles) if p.get('id') == profile_id), -1)
        if source_row < 0:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Khong tim thay profile nguon")
            return

        new_name = self.admin_name_input.text().strip()
        if not new_name:
            if hasattr(self, 'source_profile_status'):
                self.source_profile_status.setText("Nhap ten moi")
            return

        self.source_profiles[source_row]['name'] = new_name
        self._refresh_source_profiles_ui()
        self.schedule_settings_save()
        if hasattr(self, 'source_profile_status'):
            self.source_profile_status.setText(f"Da luu ten: {new_name}")

    def add_current_source_profile(self):
        """Add current source into admin list with current settings."""
        if hasattr(self, 'main_view_stack') and self.main_view_stack.currentIndex() != 0:
            self.source_profile_status.setText("Open Source Page to add new sources")
            self.show_admin_dashboard()
            return

        if not self.video_source:
            self.source_profile_status.setText("No source to add")
            return

        source = self.video_source

        source_type = self._infer_source_type(source)

        self._upsert_source_profile(source, source_type=source_type, auto_select=True)
        self.source_profile_status.setText("Source profile added")
        self.schedule_settings_save()

    def on_source_profile_selected(self, row):
        if row < 0 or row >= len(self.source_profiles):
            return
        profile = self.source_profiles[row]
        self.active_source_profile_id = profile.get('id')
        self.source_profile_status.setText(f"Selected: {profile.get('name')}")

    def open_selected_source_profile(self):
        row = self.source_list_widget.currentRow()
        if row < 0 or row >= len(self.source_profiles):
            self.source_profile_status.setText("Select a source profile first")
            return

        self._cancel_pending_stream_resolve()
        profile = self.source_profiles[row]
        self.active_source_profile_id = profile.get('id')
        self.video_source = profile.get('source')
        self._apply_source_profile_settings(profile.get('settings', {}))
        self._refresh_runtime_overview()
        self.btn_start.setEnabled(True)
        self.status_label.setText(f"Source ready: {profile.get('name')}")
        self.show_player_view()

        # Let UI settle before heavy startup to reduce launch hitch.
        QTimer.singleShot(0, self.start_processing)

    def update_selected_source_profile(self):
        row = self.source_list_widget.currentRow()
        if row < 0 or row >= len(self.source_profiles):
            self.source_profile_status.setText("Select a source profile first")
            return

        self.source_profiles[row]['settings'] = self._capture_source_profile_settings()
        self.source_profile_status.setText("Source profile updated")
        self.schedule_settings_save()

    def remove_selected_source_profile(self):
        row = self.source_list_widget.currentRow()
        if row < 0 or row >= len(self.source_profiles):
            self.source_profile_status.setText("Select a source profile first")
            return

        removed = self.source_profiles.pop(row)
        removed_key = self._source_to_key(removed.get('source'))
        self.source_preview_cache.pop(removed_key, None)
        self.source_preview_meta.pop(removed_key, None)
        self.preview_retry_after.pop(removed_key, None)
        if removed.get('id') == self.active_source_profile_id:
            self.active_source_profile_id = None
        self._refresh_source_profiles_ui()
        self.source_profile_status.setText("Source profile removed")
        self.schedule_settings_save()
    
    def save_stream_to_history(self, url: str):
        """Save stream URL to history"""
        if not url or url.strip() == "":
            return

        normalized_new = self._source_to_key(url)
        
        # Remove if already exists (to move to top)
        for idx in range(self.stream_input.count() - 1, -1, -1):
            item_text = self.stream_input.itemText(idx)
            if self._source_to_key(item_text) == normalized_new:
                self.stream_input.removeItem(idx)
        
        # Add to top
        self.stream_input.insertItem(0, url.strip())
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
            self.schedule_settings_save()
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

"""
Video Processor - Core logic xử lý video với YOLOv3 + DeepSort
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Set, List
from contextlib import nullcontext
import inspect
import time
from PIL import Image, ImageDraw, ImageFont
from src.core import config
from src.inference.person_classifier import PersonClassifier

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _TrackSnapshot:
    """Lightweight track adapter used for cached or tracker-only rendering."""

    def __init__(self, track_id, bbox, cls_id, confirmed=True, vx=0.0, vy=0.0):
        self.track_id = track_id
        self._bbox = bbox
        self.det_class = cls_id
        self._confirmed = confirmed
        self.vx = float(vx)
        self.vy = float(vy)

    def to_ltrb(self):
        return self._bbox

    def is_confirmed(self):
        return self._confirmed

    def get_velocity(self):
        return self.vx, self.vy


def propagate_tracks(tracks_with_velocity, render_ts: float, det_ts: float):
    """
    Dịch chuyển box theo velocity để bù độ trễ detect->render.
    tracks_with_velocity: list (x1,y1,x2,y2,vx,vy,track_id)
    """
    dt = render_ts - det_ts
    if dt <= 0 or dt > 0.3:
        return [(t[0], t[1], t[2], t[3], t[6]) for t in tracks_with_velocity]

    result = []
    for (x1, y1, x2, y2, vx, vy, tid) in tracks_with_velocity:
        nx1 = x1 + vx * dt
        ny1 = y1 + vy * dt
        nx2 = x2 + vx * dt
        ny2 = y2 + vy * dt
        result.append((nx1, ny1, nx2, ny2, tid))
    return result


class VideoProcessor:
    """Class xử lý video với dual-model ensemble và tracking"""
    
    def __init__(self, model_person, model_vehicle, tracker):
        """
        Khởi tạo VideoProcessor
        
        Args:
            model_person: YOLO model cho Person detection
            model_vehicle: YOLO model cho Vehicle detection
            tracker: DeepSort tracker instance
        """
        self.model_person = model_person
        self.model_vehicle = model_vehicle
        self.tracker = tracker
        
        # Statistics tracking
        self.unique_ids: Set[int] = set()
        self.class_counts: Dict[str, int] = {}
        self.trails: Dict[int, List[Tuple[int, int]]] = {}
        
        # Confidence threshold
        self.confidence = config.DEFAULT_CONFIDENCE
        
        # Visual settings
        self.box_thickness = 2  # Default box thickness
        self.font_size = 14  # Default font size
        self.use_point_mode = True  # True = Point Label, False = Bounding Box
        
        # ROI Manager (optional)
        self.roi_manager = None
        
        # Detect if using Train2 model (multi-class)
        self.using_train2 = self._is_train2_model()
        
        # Frame counter for periodic cleanup
        self.frame_counter = 0
        self.cleanup_interval = 50  # Giảm từ 100 → 50 (cleanup thường xuyên hơn)
        
        # Person classifier
        self.person_classifier = PersonClassifier(iou_threshold=0.3)
        # Per-frame cache to avoid duplicate person classification work
        self._frame_person_types = {}
        self.last_stats_snapshot = {
            'active_objects': 0,
            'active_class_counts': {},
            'total_objects': 0,
            'class_counts': {},
        }
        self._last_det_ts = 0.0
        self._tracker_accepts_timestamp = False
        try:
            params = inspect.signature(self.tracker.update_tracks).parameters
            self._tracker_accepts_timestamp = 'frame_timestamp' in params
        except Exception:
            self._tracker_accepts_timestamp = False
    
    def reset_statistics(self):
        """Reset tất cả statistics"""
        self.unique_ids.clear()
        self.class_counts.clear()
        self.trails.clear()
        self._last_det_ts = 0.0
        self.last_stats_snapshot = {
            'active_objects': 0,
            'active_class_counts': {},
            'total_objects': 0,
            'class_counts': {},
        }
    
    def set_confidence(self, confidence: float):
        """
        Cập nhật confidence threshold
        
        Args:
            confidence: Giá trị confidence (0.0 - 1.0)
        """
        self.confidence = max(config.MIN_CONFIDENCE, min(config.MAX_CONFIDENCE, confidence))
    
    def set_box_thickness(self, thickness: int):
        """
        Cập nhật độ dày của bounding box
        
        Args:
            thickness: Độ dày (1-5 pixels)
        """
        self.box_thickness = max(1, min(5, thickness))
    
    def set_font_size(self, size: int):
        """
        Cập nhật kích thước font chữ
        
        Args:
            size: Kích thước font (8-72 pt)
        """
        self.font_size = max(8, min(72, size))
    
    def set_font_thickness(self, thickness: int):
        """
        Cập nhật độ đậm font chữ
        
        Args:
            thickness: Độ đậm (1-5)
        """
        self.font_thickness = max(1, min(5, thickness))
    
    def set_point_mode(self, enabled: bool):
        """
        Chuyển đổi giữa Point Label và Bounding Box mode
        
        Args:
            enabled: True = Point Label, False = Bounding Box
        """
        self.use_point_mode = enabled
    
    def set_roi_manager(self, roi_manager):
        """
        Set ROI manager for filtering detections
        
        Args:
            roi_manager: ROIManager instance
        """
        self.roi_manager = roi_manager
    
    def _is_train2_model(self) -> bool:
        """
        Check if using Train2 model (multi-class)
        
        Returns:
            True if using Train2 model, False otherwise
        """
        # Check if model_person and model_vehicle are the same instance
        # (Train2 uses same model for both)
        if self.model_person is self.model_vehicle:
            # Check if model has 5 classes (Train2 signature)
            try:
                if hasattr(self.model_person, 'names'):
                    names = self.model_person.names
                    if isinstance(names, dict) and len(names) == 5:
                        # Check if it has Train2 classes
                        expected = {'person', 'car', 'bus', 'truck', 'motorcycle'}
                        actual = {str(v).lower() for v in names.values()}
                        if expected == actual:
                            return True
            except Exception:
                pass
        return False

    def _get_shared_model_names(self) -> Dict[int, str]:
        """Return native class names when a single shared model is in use."""
        if self.model_person is not self.model_vehicle:
            return {}

        try:
            names = getattr(self.model_person, 'names', {})
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {idx: str(name) for idx, name in enumerate(names)}
        except Exception:
            pass

        return {}
    
    def _get_class_name(self, cls_id: int) -> str:
        """
        Get class name based on model type
        
        Args:
            cls_id: Class ID
            
        Returns:
            Class name string
        """
        shared_names = self._get_shared_model_names()
        if shared_names:
            return shared_names.get(cls_id, f"Class {cls_id}")
        if self.using_train2:
            return config.TRAIN2_CLASSES.get(cls_id, 'Unknown')
        else:
            return config.CLASS_NAMES.get(cls_id, 'Unknown')

    def _inference_context(self):
        """Run model inference without autograd overhead when torch is available."""
        if TORCH_AVAILABLE:
            return torch.inference_mode()
        return nullcontext()

    def _extract_detections_batch(self, results, scale_factor: float, class_override: int = None) -> List:
        """
        Extract detections with a batched tensor transfer when possible.
        Falls back to per-box extraction for non-standard result wrappers.
        """
        boxes = results.boxes
        if len(boxes) == 0:
            return []

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

            if scale_factor != 1.0:
                xyxy[:, [0, 2]] /= scale_factor
                xyxy[:, [1, 3]] /= scale_factor

            detections = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w, h = x2 - x1, y2 - y1
                cls_id = class_override if class_override is not None else int(cls_ids[i])
                detections.append([[x1, y1, w, h], float(confs[i]), cls_id])
            return detections
        except Exception:
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor
                conf = float(box.conf[0].cpu().numpy())
                cls_id = class_override if class_override is not None else int(box.cls[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append([[x1, y1, w, h], conf, cls_id])
            return detections

    def _update_tracker(self, detections, frame, frame_timestamp: float = None):
        """Update tracker and pass frame timestamp only when supported."""
        if self._tracker_accepts_timestamp and frame_timestamp is not None:
            return self.tracker.update_tracks(
                detections,
                frame=frame,
                frame_timestamp=frame_timestamp
            )
        return self.tracker.update_tracks(detections, frame=frame)

    @staticmethod
    def _get_track_velocity(track):
        """Get (vx, vy) in px/s if available, otherwise return zeros."""
        if hasattr(track, 'get_velocity'):
            try:
                vx, vy = track.get_velocity()
                return float(vx), float(vy)
            except Exception:
                return 0.0, 0.0

        vx = getattr(track, 'vx', 0.0)
        vy = getattr(track, 'vy', 0.0)
        return float(vx), float(vy)
    
    def process_frame(
        self,
        frame: np.ndarray,
        resize_scale: int = 100,
        max_det: int = 20,
        frame_timestamp: float = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Xử lý một frame với detection + tracking (STABLE & FAST)
        
        Args:
            frame: Input frame
            resize_scale: Resize scale percentage (25, 50, 75, 100)
            max_det: Maximum detections per frame
        """
        # Periodic cleanup to prevent memory growth without stalling inference
        self.frame_counter += 1
        self._frame_person_types = {}
        if self.frame_counter % self.cleanup_interval == 0:
            # Force cleanup tracker
            if hasattr(self.tracker, 'tracks'):
                # Remove old tracks
                if len(self.tracker.tracks) > 30:
                    self.tracker.tracks = self.tracker.tracks[-30:]
            
            # Clear old trails AGGRESSIVELY
            if len(self.trails) > 20:  # Giảm từ 30 → 20
                trail_ids = list(self.trails.keys())
                for old_id in trail_ids[:-20]:
                    del self.trails[old_id]

        # Avoid heavy GC/tracker hard resets in real-time path; they cause visible stalls.
        
        # Store original size
        orig_h, orig_w = frame.shape[:2]
        
        # Resize for inference if needed
        if resize_scale < 100:
            scale_factor = resize_scale / 100.0
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            inference_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            inference_frame = frame
            scale_factor = 1.0
        
        detections = []

        with self._inference_context():
            # Single shared model path: custom detectors and Train2 should run
            # once and keep their native class IDs from the model itself.
            if self.model_person is self.model_vehicle:
                results = self.model_person(
                    inference_frame,  # Use resized frame
                    conf=self.confidence,
                    iou=0.6,
                    verbose=False,
                    max_det=max_det  # Use dynamic max_det
                )[0]
                detections = self._extract_detections_batch(results, scale_factor)
            else:
                # Separate detection for person
                max_det_person = max(5, max_det // 2)  # Split max_det
                max_det_vehicle = max(5, max_det // 2)

                results_person = self.model_person(
                    inference_frame,  # Use resized frame
                    conf=self.confidence,
                    iou=0.6,
                    verbose=False,
                    classes=[config.PERSON_CLASS],
                    max_det=max_det_person
                )[0]
                detections.extend(
                    self._extract_detections_batch(
                        results_person,
                        scale_factor,
                        class_override=config.PERSON_CLASS
                    )
                )

                # Separate detection for vehicles
                results_vehicle = self.model_vehicle(
                    inference_frame,  # Use resized frame
                    conf=self.confidence,
                    iou=0.6,
                    verbose=False,
                    classes=config.VEHICLE_CLASSES,
                    max_det=max_det_vehicle
                )[0]
                detections.extend(self._extract_detections_batch(results_vehicle, scale_factor))
        
        # Filter overlapping detections (additional NMS for cross-class)
        detections = self._filter_overlapping_detections(detections)
        self._last_det_ts = frame_timestamp if frame_timestamp else time.monotonic()

        # Tracking
        tracks = self._update_tracker(detections, frame=frame, frame_timestamp=self._last_det_ts)
        
        # Draw results
        processed_frame = self._draw_detections(frame, tracks, det_ts=self._last_det_ts)
        
        # Update statistics
        stats = self._build_stats_payload(tracks)
        
        return processed_frame, stats

    def process_frame_tracking_only(self, frame: np.ndarray, frame_timestamp: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Reuse the latest tracker state and draw cached overlays without running
        detector inference. This keeps playback close to 1x when inference
        falls behind wall-clock time.
        """
        self.frame_counter += 1
        self._frame_person_types = {}

        tracks = self._get_render_tracks()
        det_ts = frame_timestamp if frame_timestamp else self._last_det_ts
        processed_frame = self._draw_detections(frame, tracks, det_ts=det_ts)
        stats = self._build_stats_payload(tracks, update_totals=False)
        return processed_frame, stats
    
    def _filter_overlapping_detections(self, detections, iou_threshold=0.7):
        """Filter overlapping detections to reduce noise (OPTIMIZED)"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        # Không giới hạn số lượng nữa - giữ tất cả detections
        
        filtered = []
        for det in detections:
            bbox, conf, cls_id = det
            x1, y1, w, h = bbox
            
            # Check overlap with already accepted detections
            is_duplicate = False
            for accepted in filtered:
                acc_bbox = accepted[0]
                acc_x1, acc_y1, acc_w, acc_h = acc_bbox
                
                # Calculate IOU
                xx1 = max(x1, acc_x1)
                yy1 = max(y1, acc_y1)
                xx2 = min(x1 + w, acc_x1 + acc_w)
                yy2 = min(y1 + h, acc_y1 + acc_h)
                
                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)
                inter_area = inter_w * inter_h
                
                union_area = (w * h) + (acc_w * acc_h) - inter_area
                
                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def _draw_detections(self, frame: np.ndarray, tracks, det_ts: float = 0.0) -> np.ndarray:
        """
        Vẽ bounding boxes, labels, và trails lên frame
        """
        # Get current track IDs
        current_track_ids = set()
        
        # First pass: collect all vehicle tracks
        vehicle_tracks = []
        person_tracks = []

        confirmed_tracks = []
        tracks_with_velocity = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            confirmed_tracks.append(track)
            ltrb = track.to_ltrb()
            vx, vy = self._get_track_velocity(track)
            tracks_with_velocity.append((
                float(ltrb[0]),
                float(ltrb[1]),
                float(ltrb[2]),
                float(ltrb[3]),
                vx,
                vy,
                track.track_id,
            ))

        propagated_map = {}
        if tracks_with_velocity:
            render_ts = time.monotonic()
            propagated = propagate_tracks(tracks_with_velocity, render_ts=render_ts, det_ts=det_ts)
            propagated_map = {
                tid: np.array([x1, y1, x2, y2], dtype=np.float32)
                for x1, y1, x2, y2, tid in propagated
            }

        for track in confirmed_tracks:
            track_id = track.track_id
            current_track_ids.add(track_id)
            ltrb = propagated_map.get(track_id, np.asarray(track.to_ltrb(), dtype=np.float32))
            cls_id = track.det_class
            
            # ROI filtering - skip if outside ROI
            if self.roi_manager and self.roi_manager.is_active():
                bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                if not self.roi_manager.is_object_in_roi(bbox):
                    continue
            
            # Separate person and vehicle tracks
            if cls_id == 0:  # Person
                person_tracks.append((track, ltrb))
            else:  # Vehicle
                vehicle_tracks.append((track, ltrb, cls_id))
        
        # Second pass: classify and draw persons
        for track, ltrb in person_tracks:
            track_id = track.track_id
            
            # Classify person (Pedestrian vs Rider)
            vehicle_bboxes = [(v_ltrb, v_cls) for _, v_ltrb, v_cls in vehicle_tracks]
            person_type = self.person_classifier.classify_person(
                [ltrb[0], ltrb[1], ltrb[2], ltrb[3]],
                track_id,
                vehicle_bboxes,
                self.frame_counter
            )
            self._frame_person_types[track_id] = person_type
            
            # Choose color based on person type
            if person_type == "Pedestrian":
                color = config.COLOR_PERSON  # Green
                class_name = "Pedestrian"
            else:  # Rider or Driver
                color = (255, 165, 0)  # Orange for riders
                class_name = person_type
            
            # Draw
            self._draw_single_track(frame, ltrb, track_id, class_name, color)
        
        # Third pass: draw vehicles
        for track, ltrb, cls_id in vehicle_tracks:
            track_id = track.track_id
            color = config.get_class_color(cls_id)
            class_name = self._get_class_name(cls_id)
            
            self._draw_single_track(frame, ltrb, track_id, class_name, color)
        
        # CRITICAL: Clean up old trails
        old_track_ids = set(self.trails.keys()) - current_track_ids
        for old_id in old_track_ids:
            del self.trails[old_id]
        
        # AGGRESSIVE: Limit total trails
        if len(self.trails) > 50:
            oldest_ids = list(self.trails.keys())[:len(self.trails) - 50]
            for old_id in oldest_ids:
                del self.trails[old_id]
        
        # Cleanup person classifier history
        self.person_classifier.cleanup_old_tracks(current_track_ids)
        
        # Draw ROI overlay if active
        if self.roi_manager and self.roi_manager.is_active():
            frame = self.roi_manager.draw_roi(frame)
        
        return frame
    
    def _draw_single_track(self, frame, ltrb, track_id, class_name, color):
        """Draw a single track (person or vehicle)"""
        # Tính tâm object
        center_x = int((ltrb[0] + ltrb[2]) / 2)
        center_y = int((ltrb[1] + ltrb[3]) / 2)
            
        if self.use_point_mode:
            # ===== POINT LABEL MODE =====
            # Vẽ điểm tròn nhỏ ở tâm
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), 1)
            
            # Vẽ label bên cạnh điểm
            label_lines = [class_name, f"ID:{track_id}"]
            
            # Font scale từ slider
            font_scale = self.font_size / 30.0
            font_thickness = max(1, int(self.font_size / 12))
            
            # Vị trí text
            line_length = 25
            text_x = center_x + line_length + 5
            text_y = center_y
            
            # Vẽ đường nối
            line_end_x = center_x + line_length
            cv2.line(frame, (center_x, center_y), (line_end_x, center_y), color, 1)
            
            # Tính kích thước background box
            line_height = int(14 + font_scale * 10)
            max_width = int(max(len(label_lines[0]), len(label_lines[1])) * (6 + font_scale * 8))
            
            box_height = len(label_lines) * line_height
            box_width = max_width + 6
            
            # Vẽ background box
            cv2.rectangle(frame, 
                         (text_x - 2, text_y - line_height + 3), 
                         (text_x + box_width, text_y + box_height - line_height + 3),
                         color,
                         -1)
            
            # Vẽ text
            for i, line in enumerate(label_lines):
                y_offset = text_y + i * line_height
                cv2.putText(frame, line, (text_x, y_offset), 
                           config.FONT, font_scale, (255, 255, 255), font_thickness)
        else:
            # ===== BOUNDING BOX MODE =====
            cv2.rectangle(
                frame,
                (int(ltrb[0]), int(ltrb[1])),
                (int(ltrb[2]), int(ltrb[3])),
                color,
                self.box_thickness
            )
            
            label = f"{class_name} v{track_id}"
            font_scale = self.font_size / 20.0
            font_thickness = max(1, int(self.font_size / 10))
            
            (label_w, label_h), _ = cv2.getTextSize(label, config.FONT, font_scale, font_thickness)
            
            cv2.rectangle(
                frame,
                (int(ltrb[0]), int(ltrb[1]) - label_h - 10),
                (int(ltrb[0]) + label_w + 10, int(ltrb[1])),
                color,
                -1
            )
            
            cv2.putText(
                frame,
                label,
                (int(ltrb[0]) + 5, int(ltrb[1]) - 5),
                config.FONT,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        # Vẽ trail (chỉ nếu TRAIL_LENGTH > 0)
        if config.TRAIL_LENGTH > 0:
            center = (center_x, center_y)
            
            if track_id not in self.trails:
                self.trails[track_id] = []
            
            self.trails[track_id].append(center)
            
            if len(self.trails[track_id]) > config.TRAIL_LENGTH:
                self.trails[track_id].pop(0)
            
            if len(self.trails[track_id]) >= 2:
                for i in range(1, len(self.trails[track_id])):
                    cv2.line(
                        frame,
                        self.trails[track_id][i-1],
                        self.trails[track_id][i],
                        color,
                        config.TRAIL_THICKNESS
                    )
    
    def _update_statistics(self, tracks) -> Dict:
        """
        Cập nhật statistics dựa trên tracks hiện tại
        Đếm riêng Pedestrian và Rider dựa trên classification đầu tiên
        """
        vehicle_tracks = []
        person_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_id = track.det_class
            ltrb = track.to_ltrb()

            # ROI filtering - only count if in ROI
            if self.roi_manager and self.roi_manager.is_active():
                bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                if not self.roi_manager.is_object_in_roi(bbox):
                    continue

            if cls_id == config.PERSON_CLASS:
                person_tracks.append((track_id, ltrb))
            else:
                vehicle_tracks.append((track_id, ltrb, cls_id))

        # Count vehicles (single pass, O(n))
        for track_id, _, cls_id in vehicle_tracks:
            if track_id in self.unique_ids:
                continue
            self.unique_ids.add(track_id)
            class_name = self._get_class_name(cls_id)
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        # Count persons (reuse cached classification from drawing pass when available)
        vehicle_bboxes = [(v_ltrb, v_cls_id) for _, v_ltrb, v_cls_id in vehicle_tracks]
        for track_id, ltrb in person_tracks:
            if track_id in self.unique_ids:
                continue

            self.unique_ids.add(track_id)
            person_type = self._frame_person_types.get(track_id)
            if person_type is None:
                person_type = self.person_classifier.classify_person(
                    [ltrb[0], ltrb[1], ltrb[2], ltrb[3]],
                    track_id,
                    vehicle_bboxes,
                    self.frame_counter
                )

            # Count based on first classification to avoid double counting track changes
            self.class_counts[person_type] = self.class_counts.get(person_type, 0) + 1
        
        return {
            'total_objects': len(self.unique_ids),
            'class_counts': self.class_counts.copy()
        }

    def _build_stats_payload(self, tracks, update_totals=True) -> Dict:
        """Build both live-frame and session stats in one payload."""
        active_objects, active_class_counts = self._summarize_active_tracks(tracks)
        if update_totals:
            totals = self._update_statistics(tracks)
        else:
            totals = {
                'total_objects': len(self.unique_ids),
                'class_counts': self.class_counts.copy(),
            }

        self.last_stats_snapshot = {
            'active_objects': active_objects,
            'active_class_counts': active_class_counts,
            'total_objects': totals['total_objects'],
            'class_counts': totals['class_counts'],
        }
        return self.last_stats_snapshot.copy()

    def _summarize_active_tracks(self, tracks) -> Tuple[int, Dict[str, int]]:
        """Count current visible tracks without mutating session totals."""
        active_objects = 0
        active_class_counts: Dict[str, int] = {}
        vehicle_tracks = []
        person_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            cls_id = track.det_class

            if self.roi_manager and self.roi_manager.is_active():
                bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                if not self.roi_manager.is_object_in_roi(bbox):
                    continue

            active_objects += 1
            if cls_id == config.PERSON_CLASS:
                person_tracks.append((track.track_id, ltrb))
            else:
                vehicle_tracks.append((track.track_id, ltrb, cls_id))
                class_name = self._get_class_name(cls_id)
                active_class_counts[class_name] = active_class_counts.get(class_name, 0) + 1

        vehicle_bboxes = [(v_ltrb, v_cls_id) for _, v_ltrb, v_cls_id in vehicle_tracks]
        for track_id, ltrb in person_tracks:
            person_type = self._frame_person_types.get(track_id)
            if person_type is None:
                person_type = self.person_classifier.classify_person(
                    [ltrb[0], ltrb[1], ltrb[2], ltrb[3]],
                    track_id,
                    vehicle_bboxes,
                    self.frame_counter
                )
            active_class_counts[person_type] = active_class_counts.get(person_type, 0) + 1

        return active_objects, active_class_counts

    def _get_render_tracks(self):
        """Return live tracker objects or lightweight snapshots for drawing."""
        if hasattr(self.tracker, 'tracks'):
            return list(self.tracker.tracks)

        if hasattr(self.tracker, 'trackers'):
            render_tracks = []
            max_staleness = max(1, min(2, getattr(self.tracker, 'max_age', 1)))
            for tracker in getattr(self.tracker, 'trackers', []):
                if getattr(tracker, 'time_since_update', 0) > max_staleness:
                    continue
                if not hasattr(tracker, 'get_state'):
                    continue
                bbox = tracker.get_state()[0]
                vx = 0.0
                vy = 0.0
                try:
                    state = tracker.kf.x.flatten()
                    vx = float(state[4])
                    vy = float(state[5])
                except Exception:
                    pass
                render_tracks.append(
                    _TrackSnapshot(
                        getattr(tracker, 'id', -1),
                        bbox,
                        getattr(tracker, 'det_class', config.PERSON_CLASS),
                        vx=vx,
                        vy=vy,
                    )
                )
            return render_tracks

        return []
    
    def _draw_detections_old(self, frame: np.ndarray, tracks) -> np.ndarray:
        """
        Vẽ bounding boxes, labels, và trails lên frame
        
        Args:
            frame: Frame gốc
            tracks: Danh sách tracks từ DeepSort
            
        Returns:
            Frame đã vẽ
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cls_id = track.det_class
            
            # Chọn màu theo loại
            color = config.COLOR_PERSON if cls_id == 0 else config.COLOR_VEHICLE
            
            # Vẽ bounding box với độ dày tùy chỉnh
            cv2.rectangle(
                frame,
                (int(ltrb[0]), int(ltrb[1])),
                (int(ltrb[2]), int(ltrb[3])),
                color,
                self.box_thickness
            )
            
            # VẼ ĐIỂM TRUNG TÂM thay vì bounding box
            center_x = int((ltrb[0] + ltrb[2]) / 2)
            center_y = int((ltrb[1] + ltrb[3]) / 2)
            
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), 1)
            
            class_name = self._get_class_name(cls_id)
            label = f"{class_name}\nv{track_id}_{int(ltrb[0])}"
            
            font_scale = self.font_size / 20.0
            font_thickness = max(1, int(self.font_size / 10))
            
            text_x = center_x + 10
            text_y = center_y - 5
            
            for i, line in enumerate(label.split('\n')):
                y_offset = text_y + i * 15
                cv2.putText(frame, line, (text_x+1, y_offset+1), 
                           config.FONT, font_scale, (0, 0, 0), font_thickness+1)
                cv2.putText(frame, line, (text_x, y_offset), 
                           config.FONT, font_scale, (255, 255, 255), font_thickness)
            
            # Vẽ trail (chỉ nếu TRAIL_LENGTH > 0)
            if config.TRAIL_LENGTH > 0:
                center = (center_x, center_y)
                
                if track_id not in self.trails:
                    self.trails[track_id] = []
                
                self.trails[track_id].append(center)
                
                # Giữ chỉ N điểm gần nhất
                if len(self.trails[track_id]) > config.TRAIL_LENGTH:
                    self.trails[track_id].pop(0)
                
                # Vẽ trail lines (chỉ khi có >= 2 điểm)
                if len(self.trails[track_id]) >= 2:
                    for i in range(1, len(self.trails[track_id])):
                        cv2.line(
                            frame,
                            self.trails[track_id][i-1],
                            self.trails[track_id][i],
                            color,
                            config.TRAIL_THICKNESS
                        )
        
        # KHÔNG vẽ thống kê lên video nữa
        # Statistics sẽ hiển thị riêng trong UI
        
        return frame
    
    def _put_vietnamese_text(self, frame, text, position, color, font_size=20):
        """
        Vẽ text tiếng Việt lên frame sử dụng PIL
        
        Args:
            frame: Frame numpy array (BGR)
            text: Text cần vẽ (hỗ trợ tiếng Việt)
            position: Tuple (x, y) vị trí góc trên trái
            color: Màu BGR tuple
            font_size: Kích thước font
            
        Returns:
            Frame đã vẽ text
        """
        # Convert BGR to RGB
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Load font - thử các font có sẵn trên Windows
        try:
            # Thử Arial Unicode (Windows)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Fallback: DejaVu
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                # Fallback: Default
                font = ImageFont.load_default()
        
        # Đo kích thước text
        bbox = draw.textbbox(position, text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Vẽ background
        bg_color = tuple(reversed(color))  # BGR to RGB
        draw.rectangle(
            [position[0], position[1], position[0] + text_width + 10, position[1] + text_height + 5],
            fill=bg_color
        )
        
        # Vẽ text trắng
        draw.text((position[0] + 5, position[1]), text, font=font, fill=(255, 255, 255))
        
        # Convert back to BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _draw_statistics_panel(self, frame: np.ndarray):
        """
        Vẽ bảng hiển thị thống kê đối tượng lên frame
        
        Args:
            frame: Frame để vẽ (in-place modification)
        """
        if not self.class_counts:
            return
        
        # Tính chiều cao panel
        panel_height = 50 + len(self.class_counts) * config.STATS_HEIGHT_PER_ITEM
        
        # Vẽ background
        cv2.rectangle(
            frame,
            (0, 0),
            (config.STATS_WIDTH, panel_height),
            config.STATS_BG_COLOR,
            -1
        )
        
        # Vẽ title
        cv2.putText(
            frame,
            "THONG KE:",
            (10, 25),
            config.FONT,
            0.7,
            config.TEXT_COLOR,
            2
        )
        
        # Vẽ từng dòng thống kê
        y_offset = 55
        for class_name, count in self.class_counts.items():
            text = f"{class_name}: {count}"
            text_color = config.COLOR_PERSON if class_name == 'Người' else config.COLOR_VEHICLE
            
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                config.FONT,
                config.FONT_SCALE,
                text_color,
                config.FONT_THICKNESS
            )
            y_offset += config.STATS_HEIGHT_PER_ITEM
    
    def process_video(self, input_path: str, output_path: str, 
                     progress_callback=None, max_frames: int = None,
                     frame_skip: int = 0, resize_scale: int = 100) -> Dict:
        """
        Xử lý toàn bộ video với optimization
        
        Args:
            input_path: Đường dẫn video đầu vào
            output_path: Đường dẫn video đầu ra
            progress_callback: Callback function để cập nhật progress
            max_frames: Số frame tối đa cần xử lý (None = all)
            frame_skip: Số frame bỏ qua giữa mỗi frame xử lý (0 = xử lý tất cả)
            resize_scale: Phần trăm resize (100 = giữ nguyên, 50 = giảm một nửa)
            
        Returns:
            Dictionary chứa statistics và thông tin xử lý
        """
        # Reset statistics
        self.reset_statistics()
        
        # Mở video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {input_path}")
        
        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 24
        
        # Tính kích thước mới nếu resize
        if resize_scale != 100:
            new_width = int(width * resize_scale / 100)
            new_height = int(height * resize_scale / 100)
        else:
            new_width, new_height = width, height
        
        # Giới hạn số frame nếu cần
        if max_frames:
            total_frames = min(total_frames, max_frames * (frame_skip + 1))
        
        # Tạo video writer với kích thước mới
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
        # Điều chỉnh FPS nếu skip frames
        output_fps = fps if frame_skip == 0 else fps // (frame_skip + 1)
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (new_width, new_height))
        
        frame_count = 0
        processed_count = 0
        
        try:
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame skipping
                if frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Resize frame nếu cần
                if resize_scale != 100:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Xử lý frame
                processed_frame, stats = self.process_frame(frame)
                
                # Ghi frame
                out.write(processed_frame)
                
                frame_count += 1
                processed_count += 1
                
                # Callback progress
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress, processed_count, 
                                    total_frames // (frame_skip + 1), stats)
        
        finally:
            cap.release()
            out.release()
        
        return {
            'frames_processed': processed_count,
            'total_objects': len(self.unique_ids),
            'class_counts': self.class_counts.copy(),
            'output_path': output_path
        }

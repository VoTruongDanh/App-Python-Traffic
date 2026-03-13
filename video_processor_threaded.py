"""
Threaded Video Processor - Parallel inference and drawing
Tách inference (GPU) và drawing (CPU) ra 2 threads để tăng FPS
"""
import cv2
import numpy as np
from typing import Tuple, Dict
import threading
import queue
from video_processor import VideoProcessor


class ThreadedVideoProcessor(VideoProcessor):
    """
    Video Processor with parallel inference and drawing
    - Thread 1: Inference (GPU)
    - Thread 2: Drawing (CPU)
    - Main thread: Tracking
    """
    
    def __init__(self, model_person, model_vehicle, tracker):
        super().__init__(model_person, model_vehicle, tracker)
        
        # Queues for parallel processing
        self.inference_queue = queue.Queue(maxsize=2)
        self.drawing_queue = queue.Queue(maxsize=2)
        
        # Threading
        self.inference_thread = None
        self.drawing_thread = None
        self.running = False
        
        # Cache
        self.last_detections = []
        
    def start_threads(self):
        """Start inference and drawing threads"""
        if self.running:
            return
            
        self.running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
        # Start drawing thread
        self.drawing_thread = threading.Thread(target=self._drawing_worker, daemon=True)
        self.drawing_thread.start()
        
    def stop_threads(self):
        """Stop all threads"""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        if self.drawing_thread:
            self.drawing_thread.join(timeout=1.0)
    
    def _inference_worker(self):
        """Worker thread for GPU inference"""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame, resize_scale, max_det = self.inference_queue.get(timeout=0.01)
                
                # Run inference (GPU)
                detections = self._run_inference(frame, resize_scale, max_det)
                
                # Put results to drawing queue
                try:
                    self.drawing_queue.put((frame, detections), block=False)
                except queue.Full:
                    pass  # Skip if queue full
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                continue
    
    def _drawing_worker(self):
        """Worker thread for CPU drawing"""
        while self.running:
            try:
                # Get frame and detections (non-blocking)
                frame, detections = self.drawing_queue.get(timeout=0.01)
                
                # This will be handled by main thread
                # Just cache the detections
                self.last_detections = detections
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Drawing error: {e}")
                continue
    
    def _run_inference(self, frame, resize_scale, max_det):
        """Run inference only (no tracking, no drawing)"""
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
        
        # Check if using Train2 model
        if self.using_train2:
            results = self.model_person(
                inference_frame,
                conf=self.confidence,
                iou=0.6,
                verbose=False,
                max_det=max_det
            )[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append([[x1, y1, w, h], conf, cls_id])
        
        elif self.model_person is self.model_vehicle:
            import config
            all_classes = [config.PERSON_CLASS] + config.VEHICLE_CLASSES
            results = self.model_person(
                inference_frame,
                conf=self.confidence,
                iou=0.6,
                verbose=False,
                classes=all_classes,
                max_det=max_det
            )[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append([[x1, y1, w, h], conf, cls_id])
        else:
            # Dual model
            import config
            max_det_person = max(5, max_det // 2)
            max_det_vehicle = max(5, max_det // 2)
            
            results_person = self.model_person(
                inference_frame,
                conf=self.confidence,
                iou=0.6,
                verbose=False,
                classes=[config.PERSON_CLASS],
                max_det=max_det_person
            )[0]
            
            for box in results_person.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor
                conf = float(box.conf[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append([[x1, y1, w, h], conf, config.PERSON_CLASS])
            
            results_vehicle = self.model_vehicle(
                inference_frame,
                conf=self.confidence,
                iou=0.6,
                verbose=False,
                classes=config.VEHICLE_CLASSES,
                max_det=max_det_vehicle
            )[0]
            
            for box in results_vehicle.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                detections.append([[x1, y1, w, h], conf, cls_id])
        
        return detections
    
    def process_frame_threaded(self, frame: np.ndarray, resize_scale: int = 100, max_det: int = 20) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with parallel inference and drawing
        """
        # Start threads if not running
        if not self.running:
            self.start_threads()
        
        # Periodic cleanup
        self.frame_counter += 1
        if self.frame_counter % self.cleanup_interval == 0:
            if hasattr(self.tracker, 'tracks'):
                if len(self.tracker.tracks) > 30:
                    self.tracker.tracks = self.tracker.tracks[-30:]
            
            if len(self.trails) > 30:
                trail_ids = list(self.trails.keys())
                for old_id in trail_ids[:-30]:
                    del self.trails[old_id]
            
            if self.frame_counter % 500 == 0:
                import gc
                gc.collect()
        
        # Submit frame for inference (non-blocking)
        try:
            self.inference_queue.put((frame.copy(), resize_scale, max_det), block=False)
        except queue.Full:
            pass  # Skip if queue full
        
        # Use last detections or wait for new ones
        detections = self.last_detections if self.last_detections else []
        
        # Filter overlapping
        detections = self._filter_overlapping_detections(detections)
        
        # Tracking (main thread)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Drawing (main thread - fast)
        processed_frame = self._draw_detections(frame, tracks)
        
        # Update statistics
        stats = self._update_statistics(tracks)
        
        return processed_frame, stats

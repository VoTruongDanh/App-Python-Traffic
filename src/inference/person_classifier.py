"""
Person Classifier - Phân biệt người đi bộ vs người trên xe
Sử dụng IOU overlap và velocity analysis
"""
import numpy as np
from typing import Dict, List, Tuple


class PersonClassifier:
    """
    Classify person as Pedestrian or Rider based on:
    1. IOU overlap with vehicles
    2. Movement velocity
    """
    
    def __init__(self, iou_threshold=0.3, velocity_threshold=15.0):
        """
        Args:
            iou_threshold: IOU threshold to consider person on vehicle
            velocity_threshold: Speed threshold (pixels/frame) to classify as rider
        """
        self.iou_threshold = iou_threshold
        self.velocity_threshold = velocity_threshold
        
        # Track history for velocity calculation
        self.track_history = {}  # {track_id: [(x, y, frame_num), ...]}
        self.max_history = 10  # Keep last 10 positions
        
        # Classification history for smoothing and tracking
        self.classification_history = {}  # {track_id: [classifications]}
        self.classification_window = 5  # Smooth over last 5 frames
        
        # Track first classification for counting
        self.first_classification = {}  # {track_id: "Pedestrian" or "Rider"}
        
    def classify_person(self, person_bbox: List[float], person_track_id: int, 
                       vehicle_bboxes: List[Tuple[List[float], int]], 
                       frame_num: int) -> str:
        """
        Classify person as Pedestrian or Rider with smoothing
        
        Args:
            person_bbox: [x1, y1, x2, y2] of person
            person_track_id: Track ID of person
            vehicle_bboxes: List of (bbox, vehicle_class_id) for vehicles
            frame_num: Current frame number
            
        Returns:
            "Pedestrian", "Rider", or "Driver"
        """
        # Get raw classification
        raw_classification = self._classify_raw(person_bbox, person_track_id, vehicle_bboxes, frame_num)
        
        # Update classification history
        if person_track_id not in self.classification_history:
            self.classification_history[person_track_id] = []
        
        self.classification_history[person_track_id].append(raw_classification)
        
        # Keep only recent history
        if len(self.classification_history[person_track_id]) > self.classification_window:
            self.classification_history[person_track_id].pop(0)
        
        # Smooth classification using majority vote
        history = self.classification_history[person_track_id]
        
        # Count occurrences
        pedestrian_count = history.count("Pedestrian")
        rider_count = history.count("Rider")
        driver_count = history.count("Driver")
        
        # Majority vote
        if rider_count >= pedestrian_count and rider_count >= driver_count:
            smoothed = "Rider"
        elif driver_count >= pedestrian_count:
            smoothed = "Driver"
        else:
            smoothed = "Pedestrian"
        
        # Store first classification for this track (for counting)
        if person_track_id not in self.first_classification:
            self.first_classification[person_track_id] = smoothed
        
        return smoothed
    
    def _classify_raw(self, person_bbox: List[float], person_track_id: int,
                     vehicle_bboxes: List[Tuple[List[float], int]], 
                     frame_num: int) -> str:
        """
        Raw classification without smoothing
        """
        # Method 1: Check IOU overlap with vehicles
        max_iou = 0.0
        overlapping_vehicle = None
        
        for vehicle_bbox, vehicle_class in vehicle_bboxes:
            iou = self._calculate_iou(person_bbox, vehicle_bbox)
            if iou > max_iou:
                max_iou = iou
                overlapping_vehicle = vehicle_class
        
        # If significant overlap with vehicle
        if max_iou > self.iou_threshold:
            if overlapping_vehicle in [1, 3]:  # Bicycle or Motorcycle
                return "Rider"
            elif overlapping_vehicle in [2, 5, 7]:  # Car, Bus, Truck
                return "Driver"
            else:
                return "Rider"  # Default for unknown vehicles
        
        # Method 2: Check velocity
        velocity = self._calculate_velocity(person_bbox, person_track_id, frame_num)
        
        if velocity is not None and velocity > self.velocity_threshold:
            return "Rider"  # Moving fast, likely on vehicle
        
        # Default: Pedestrian
        return "Pedestrian"
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IOU between two bounding boxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
            
        Returns:
            IOU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_velocity(self, bbox: List[float], track_id: int, frame_num: int) -> float:
        """
        Calculate velocity of person based on track history
        
        Args:
            bbox: [x1, y1, x2, y2]
            track_id: Track ID
            frame_num: Current frame number
            
        Returns:
            Velocity in pixels/frame, or None if not enough history
        """
        # Get center point
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Update history
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append((center_x, center_y, frame_num))
        
        # Keep only recent history
        if len(self.track_history[track_id]) > self.max_history:
            self.track_history[track_id].pop(0)
        
        # Need at least 3 points to calculate velocity
        if len(self.track_history[track_id]) < 3:
            return None
        
        # Calculate average velocity over last N frames
        history = self.track_history[track_id]
        velocities = []
        
        for i in range(1, len(history)):
            x1, y1, f1 = history[i-1]
            x2, y2, f2 = history[i]
            
            if f2 - f1 > 0:
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                velocity = distance / (f2 - f1)
                velocities.append(velocity)
        
        if not velocities:
            return None
        
        return np.mean(velocities)
    
    def get_first_classification(self, track_id: int) -> str:
        """
        Get the first (initial) classification for a track
        Used for counting to avoid double-counting when classification changes
        
        Args:
            track_id: Track ID
            
        Returns:
            First classification ("Pedestrian", "Rider", or "Driver")
            Returns "Pedestrian" if not found
        """
        return self.first_classification.get(track_id, "Pedestrian")
    
    def cleanup_old_tracks(self, active_track_ids: set):
        """
        Remove history for tracks that no longer exist
        
        Args:
            active_track_ids: Set of currently active track IDs
        """
        old_ids = set(self.track_history.keys()) - active_track_ids
        for old_id in old_ids:
            del self.track_history[old_id]
        
        # Also cleanup classification history
        old_ids = set(self.classification_history.keys()) - active_track_ids
        for old_id in old_ids:
            del self.classification_history[old_id]
        
        old_ids = set(self.first_classification.keys()) - active_track_ids
        for old_id in old_ids:
            del self.first_classification[old_id]

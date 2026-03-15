import numpy as np
from scipy.optimize import linear_sum_assignment

class TrackerTrack:
    """Mock track class to mimic DeepSort track interface"""
    def __init__(self, track_id, bbox, cls_id, conf):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.det_class = cls_id
        self.conf = conf
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def to_ltrb(self):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]]

    def is_confirmed(self):
        return self.hits >= 1  # Simple tracker confirms immediately or after 1 hit

class SimpleTracker:
    """
    A simplified IOU-based tracker for high performance.
    replaces DeepSort when speed is priority.
    """
    def __init__(self, max_age=30, iou_threshold=0.4):  # Tăng từ 0.3 → 0.4
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update_tracks(self, detections, frame=None):
        """
        Args:
            detections: List of [[x1, y1, w, h], conf, cls_id]
            frame: Unused, kept for API compatibility with VideoProcessor
        """
        self.frame_count += 1
        
        # Format detections for IOU matching
        # detections list: [ [[x,y,w,h], conf, cls_id], ... ]
        
        # Predict: Just assume constant velocity or static (here static for simplicity/speed)
        # For a truly simple tracker, we just match to previous frame boxes.
        
        updated_tracks = []
        unmatched_dets = []
        
        if len(self.tracks) == 0:
            for det in detections:
                bbox, conf, cls_id = det
                self._init_track(bbox, cls_id, conf)
            return self.tracks

        # Match existing tracks to detections using IOU
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                det_bbox = det[0]
                cost_matrix[t, d] = 1.0 - self._iou(track.bbox, det_bbox)

        # Solving the assignment problem
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        used_rows = set(row_inds)
        used_cols = set(col_inds)

        # Update matched tracks
        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] > (1.0 - self.iou_threshold):
                # Too far apart, treat as unmatched
                used_rows.remove(r)
                used_cols.remove(c)
                continue
            
            track = self.tracks[r]
            det = detections[c]
            bbox, conf, cls_id = det
            
            track.bbox = bbox
            track.conf = conf
            track.det_class = cls_id # Update class just in case
            track.time_since_update = 0
            track.hits += 1
            track.hit_streak += 1
            track.age += 1

        # Create new tracks for unmatched detections
        for d, det in enumerate(detections):
            if d not in used_cols:
                bbox, conf, cls_id = det
                self._init_track(bbox, cls_id, conf)

        # Mark lost tracks
        new_tracks = []
        for t, track in enumerate(self.tracks):
            if t not in used_rows:
                track.time_since_update += 1
                track.hit_streak = 0
            
            # Remove dead tracks
            if track.time_since_update <= self.max_age:
                new_tracks.append(track)
        
        self.tracks = new_tracks
        return self.tracks

    def _init_track(self, bbox, cls_id, conf):
        self.tracks.append(TrackerTrack(self.next_id, bbox, cls_id, conf))
        self.next_id += 1

    def _iou(self, bbox1, bbox2):
        """
        Calculate IOU between two bounding boxes [x, y, w, h]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1+w1, x2+w2)
        yy2 = min(y1+h1, y2+h2)

        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)

        inter_area = w * h
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        
        if union_area <= 0:
            return 0
        return inter_area / union_area

"""
SORT: Simple Online and Realtime Tracking
Fast tracking algorithm using Kalman Filter + Hungarian Algorithm
Faster than DeepSORT, more stable than Simple tracker
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes
    State: [x, y, s, r, vx, vy, vs]
    - x, y: center position
    - s: scale (area)
    - r: aspect ratio
    - vx, vy, vs: velocities
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with detection bbox
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """
        Update tracker with new detection
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        """
        Predict next state
        
        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """
        Get current bbox
        
        Returns:
            Current bbox [x1, y1, x2, y2]
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """
        Convert [x1,y1,x2,y2] to [x,y,s,r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        """
        Convert [x,y,s,r] to [x1,y1,x2,y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


class SORTTracker:
    """
    SORT Tracker - Simple Online and Realtime Tracking
    """
    
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of [bbox, conf, cls] where bbox is [[x1,y1,w,h]]
            
        Returns:
            List of active tracks with format compatible with DeepSORT
        """
        self.frame_count += 1
        
        # Convert detections to [x1, y1, x2, y2, conf, cls]
        dets = []
        for det in detections:
            bbox, conf, cls = det
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 6))
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.det_class = int(dets[i, 5])  # Store class
            trk.det_conf = dets[i, 4]  # Store confidence
            self.trackers.append(trk)
            
        # Return active tracks in DeepSORT-compatible format
        tracks = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                # Create track object compatible with DeepSORT
                class Track:
                    def __init__(self, track_id, det_class, det_conf, bbox):
                        self.track_id = track_id
                        self.det_class = det_class
                        self.det_conf = det_conf
                        self._bbox = bbox
                    
                    def to_ltrb(self):
                        return self._bbox
                    
                    def is_confirmed(self):
                        return True
                
                track = Track(trk.id, getattr(trk, 'det_class', 0), getattr(trk, 'det_conf', 1.0), d)
                tracks.append(track)
                
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return tracks
    
    @staticmethod
    def _iou_batch(bb_test, bb_gt):
        """
        Compute IOU between two sets of boxes
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Match detections to trackers using Hungarian algorithm
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = self._iou_batch(detections[:, :4], trackers[:, :4])
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update_tracks(self, detections, frame=None):
        """
        Wrapper method compatible with DeepSORT interface
        """
        return self.update(detections)

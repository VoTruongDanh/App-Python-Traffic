"""
ByteTrack-style tracker (lightweight implementation)
Two-stage association with high/low confidence detections for stable IDs.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


class _ByteTrackTrack:
    def __init__(self, track_id, ltrb, det_class=0, det_conf=0.0):
        self.track_id = int(track_id)
        self._bbox = np.asarray(ltrb, dtype=float)
        self.det_class = int(det_class)
        self.det_conf = float(det_conf)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1

    def update(self, ltrb, det_class, det_conf):
        self._bbox = np.asarray(ltrb, dtype=float)
        self.det_class = int(det_class)
        self.det_conf = float(det_conf)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def age_one(self):
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0

    def to_ltrb(self):
        return self._bbox

    def is_confirmed(self):
        return self.hits >= 1

    def get_velocity(self):
        return 0.0, 0.0


class ByteTracker:
    """ByteTrack-inspired tracker with high/low confidence matching."""

    def __init__(
        self,
        max_age=20,
        min_hits=2,
        match_iou_threshold=0.25,
        high_conf_threshold=0.5,
        low_conf_threshold=0.15,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.match_iou_threshold = float(match_iou_threshold)
        self.high_conf_threshold = float(high_conf_threshold)
        self.low_conf_threshold = float(low_conf_threshold)
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self._last_ts = time.monotonic()

    @staticmethod
    def _iou_matrix(track_boxes, det_boxes):
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_boxes), len(det_boxes)), dtype=float)

        tb = np.asarray(track_boxes, dtype=float)
        db = np.asarray(det_boxes, dtype=float)

        xx1 = np.maximum(tb[:, None, 0], db[None, :, 0])
        yy1 = np.maximum(tb[:, None, 1], db[None, :, 1])
        xx2 = np.minimum(tb[:, None, 2], db[None, :, 2])
        yy2 = np.minimum(tb[:, None, 3], db[None, :, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        a = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
        b = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
        union = np.maximum(a[:, None] + b[None, :] - inter, 1e-6)
        return inter / union

    def _associate(self, track_indices, detections, iou_threshold):
        if len(track_indices) == 0 or len(detections) == 0:
            return [], list(track_indices), list(range(len(detections)))

        track_boxes = [self.tracks[i].to_ltrb() for i in track_indices]
        det_boxes = [d[0] for d in detections]
        iou = self._iou_matrix(track_boxes, det_boxes)

        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)

        matches = []
        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_threshold:
                matches.append((track_indices[r], c))
                used_rows.add(r)
                used_cols.add(c)

        unmatched_tracks = [track_indices[i] for i in range(len(track_indices)) if i not in used_rows]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_cols]
        return matches, unmatched_tracks, unmatched_dets

    @staticmethod
    def _to_ltrb(det):
        bbox, conf, cls_id = det
        x1, y1, w, h = bbox
        return [float(x1), float(y1), float(x1 + w), float(y1 + h)], float(conf), int(cls_id)

    def update_tracks(self, detections, frame=None, frame_timestamp: float = None):
        _ = frame
        now = frame_timestamp if frame_timestamp else time.monotonic()
        self._last_ts = now
        self.frame_count += 1

        parsed = [self._to_ltrb(det) for det in detections]
        high = [d for d in parsed if d[1] >= self.high_conf_threshold]
        low = [d for d in parsed if self.low_conf_threshold <= d[1] < self.high_conf_threshold]

        for trk in self.tracks:
            trk.age_one()

        all_track_indices = list(range(len(self.tracks)))
        matches_high, unmatched_tracks, unmatched_high = self._associate(
            all_track_indices,
            high,
            self.match_iou_threshold,
        )

        for track_idx, det_idx in matches_high:
            bbox, conf, cls_id = high[det_idx]
            self.tracks[track_idx].update(bbox, cls_id, conf)

        if low and unmatched_tracks:
            matches_low, unmatched_tracks, _ = self._associate(
                unmatched_tracks,
                low,
                max(0.15, self.match_iou_threshold - 0.05),
            )
            for track_idx, det_idx in matches_low:
                bbox, conf, cls_id = low[det_idx]
                self.tracks[track_idx].update(bbox, cls_id, conf)

        for det_idx in unmatched_high:
            bbox, conf, cls_id = high[det_idx]
            self.tracks.append(_ByteTrackTrack(self.next_id, bbox, cls_id, conf))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        outputs = []
        for trk in self.tracks:
            if trk.time_since_update == 0 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(trk)

        return outputs

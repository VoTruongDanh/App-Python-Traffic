"""
ROI (Region of Interest) Manager
Quản lý vùng quan tâm và kiểm tra objects
"""
import cv2
import numpy as np
from typing import List, Tuple


class ROIManager:
    """Quản lý ROI polygon và kiểm tra objects"""
    
    def __init__(self):
        self.roi_points = []  # List of (x, y) points
        self.roi_polygon = None  # numpy array for cv2
        self.threshold = 0.5  # 50% object phải nằm trong ROI
        self.visible = True  # ROI overlay visibility
        
    def set_points(self, points: List[Tuple[int, int]]):
        """Set ROI points"""
        self.roi_points = points
        if len(points) >= 3:
            self.roi_polygon = np.array(points, dtype=np.int32)
        else:
            self.roi_polygon = None
    
    def add_point(self, x: int, y: int):
        """Add a point to ROI"""
        self.roi_points.append((x, y))
        if len(self.roi_points) >= 3:
            self.roi_polygon = np.array(self.roi_points, dtype=np.int32)
    
    def clear(self):
        """Clear ROI"""
        self.roi_points = []
        self.roi_polygon = None
    
    def set_threshold(self, threshold: float):
        """Set threshold (0.0 - 1.0)"""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def is_active(self) -> bool:
        """Check if ROI is active"""
        return self.roi_polygon is not None and len(self.roi_points) >= 3
    
    def is_object_in_roi(self, bbox: List[float]) -> bool:
        """
        Kiểm tra object có trong ROI không
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            True nếu >= threshold% của object nằm trong ROI
        """
        if not self.is_active():
            return True  # Nếu không có ROI, chấp nhận tất cả
        
        x1, y1, x2, y2 = bbox
        
        # Tạo mask cho bbox
        bbox_width = int(x2 - x1)
        bbox_height = int(y2 - y1)
        
        if bbox_width <= 0 or bbox_height <= 0:
            return False
        
        # Sample points trong bbox
        sample_points = []
        step = 5  # Sample mỗi 5 pixels
        for y in range(int(y1), int(y2), step):
            for x in range(int(x1), int(x2), step):
                sample_points.append((x, y))
        
        if not sample_points:
            # Fallback: check center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            result = cv2.pointPolygonTest(self.roi_polygon, (center_x, center_y), False)
            return result >= 0
        
        # Đếm số points trong ROI
        points_in_roi = 0
        for point in sample_points:
            result = cv2.pointPolygonTest(self.roi_polygon, point, False)
            if result >= 0:
                points_in_roi += 1
        
        # Tính % trong ROI
        ratio = points_in_roi / len(sample_points)
        return ratio >= self.threshold
    
    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255), thickness=2):
        """
        Vẽ ROI lên frame
        
        Args:
            frame: Frame để vẽ
            color: Màu (B, G, R)
            thickness: Độ dày đường vẽ
            
        Returns:
            Frame đã vẽ (hoặc không vẽ nếu visible=False)
        """
        if not self.is_active() or not self.visible:
            return frame
        
        # Vẽ polygon
        cv2.polylines(frame, [self.roi_polygon], True, color, thickness)
        
        # Vẽ fill semi-transparent
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi_polygon], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Vẽ các điểm
        for point in self.roi_points:
            cv2.circle(frame, point, 5, color, -1)
        
        # Vẽ text thông tin
        text = f"ROI: {len(self.roi_points)} points | Threshold: {self.threshold*100:.0f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        return frame
    
    def get_config(self) -> dict:
        """Get ROI configuration"""
        return {
            'points': self.roi_points,
            'threshold': self.threshold,
            'active': self.is_active()
        }
    
    def load_config(self, config: dict):
        """Load ROI configuration"""
        if 'points' in config:
            self.set_points(config['points'])
        if 'threshold' in config:
            self.set_threshold(config['threshold'])

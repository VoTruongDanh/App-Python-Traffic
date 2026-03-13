"""
FPS Booster - Các kỹ thuật tăng FPS dễ implement
1. Dynamic Resolution Scaling
2. Adaptive Frame Skip
3. Smart Detection Throttling
"""
import time
from typing import Optional


class DynamicResolutionScaler:
    """
    Tự động điều chỉnh resolution dựa trên FPS hiện tại
    Mục tiêu: Duy trì FPS ổn định
    """
    
    def __init__(self, target_fps: int = 30, min_scale: int = 50, max_scale: int = 100):
        """
        Args:
            target_fps: FPS mục tiêu
            min_scale: Resolution tối thiểu (%)
            max_scale: Resolution tối đa (%)
        """
        self.target_fps = target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.current_scale = max_scale
        
        # Smoothing
        self.fps_history = []
        self.history_size = 10
        
    def adjust(self, current_fps: float) -> int:
        """
        Điều chỉnh resolution scale dựa trên FPS
        
        Args:
            current_fps: FPS hiện tại
            
        Returns:
            Resolution scale (50-100%)
        """
        # Update history
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.history_size:
            self.fps_history.pop(0)
        
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Adjust scale
        if avg_fps < self.target_fps - 5:
            # FPS quá thấp, giảm resolution
            self.current_scale = max(self.min_scale, self.current_scale - 5)
        elif avg_fps > self.target_fps + 5:
            # FPS cao, tăng resolution
            self.current_scale = min(self.max_scale, self.current_scale + 5)
        
        return self.current_scale
    
    def reset(self):
        """Reset về max scale"""
        self.current_scale = self.max_scale
        self.fps_history = []


class AdaptiveFrameSkipper:
    """
    Tự động skip frames khi FPS thấp
    Trade-off: Có thể miss objects nhanh
    """
    
    def __init__(self, target_fps: int = 30, max_skip: int = 2):
        """
        Args:
            target_fps: FPS mục tiêu
            max_skip: Số frames tối đa có thể skip
        """
        self.target_fps = target_fps
        self.max_skip = max_skip
        self.skip_count = 0
        
        # Smoothing
        self.fps_history = []
        self.history_size = 10
        
    def should_process(self, current_fps: float, frame_count: int) -> bool:
        """
        Quyết định có nên xử lý frame này không
        
        Args:
            current_fps: FPS hiện tại
            frame_count: Frame number
            
        Returns:
            True nếu nên xử lý frame này
        """
        # Update history
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.history_size:
            self.fps_history.pop(0)
        
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Adjust skip count
        if avg_fps < self.target_fps - 5:
            # FPS thấp, tăng skip
            self.skip_count = min(self.max_skip, self.skip_count + 1)
        elif avg_fps > self.target_fps + 5:
            # FPS cao, giảm skip
            self.skip_count = max(0, self.skip_count - 1)
        
        # Decide whether to process
        if self.skip_count == 0:
            return True  # Không skip
        
        return frame_count % (self.skip_count + 1) == 0
    
    def reset(self):
        """Reset skip count"""
        self.skip_count = 0
        self.fps_history = []


class SmartDetectionThrottler:
    """
    Giảm tần suất detection khi có ít thay đổi
    Sử dụng motion detection để quyết định
    """
    
    def __init__(self, min_interval: float = 0.1, max_interval: float = 1.0):
        """
        Args:
            min_interval: Interval tối thiểu giữa các detection (seconds)
            max_interval: Interval tối đa (seconds)
        """
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = min_interval
        
        self.last_detection_time = 0
        self.motion_threshold = 0.05  # 5% pixels changed
        
    def should_detect(self, motion_score: Optional[float] = None) -> bool:
        """
        Quyết định có nên chạy detection không
        
        Args:
            motion_score: Tỷ lệ pixels thay đổi (0-1), None = always detect
            
        Returns:
            True nếu nên chạy detection
        """
        current_time = time.time()
        elapsed = current_time - self.last_detection_time
        
        # Adjust interval based on motion
        if motion_score is not None:
            if motion_score > self.motion_threshold:
                # Nhiều chuyển động, detect thường xuyên
                self.current_interval = self.min_interval
            else:
                # Ít chuyển động, detect ít hơn
                self.current_interval = min(self.max_interval, self.current_interval * 1.1)
        
        # Check if should detect
        if elapsed >= self.current_interval:
            self.last_detection_time = current_time
            return True
        
        return False
    
    def reset(self):
        """Reset interval"""
        self.current_interval = self.min_interval
        self.last_detection_time = 0


class FPSBooster:
    """
    Tổng hợp các kỹ thuật tăng FPS
    """
    
    def __init__(self, target_fps: int = 30, enable_all: bool = True):
        """
        Args:
            target_fps: FPS mục tiêu
            enable_all: Bật tất cả optimizations
        """
        self.target_fps = target_fps
        
        # Components
        self.resolution_scaler = DynamicResolutionScaler(target_fps) if enable_all else None
        self.frame_skipper = AdaptiveFrameSkipper(target_fps, max_skip=1) if enable_all else None
        self.detection_throttler = SmartDetectionThrottler() if enable_all else None
        
        # Stats
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'detections_run': 0,
            'detections_skipped': 0,
            'avg_resolution': 100
        }
    
    def get_resolution_scale(self, current_fps: float) -> int:
        """
        Lấy resolution scale nên dùng
        
        Args:
            current_fps: FPS hiện tại
            
        Returns:
            Resolution scale (50-100%)
        """
        if self.resolution_scaler:
            scale = self.resolution_scaler.adjust(current_fps)
            self.stats['avg_resolution'] = scale
            return scale
        return 100
    
    def should_process_frame(self, current_fps: float, frame_count: int) -> bool:
        """
        Quyết định có nên xử lý frame này không
        
        Args:
            current_fps: FPS hiện tại
            frame_count: Frame number
            
        Returns:
            True nếu nên xử lý
        """
        if self.frame_skipper:
            should_process = self.frame_skipper.should_process(current_fps, frame_count)
            if should_process:
                self.stats['frames_processed'] += 1
            else:
                self.stats['frames_skipped'] += 1
            return should_process
        
        self.stats['frames_processed'] += 1
        return True
    
    def should_run_detection(self, motion_score: Optional[float] = None) -> bool:
        """
        Quyết định có nên chạy detection không
        
        Args:
            motion_score: Tỷ lệ pixels thay đổi (0-1)
            
        Returns:
            True nếu nên chạy detection
        """
        if self.detection_throttler:
            should_detect = self.detection_throttler.should_detect(motion_score)
            if should_detect:
                self.stats['detections_run'] += 1
            else:
                self.stats['detections_skipped'] += 1
            return should_detect
        
        self.stats['detections_run'] += 1
        return True
    
    def get_stats(self) -> dict:
        """
        Lấy statistics
        
        Returns:
            Dict với stats
        """
        total_frames = self.stats['frames_processed'] + self.stats['frames_skipped']
        total_detections = self.stats['detections_run'] + self.stats['detections_skipped']
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'frames_skipped': self.stats['frames_skipped'],
            'frame_skip_rate': f"{self.stats['frames_skipped'] / max(1, total_frames) * 100:.1f}%",
            'detections_run': self.stats['detections_run'],
            'detections_skipped': self.stats['detections_skipped'],
            'detection_skip_rate': f"{self.stats['detections_skipped'] / max(1, total_detections) * 100:.1f}%",
            'avg_resolution': f"{self.stats['avg_resolution']}%"
        }
    
    def reset(self):
        """Reset tất cả"""
        if self.resolution_scaler:
            self.resolution_scaler.reset()
        if self.frame_skipper:
            self.frame_skipper.reset()
        if self.detection_throttler:
            self.detection_throttler.reset()
        
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'detections_run': 0,
            'detections_skipped': 0,
            'avg_resolution': 100
        }


# Example usage
if __name__ == "__main__":
    print("FPS Booster - Test")
    print("=" * 60)
    
    booster = FPSBooster(target_fps=30, enable_all=True)
    
    # Simulate processing
    for frame_count in range(100):
        current_fps = 20 + (frame_count % 10)  # Simulate varying FPS
        
        # Get resolution scale
        scale = booster.get_resolution_scale(current_fps)
        
        # Check if should process frame
        if booster.should_process_frame(current_fps, frame_count):
            # Check if should run detection
            motion_score = 0.1 if frame_count % 5 == 0 else 0.01
            if booster.should_run_detection(motion_score):
                print(f"Frame {frame_count}: Process + Detect (scale={scale}%, fps={current_fps:.1f})")
            else:
                print(f"Frame {frame_count}: Process only (scale={scale}%, fps={current_fps:.1f})")
        else:
            print(f"Frame {frame_count}: Skip (fps={current_fps:.1f})")
    
    print()
    print("=" * 60)
    print("Statistics:")
    print("=" * 60)
    stats = booster.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

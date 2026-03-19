"""Low-latency frame buffer that always keeps only the newest decoded frame."""
import threading
import time
import cv2


class LatestFrameBuffer:
    """Drop-in replacement for cap.read() that always returns the newest frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._alive:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._ts = time.monotonic()
            else:
                time.sleep(0.002)

    def read(self):
        """Return (ret, frame, timestamp)."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._ts

    def release(self):
        self._alive = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._cap.release()

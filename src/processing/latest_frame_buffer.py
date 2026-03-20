"""Low-latency frame buffer that always keeps only the newest decoded frame."""
import threading
import time
import cv2


class LatestFrameBuffer:
    """Drop-in replacement for cap.read() that always returns the newest frame."""

    def __init__(self, cap: cv2.VideoCapture, stale_timeout_sec: float = 0.8, drain_grabs: int = 2):
        self._cap = cap
        self._frame = None
        self._ts = 0.0
        self._stale_timeout_sec = max(0.2, float(stale_timeout_sec))
        self._drain_grabs = max(0, int(drain_grabs))
        self._last_error = ""
        self._error_count = 0
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _read_latest_frame(self):
        """Read one frame and opportunistically drain buffered stale frames."""
        ret, frame = self._cap.read()
        if not ret:
            return False, None

        if self._drain_grabs <= 0:
            return True, frame

        grabbed = 0
        for _ in range(self._drain_grabs):
            if not self._cap.grab():
                break
            grabbed += 1

        if grabbed > 0:
            ret2, frame2 = self._cap.retrieve()
            if ret2:
                frame = frame2

        return True, frame

    def _capture_loop(self):
        while self._alive:
            try:
                ret, frame = self._read_latest_frame()
            except cv2.error as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._error_count += 1
                time.sleep(0.01)
                continue
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._error_count += 1
                time.sleep(0.01)
                continue

            if ret:
                with self._lock:
                    self._frame = frame
                    self._ts = time.monotonic()
                    self._error_count = 0
                    self._last_error = ""
            else:
                time.sleep(0.002)

    def read(self):
        """Return (ret, frame, timestamp)."""
        with self._lock:
            frame = self._frame
            ts = self._ts
            last_error = self._last_error

        if frame is None or ts <= 0.0:
            return False, None, 0.0

        if not self._thread.is_alive():
            if last_error:
                print(f"[WARN] LatestFrameBuffer capture thread stopped: {last_error}")
            return False, None, 0.0

        # If no fresh frame arrives for too long, force reconnect path upstream.
        if (time.monotonic() - ts) > self._stale_timeout_sec:
            if last_error:
                print(f"[WARN] LatestFrameBuffer stale frame: {last_error}")
            return False, None, 0.0

        # Return the latest immutable frame reference to avoid per-frame copy overhead.
        return True, frame, ts

    def release(self):
        self._alive = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        try:
            self._cap.release()
        except Exception:
            pass

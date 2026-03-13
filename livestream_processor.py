"""
Livestream processing module - Real-time detection from streams
"""
import cv2
import streamlit as st
import numpy as np
from video_processor import VideoProcessor
import config
import time


import threading
import queue

class ThreadedCamera:
    """Loại bỏ độ trễ IO bằng cách đọc frame trong luồng riêng"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer
        self.q = queue.Queue(maxsize=1)
        self.status = False
        self.reading = True
        
        if self.cap.isOpened():
            self.status = True
            # Start thread
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.reading:
            if not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            if not ret:
                self.status = False
                break
            
            # Keep only latest frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def read(self):
        if not self.q.empty():
            return True, self.q.get()
        return self.status, None

    def release(self):
        self.reading = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

def process_livestream(stream_url, stream_type, confidence, frame_skip, resize_scale, tracker_choice='DeepSort'):
    """
    Xử lý livestream real-time (Optimized v2)
    """
    from model_loader import load_yolo_models, initialize_tracker
    
    st.toast("🔄 Đang kết nối...", icon="🔄")
    
    # Load models
    model_person, model_vehicle = load_yolo_models()
    tracker = initialize_tracker(tracker_choice)
    processor = VideoProcessor(model_person, model_vehicle, tracker)
    processor.set_confidence(confidence)
    
    # Get direct URL if YouTube
    start_url = stream_url
    if stream_type == "YouTube Live":
        try:
            import yt_dlp
            # Optimize: Get lowest decent quality for speed (480p/720p)
            ydl_opts = {
                'format': 'best[height<=720][ext=mp4]/best[height<=720]',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(stream_url, download=False)
                start_url = info['url']
                st.toast(f"✅ Stream: {info.get('title', 'Unknown')[:30]}...", icon="✅")
        except Exception:
            # Fallback
            pass

    # Threaded Capture
    try:
        cap = ThreadedCamera(start_url)
        if not cap.status:
            st.error("❌ Không thể kết nối stream")
            return
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")
        return

    st.toast("🚀 Đang chạy tối ưu hóa...", icon="🚀")
    
    # Layout UI
    video_placeholder = st.empty()
    st.divider()
    stats_placeholder = st.empty()
    
    st.session_state['livestream_running'] = True
    
    frame_cnt = 0
    fps_start = time.time()
    fps_val = 0
    frame_skip_counter = 0
    
    try:
        while st.session_state.get('livestream_running', False):
            # Non-blocking read
            ret, frame = cap.read()
            
            if not ret or frame is None:
                if not cap.status:
                    st.toast("⚠️ Mất kết nối stream", icon="⚠️")
                    break
                time.sleep(0.005) # Yield CPU
                continue
            
            # 1. Frame Skipping Logic
            if frame_skip > 0:
                frame_skip_counter += 1
                if frame_skip_counter <= frame_skip:
                    continue
                frame_skip_counter = 0

            # 2. Hard Resize for Performance (Max logical width 640 for Streamlit display)
            # Resize input frame based on user setting
            if resize_scale != 100:
                h, w = frame.shape[:2]
                new_w = int(w * resize_scale / 100)
                new_h = int(h * resize_scale / 100)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # 3. Process
            processed_frame, stats = processor.process_frame(frame)
            
            # 4. Display Optimization: Ensure max display width 720p to save bandwidth
            disp_h, disp_w = processed_frame.shape[:2]
            if disp_w > 800:
                scale = 800 / disp_w
                processed_frame = cv2.resize(processed_frame, (800, int(disp_h * scale)))
                
            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            
            # FPS Calculation
            frame_cnt += 1
            if frame_cnt % 5 == 0:  # Update FPS every 5 processed frames
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    fps_val = 5 / elapsed
                fps_start = time.time()
                
                # Update stats rarely
                with stats_placeholder.container():
                    cols = st.columns(5)
                    cols[0].metric("📊 Objects", stats['total_objects'])
                    cols[1].metric("🔍 Classes", len(stats['class_counts']))
                    cols[2].metric("🚀 FPS", f"{fps_val:.1f}")
                    cols[3].metric("🤖 Model", st.session_state.get('model_choice', 'YOLOv3'))
                    cols[4].metric("🐇 Tracker", st.session_state.get('tracker_choice', 'DeepSort')[:6])
    
    finally:
        cap.release()
        st.session_state['livestream_running'] = False
        st.toast("⏹️ Stream stopped", icon="⏹️")

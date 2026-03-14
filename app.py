# """
# 🎥 Ứng dụng Xử lý Video với YOLOv3 + DeepSort
# Phát hiện và theo dõi Người & Xe trong video
# """
# import streamlit as st
# import os
# import sys
# import time
# import json
# from pathlib import Path

# # Import modules
# import config
# import utils
# from model_loader import load_yolo_models, initialize_tracker, display_model_info
# from video_processor import VideoProcessor
# from livestream_processor import process_livestream


# # =============================================================================
# # PAGE CONFIG
# # =============================================================================
# st.set_page_config(
#     page_title="Video Object Tracking",
#     page_icon="🎥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # =============================================================================
# # CUSTOM CSS
# # =============================================================================
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(90deg, #f6d365 0%, #4ade80 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .stButton>button {
#         width: 100%;
#         background: linear-gradient(90deg, #fbbf24 0%, #22c55e 100%);
#         color: white;
#         font-weight: bold;
#         border: none;
#         padding: 0.75rem;
#         border-radius: 0.5rem;
#     }
#     .stButton>button:hover {
#         background: linear-gradient(90deg, #f59e0b 0%, #16a34a 100%);
#         box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
#     }
#     .stats-box {
#         background: #f0f2f6;
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #fbbf24;
#     }
# </style>
# """, unsafe_allow_html=True)


# # =============================================================================
# # MAIN APP
# # =============================================================================
# def main():
#     # Header
#     st.markdown('<h1 class="main-header">Video Object Tracking</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Phát hiện & Theo dõi Người và Xe với YOLOv3 + DeepSort</p>', unsafe_allow_html=True)
    
#     # Tạo directories
#     utils.create_directories()
    
#     # Sidebar - Configuration
#     with st.sidebar:
#         st.header("Cấu hình")
        
#         # GPU/CPU Selection
#         st.subheader("Thiết bị xử lý")
        
#         # Check CUDA availability
#         import torch
#         cuda_available = torch.cuda.is_available()
        
#         if cuda_available:
#             gpu_name = torch.cuda.get_device_name(0)
#             st.success(f"GPU khả dụng: {gpu_name}")
#         else:
#             st.warning("GPU không khả dụng - chỉ có CPU")
        
#         # Toggle
#         use_gpu = st.checkbox(
#             "Sử dụng GPU",
#             value=config.USE_GPU,  # Use current config value
#             disabled=not cuda_available,
#             help="Bật để sử dụng GPU (nhanh hơn 3-5 lần). Tắt để dùng CPU."
#         )
        
#         # Detect change and force reload
#         if use_gpu != config.USE_GPU:
#             config.USE_GPU = use_gpu
#             st.cache_resource.clear()
#             st.warning(f"🔄 Đang chuyển sang {'GPU' if use_gpu else 'CPU'} mode...")
#             st.rerun()
        
#         # Display current mode
#         if config.USE_GPU and cuda_available:
#             st.success("✅ Đang sử dụng GPU mode")
#         else:
#             st.info("💻 Đang sử dụng CPU mode")
        
#         st.divider()
        
#         # YOLO Model Selection
#         st.subheader("⚙️ Model Selection")
        
#         model_choice = st.selectbox(
#             "YOLO Model",
#             options=["YOLOv3", "YOLOv8n", "YOLOv11n", "YOLOv26"],
#             index=1,  # Default to YOLOv8n for better performance
#             help="""
#             - **YOLOv3**: Cân bằng (5-6 FPS)
#             - **YOLOv8n**: Nhanh hơn 2-3x (~12-15 FPS) ⚡
#             - **YOLOv11n**: Nhanh nhất, mới nhất (~15-20 FPS) 🚀
#             - **YOLOv26**: Experimental (Beta) 🧪
#             """
#         )
        
#         # Store in session state
#         if 'model_choice' not in st.session_state or st.session_state['model_choice'] != model_choice:
#             st.session_state['model_choice'] = model_choice
#             st.cache_resource.clear()  # Clear cache to reload models
#             st.toast(f"🔄 Đã chọn {model_choice}", icon="🔄")
        
#         # Tracker Selection
#         tracker_choice = st.selectbox(
#             "Tracking Algorithm",
#             options=["DeepSort", "Simple (Fastest)"],
#             index=0,
#             help="""
#             - **DeepSort**: Chính xác hơn, dùng Deep Learning để nhận diện lại ID. (Chậm hơn)
#             - **Simple**: Rất nhanh, chỉ dựa vào vị trí (IOU). Tốt cho FPS cao.
#             """
#         )
        
#         if 'tracker_choice' not in st.session_state or st.session_state['tracker_choice'] != tracker_choice:
#             st.session_state['tracker_choice'] = tracker_choice
#             st.cache_resource.clear()
#             st.toast(f"🔄 Tracker: {tracker_choice}", icon="🔄")
        
#         # Display current active model & tracker
#         current_model = st.session_state.get('model_choice', 'YOLOv3')
#         current_tracker = st.session_state.get('tracker_choice', 'DeepSort')
        
#         col_m1, col_m2 = st.columns(2)
#         with col_m1:
#             if current_model == 'YOLOv3':
#                 st.info(f"YOLOv3")
#             elif current_model == 'YOLOv8n':
#                 st.success(f"YOLOv8n ⚡")
#             elif current_model == 'YOLOv26':
#                 st.warning(f"YOLOv26 🧪")
#             else:
#                 st.success(f"YOLOv11n 🚀")
        
#         with col_m2:
#             if current_tracker == 'DeepSort':
#                 st.info("DeepSort 🐢")
#             else:
#                 st.success("Simple 🐇")
        
#         st.divider()
        
#         # Confidence threshold
#         confidence = st.slider(
#             "Confidence Threshold",
#             min_value=config.MIN_CONFIDENCE,
#             max_value=config.MAX_CONFIDENCE,
#             value=config.DEFAULT_CONFIDENCE,
#             step=0.05,
#             help="Ngưỡng confidence cho detection (càng cao càng chính xác nhưng có thể bỏ sót)"
#         )
        
#         # Max duration
#         max_duration = st.slider(
#             "Giới hạn thời gian xử lý (giây)",
#             min_value=10,
#             max_value=120,
#             value=30,
#             step=10,
#             help="Giới hạn video để xử lý nhanh hơn"
#         )
        
#         st.divider()
        
#         # Performance optimization
#         st.subheader("Tối ưu hóa tốc độ")
        
#         # Frame skip
#         frame_skip = st.slider(
#             "Bỏ qua frames",
#             min_value=0,
#             max_value=10,
#             value=2,
#             step=1,
#             help="Chỉ xử lý mỗi N frame. Càng cao càng nhanh nhưng mất detail. Khuyến nghị: 2-3"
#         )
        
#         # Resize
#         resize_scale = st.selectbox(
#             "Thu nhỏ video",
#             options=[100, 75, 50, 25],
#             index=2,  # Default 50%
#             help="Thu nhỏ video để xử lý nhanh hơn. 50% = nhanh gấp 4 lần"
#         )
        
#         st.info(f"⚡ Tốc độ ước tính: **{(1 + frame_skip) * (100/resize_scale)**2:.0f}x** nhanh hơn")
        
#         st.divider()
        
#         # Model info
#         display_model_info()
        
#         st.divider()
        
#         # Instructions
#         with st.expander("Hướng dẫn sử dụng"):
#             st.markdown("""
#             **Bước 1:** Upload file video (MP4, AVI, MOV, WEBM)
            
#             **Bước 2:** Điều chỉnh confidence threshold nếu cần
            
#             **Bước 3:** Nhấn "Xử lý Video"
            
#             **Bước 4:** Xem kết quả và tải video đã xử lý
            
#             ⚡ **Lưu ý:** 
#             - GPU sẽ tăng tốc xử lý đáng kể
#             - Video càng dài càng mất nhiều thời gian
#             - Confidence thấp = nhiều detection hơn nhưng có thể sai
#             """)
    
#     # Main content - Full width layout
#     st.subheader("📹 Nguồn Video & Livestream")
    
#     # Tabs cho input sources - Full width
#     tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 Từ URL", "📡 Livestream"])
    
#     with tab1:
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             uploaded_file = st.file_uploader(
#                 "Chọn file video",
#                 type=config.SUPPORTED_FORMATS,
#                 help="Hỗ trợ: MP4, AVI, MOV, WEBM, MKV"
#             )
        
#         if uploaded_file:
#             col1, col2 = st.columns([3, 2])
            
#             with col1:
#                 # Lưu file tạm
#                 temp_input = utils.get_temp_filepath(prefix="input_", suffix=".mp4")
#                 with open(temp_input, 'wb') as f:
#                     f.write(uploaded_file.read())
                
#                 # Hiển thị video gốc
#                 st.video(temp_input)
            
#             with col2:
#                 # Hiển thị thông tin file
#                 file_size_mb = uploaded_file.size / (1024 * 1024)
#                 st.success(f"✅ **{uploaded_file.name}**")
#                 st.info(f"📦 Dung lượng: {file_size_mb:.2f} MB")
                
#                 # Lấy thông tin video
#                 video_info = utils.get_video_info(temp_input)
#                 if video_info:
#                     st.markdown("**📊 Thông tin:**")
#                     st.write(f"🖼️ {video_info['width']}x{video_info['height']}")
#                     st.write(f"🎬 {video_info['fps']} FPS")
#                     st.write(f"⏱️ {video_info['duration_seconds']:.1f}s")
#                     st.write(f"📹 {video_info['total_frames']} frames")
                
#                 # Store in session state
#                 st.session_state['input_video'] = temp_input
#                 st.session_state['video_info'] = video_info
                
#                 st.divider()
                
#                 # Nút xử lý ngay trong tab
#                 if st.button("🎬 Xử lý Video Ngay", use_container_width=True, type="primary"):
#                     process_video(
#                         temp_input,
#                         confidence,
#                         max_duration,
#                         frame_skip,
#                         resize_scale
#                     )
    
#     with tab2:
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             st.markdown("**Hỗ trợ:**")
#             st.markdown("- 🎥 YouTube (youtube.com, youtu.be)")
#             st.markdown("- 🔗 Direct links (.mp4, .avi, .mov, .webm)")
            
#             video_url = st.text_input(
#                 "Nhập URL video",
#                 placeholder="https://youtube.com/watch?v=... hoặc direct link",
#                 help="Paste link YouTube hoặc direct video link"
#             )
        
#         with col2:
#             if video_url:
#                 if st.button("⬇️ Tải video từ URL", use_container_width=True, type="primary"):
#                     with st.spinner("⏳ Đang tải video..."):
#                         temp_input = utils.get_temp_filepath(prefix="downloaded_", suffix=".mp4")
                        
#                         # Download video
#                         success = utils.download_video_from_url(video_url, temp_input)
                        
#                         if success:
#                             st.success("✅ Tải thành công!")
#                             st.session_state['input_video'] = temp_input
#                             st.session_state['video_info'] = utils.get_video_info(temp_input)
#                             st.rerun()
#                         else:
#                             st.error("❌ Không thể tải video")
        
#         # Display downloaded video
#         if 'input_video' in st.session_state and not uploaded_file:
#             st.divider()
#             col1, col2 = st.columns([3, 2])
            
#             with col1:
#                 st.video(st.session_state['input_video'])
            
#             with col2:
#                 video_info = st.session_state.get('video_info')
#                 if video_info:
#                     st.markdown("**📊 Thông tin:**")
#                     st.write(f"🖼️ {video_info['width']}x{video_info['height']}")
#                     st.write(f"🎬 {video_info['fps']} FPS")
#                     st.write(f"⏱️ {video_info['duration_seconds']:.1f}s")
                
#                 st.divider()
                
#                 if st.button("🎬 Xử lý Video", use_container_width=True, type="primary", key="process_url"):
#                     process_video(
#                         st.session_state['input_video'],
#                         confidence,
#                         max_duration,
#                         frame_skip,
#                         resize_scale
#                     )
    
#     with tab3:
#         st.markdown("**📡 Livestream Real-time Detection**")
        
#         col1, col2, col3 = st.columns([2, 2, 1])
        
#         with col1:
#             stream_type = st.radio(
#                 "Chọn nguồn:",
#                 ["YouTube Live", "RTSP/IP Camera", "Webcam"],
#                 horizontal=True
#             )
        
#         stream_url = None
        
#         with col2:
#             if stream_type == "YouTube Live":
#                 stream_url = st.text_input(
#                     "YouTube Live URL",
#                     placeholder="https://youtube.com/live/...",
#                     help="Nhập URL YouTube livestream"
#                 )
#             elif stream_type == "RTSP/IP Camera":
#                 stream_url = st.text_input(
#                     "RTSP URL",
#                     placeholder="rtsp://user:pass@ip:port/stream",
#                     help="Nhập URL RTSP của IP camera"
#                 )
#             else:  # Webcam
#                 webcam_id = st.number_input(
#                     "Webcam ID", 
#                     min_value=0, 
#                     max_value=10, 
#                     value=0,
#                     help="0 = webcam mặc định"
#                 )
#                 stream_url = str(webcam_id)
        
#         with col3:
#             st.write("")  # Spacing
#             st.write("")  # Spacing
#             if st.button("▶️ Bắt đầu", key="start_livestream", use_container_width=True, type="primary"):
#                 if stream_url:
#                     st.session_state['livestream_active'] = True
#                     st.session_state['stream_url'] = stream_url
#                     st.session_state['stream_type'] = stream_type
#                     st.rerun()
#                 else:
#                     st.warning("⚠️ Nhập URL")
            
#             if st.button("⏹️ Dừng", key="stop_livestream", use_container_width=True):
#                 if 'livestream_running' in st.session_state:
#                     st.session_state['livestream_running'] = False
#                     st.session_state['livestream_active'] = False
#                     st.success("✅ Đã dừng")
#                     st.rerun()
    
#     # Results section - Full width
#     st.divider()
    
#     # LIVESTREAM DISPLAY - Outside tabs, full width
#     if st.session_state.get('livestream_active') and not st.session_state.get('livestream_running'):
#         st.subheader("📡 Livestream Detection")
#         process_livestream(
#             st.session_state.get('stream_url'),
#             st.session_state.get('stream_type'),
#             confidence,
#             frame_skip,
#             resize_scale,
#             st.session_state.get('tracker_choice', 'DeepSort')  # Pass tracker choice
#         )
    
#     # Video Gallery - Quản lý video đã xử lý
#     st.divider()
#     show_video_gallery()


# def process_video(input_path: str, confidence: float, max_duration: int,
#                   frame_skip: int = 0, resize_scale: int = 100):
#     """
#     Xử lý video với progress tracking và optimization
    
#     Args:
#         input_path: Đường dẫn video input
#         confidence: Confidence threshold
#         max_duration: Thời lượng tối đa (giây)
#         frame_skip: Số frame bỏ qua (0 = không bỏ)
#         resize_scale: Phần trăm resize (100 = không resize)
#     """
#     try:
#         # Load models (cached)
#         with st.spinner("🧠 Đang load models..."):
#             model_person, model_vehicle = load_yolo_models()
#             tracker = initialize_tracker()
        
#         # Khởi tạo processor
#         processor = VideoProcessor(model_person, model_vehicle, tracker)
#         processor.set_confidence(confidence)
        
#         # Show optimization info
#         st.info(f"""
#         **Tối ưu hóa:**
#         - Bỏ qua: Mỗi {frame_skip + 1} frame
#         - Resize: {resize_scale}%
#         - Tốc độ ước tính: **{(1 + frame_skip) * (100/resize_scale)**2:.0f}x** nhanh hơn
#         """)
        
#         # Tính max frames
#         video_info = st.session_state.get('video_info')
#         max_frames = None
#         if video_info:
#             max_frames = int(min(
#                 video_info['total_frames'],
#                 video_info['fps'] * max_duration
#             ))
#             # Điều chỉnh cho frame skip
#             if frame_skip > 0:
#                 max_frames = max_frames // (frame_skip + 1)
        
#         # Tạo output path
#         output_path = utils.get_temp_filepath(prefix="output_", suffix=".mp4")
        
#         # Progress bar
#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         stats_container = st.empty()
#         fps_display = st.empty()  # Display FPS
        
#         # Timer
#         import time
#         last_update_time = time.time()
#         processed_frames_since_update = 0
#         current_fps = 0
        
#         def update_progress(progress, frame_count, total_frames, stats):
#             """Callback để cập nhật progress với FPS"""
#             nonlocal last_update_time, processed_frames_since_update, current_fps
            
#             progress_bar.progress(progress)
            
#             # Calculate FPS
#             processed_frames_since_update += 1
#             current_time = time.time()
#             time_elapsed = current_time - last_update_time
            
#             if time_elapsed >= 1.0:  # Update FPS every second
#                 current_fps = processed_frames_since_update / time_elapsed
#                 last_update_time = current_time
#                 processed_frames_since_update = 0
            
#             status_text.text(f"⏳ Đang xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
#             fps_display.success(f"🚀 Tốc độ xử lý: **{current_fps:.1f} FPS**")
            
#             # Hiển thị stats
#             with stats_container.container():
#                 st.markdown("### Thống kê Real-time")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Tổng đối tượng", stats['total_objects'])
#                 with col2:
#                     st.metric("Loại phát hiện", len(stats['class_counts']))
                
#                 # Chi tiết
#                 if stats['class_counts']:
#                     st.markdown("**Chi tiết:**")
#                     for cls_name, count in stats['class_counts'].items():
#                         st.write(f"- {cls_name}: **{count}**")
        
#         # Xử lý video
#         start_time = time.time()
#         results = processor.process_video(
#             input_path,
#             output_path,
#             progress_callback=update_progress,
#             max_frames=max_frames,
#             frame_skip=frame_skip,
#             resize_scale=resize_scale
#         )
#         processing_time = time.time() - start_time
        
#         progress_bar.progress(1.0)
#         status_text.success("✅ Xử lý hoàn tất!")
        
#         # Convert sang H.264
#         with st.spinner("🔄 Đang convert video..."):
#             final_output = os.path.join(config.OUTPUT_DIR, "output_final.mp4")
#             if utils.convert_video_h264(output_path, final_output):
#                 st.success("✅ Video đã được convert thành công!")
                
#                 # Hiển thị video
#                 st.video(final_output)
                
#                 # Lưu vào gallery
#                 save_to_gallery(final_output, results)
                
#                 # Download button
#                 with open(final_output, 'rb') as f:
#                     st.download_button(
#                         label="📥 Tải video đã xử lý",
#                         data=f,
#                         file_name="processed_video.mp4",
#                         mime="video/mp4",
#                         use_container_width=True
#                     )
                
#                 st.success("✅ Video đã được lưu vào Gallery!")
                
#                 # Kết quả cuối cùng
#                 st.markdown("### 📈 Kết quả cuối cùng")
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Frames đã xử lý", results['frames_processed'])
#                 with col2:
#                     st.metric("Tổng đối tượng", results['total_objects'])
#                 with col3:
#                     st.metric("Thời gian xử lý", f"{processing_time:.1f}s")
                
#                 # Class breakdown
#                 if results['class_counts']:
#                     st.markdown("**Phân loại đối tượng:**")
#                     for cls_name, count in sorted(results['class_counts'].items()):
#                         st.write(f"- {cls_name}: **{count}**")
            
#             else:
#                 st.error("❌ Lỗi khi convert video. Vui lòng kiểm tra FFmpeg.")
    
#     except Exception as e:
#         st.error(f"❌ Lỗi xử lý video: {e}")
#         import traceback
#         st.code(traceback.format_exc())


# def save_to_gallery(video_path: str, results: dict):
#     """
#     Lưu video đã xử lý vào gallery
    
#     Args:
#         video_path: Đường dẫn video đã xử lý
#         results: Kết quả xử lý
#     """
#     import shutil
#     from datetime import datetime
#     import json
    
#     # Tạo thư mục gallery
#     gallery_dir = os.path.join(config.OUTPUT_DIR, "gallery")
#     os.makedirs(gallery_dir, exist_ok=True)
    
#     # Tạo tên file unique
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     new_filename = f"processed_{timestamp}.mp4"
#     new_path = os.path.join(gallery_dir, new_filename)
    
#     # Copy video
#     shutil.copy2(video_path, new_path)
    
#     # Lưu metadata
#     metadata = {
#         'filename': new_filename,
#         'timestamp': timestamp,
#         'results': results,
#         'file_size': os.path.getsize(new_path)
#     }
    
#     metadata_path = os.path.join(gallery_dir, f"metadata_{timestamp}.json")
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=2)


# def show_video_gallery():
#     """
#     Hiển thị gallery các video đã xử lý với chức năng xóa
#     """
#     st.subheader("Gallery - Video đã xử lý")
    
#     gallery_dir = os.path.join(config.OUTPUT_DIR, "gallery")
    
#     if not os.path.exists(gallery_dir):
#         st.info("📭 Gallery trống. Hãy xử lý video để thấy kết quả ở đây.")
#         return
    
#     # Lấy danh sách video
#     video_files = [f for f in os.listdir(gallery_dir) if f.endswith('.mp4')]
    
#     if not video_files:
#         st.info("📭 Gallery trống. Hãy xử lý video để thấy kết quả ở đây.")
#         return
    
#     # Sắp xếp theo thời gian (mới nhất trước)
#     video_files.sort(reverse=True)
    
#     st.info(f"Tổng số video: **{len(video_files)}**")
    
#     # Hiển thị từng video
#     for idx, video_file in enumerate(video_files):
#         video_path = os.path.join(gallery_dir, video_file)
#         metadata_file = video_file.replace('processed_', 'metadata_').replace('.mp4', '.json')
#         metadata_path = os.path.join(gallery_dir, metadata_file)
        
#         with st.expander(f"{video_file}", expanded=(idx == 0)):
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 # Hiển thị video
#                 st.video(video_path)
            
#             with col2:
#                 # Hiển thị metadata
#                 if os.path.exists(metadata_path):
#                     with open(metadata_path, 'r', encoding='utf-8') as f:
#                         metadata = json.load(f)
                    
#                     st.markdown("**Thống kê:**")
#                     results = metadata.get('results', {})
#                     st.write(f"- Frames: {results.get('frames_processed', 'N/A')}")
#                     st.write(f"- Đối tượng: {results.get('total_objects', 'N/A')}")
                    
#                     if results.get('class_counts'):
#                         st.markdown("**Phân loại:**")
#                         for cls_name, count in results['class_counts'].items():
#                             st.write(f"- {cls_name}: {count}")
                
#                 # File size
#                 file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
#                 st.write(f"Dung lượng: {file_size_mb:.2f} MB")
                
#                 st.divider()
                
#                 # Actions
#                 col_a, col_b = st.columns(2)
                
#                 with col_a:
#                     # Download
#                     with open(video_path, 'rb') as f:
#                         st.download_button(
#                             "Tải xuống",
#                             data=f,
#                             file_name=video_file,
#                             mime="video/mp4",
#                             key=f"download_{video_file}",
#                             use_container_width=True
#                         )
                
#                 with col_b:
#                     # Delete
#                     if st.button("Xóa", key=f"delete_{video_file}", use_container_width=True):
#                         try:
#                             os.remove(video_path)
#                             if os.path.exists(metadata_path):
#                                 os.remove(metadata_path)
#                             st.success(f"Đã xóa {video_file}")
#                             st.rerun()
#                         except Exception as e:
#                             st.error(f"Lỗi khi xóa: {e}")
    
#     # Nút xóa tất cả
#     st.divider()
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         if st.button("Xóa tất cả video", type="secondary", use_container_width=True):
#             if st.session_state.get('confirm_delete_all'):
#                 # Xóa thực sự
#                 try:
#                     import shutil
#                     shutil.rmtree(gallery_dir)
#                     os.makedirs(gallery_dir)
#                     st.success("Đã xóa tất cả video!")
#                     st.session_state['confirm_delete_all'] = False
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Lỗi: {e}")
#             else:
#                 # Yêu cầu xác nhận
#                 st.session_state['confirm_delete_all'] = True
#                 st.warning("CẢNH BÁO: Nhấn lại để xác nhận xóa TẤT CẢ!")
#                 st.rerun()


# # =============================================================================
# # FOOTER
# # =============================================================================
# def show_footer():
#     st.divider()
#     st.markdown("""
#     <div style='text-align: center; color: #666; padding: 2rem;'>
#         <p>🎓 Đồ án Kỹ sư - Xử lý Video với YOLOv3 + DeepSort</p>
#         <p>💡 Powered by Streamlit, Ultralytics YOLO, and DeepSORT</p>
#     </div>
#     """, unsafe_allow_html=True)


# # =============================================================================
# # RUN APP
# # =============================================================================
# if __name__ == "__main__":
#     try:
#         from streamlit.runtime.scriptrunner import get_script_run_ctx
#         running_in_streamlit = get_script_run_ctx() is not None
#     except Exception:
#         running_in_streamlit = False

#     if running_in_streamlit:
#         main()
#         show_footer()
#     else:
#         from streamlit.web import cli as stcli
#         sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
#         raise SystemExit(stcli.main())

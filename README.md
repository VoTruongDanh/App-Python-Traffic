# 🦅 Real-time Object Tracking Encom (PyQt5)

Ứng dụng Desktop hiệu năng cao dể phát hiện và theo dõi đối tượng (Người, Xe cộ) sử dụng mạng YOLOv11/v8/v3 và thuật toán DeepSort/SORT.

## 🚀 Tính năng nổi bật

- **Tracking Real-time**: Theo dõi đa đối tượng với ID ổn định.
- **High Performance Mode**: Chế độ tối ưu hóa cho FPS cao (30-60+ FPS) trên phần cứng trung bình.
- **Multi-Threading**: Tách luồng inference và render để tận dụng GPU.
- **Hỗ trợ đa nguồn**: RTSP Camera, Webcam, Video File, YouTube URL.
- **Tùy chỉnh linh hoạt**: Resize, Confidence, Frame Skip, ROI (Vùng quan tâm).

##  cài đặt & Chạy ứng dụng

### Yêu cầu hệ thống
- Windows 10/11
- Python 3.9+
- GPU NVIDIA (Khuyến kích) để có FPS cao nhất.

### Cách chạy nhanh
Double-click vào file `RUN_PYQT_APP.bat`. Script sẽ tự động:
1. Tạo môi trường ảo (venv)
2. Cài đặt thư viện cần thiết (`requirements_pyqt.txt`)
3. Khởi động ứng dụng

Hoặc chạy thủ công:
```bash
pip install -r requirements_pyqt.txt
python pyqt_app.py
```

## ⚙️ Các chế độ hiệu năng (Performance Modes)

Ứng dụng cung cấp 3 chế độ xử lý trong phần settings bên phải:

1. **Standard Mode** (Mặc định):
   - Cân bằng giữa chất lượng và tốc độ.
   - Vẽ đầy đủ thông tin (Box, Label, Conf, Trails).
   - Phù hợp test và debug.

2. **⚡ Optimized Mode (Nhanh nhất)**:
   - Sử dụng `VideoProcessorOptimized`.
   - Giảm thiểu các thao tác vẽ (chỉ vẽ Box + ID).
   - Tối ưu hóa xử lý mảng Numpy.
   - **Khuyến nghị** cho Camera giám sát hoặc máy cấu hình yếu.

3. **🚀 Threaded Mode (Parallel)**:
   - Chạy Inference và Drawing trên 2 luồng riêng biệt.
   - Giúp UI mượt mà hơn, nhưng có thể tăng độ trễ (latency) nhẹ do hàng đợi.
   - Tốt khi GPU mạnh nhưng CPU yếu.

## 🛠️ Xử lý sự cố

- **Lỗi không mở được RTSP**: Kiểm tra URL (dùng VLC để test). Ứng dụng tự động dùng FFMPEG flags để giảm delay.
- **FPS thấp**: 
  - Bật **Optimized Mode**.
  - Giảm "Resize Scale" xuống 75% hoặc 50%.
  - Tăng "Frame Skip" lên 1-2.
- **Lỗi CUDA**: Chạy `python cuda_fix.py` để kiểm tra môi trường torch/cuda.

## 📁 Cấu trúc thư mục
- `pyqt_app.py`: File chính.
- `video_processor_optimized.py`: Module xử lý tối ưu.
- `model_loader.py`: Quản lý load model.
- `config.py`: Cấu hình hệ thống.

"""
Utility functions cho Video Processing Application
"""
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Dict
import requests
import streamlit as st
from src.core import config


def create_directories():
    """Tạo các thư mục cần thiết"""
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def cleanup_temp_files():
    """Dọn dẹp các file tạm"""
    if os.path.exists(config.TEMP_DIR):
        shutil.rmtree(config.TEMP_DIR)
        os.makedirs(config.TEMP_DIR)


def get_temp_filepath(prefix: str = "temp", suffix: str = ".mp4") -> str:
    """
    Tạo đường dẫn file tạm duy nhất
    
    Args:
        prefix: Prefix cho tên file
        suffix: Extension của file
        
    Returns:
        Đường dẫn file tạm
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=config.TEMP_DIR)
    os.close(fd)
    return path


def convert_video_h264(input_path: str, output_path: str) -> bool:
    """
    Convert video sang H.264 codec để tương thích với browser
    
    Args:
        input_path: Đường dẫn video đầu vào
        output_path: Đường dẫn video đầu ra
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-v', 'error',  # Only show errors
            '-i', input_path,
            '-vcodec', config.FINAL_CODEC,
            '-pix_fmt', config.PIXEL_FORMAT,
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        else:
            st.error(f"FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        st.error("⚠️ FFmpeg không được cài đặt. Vui lòng cài đặt FFmpeg để convert video.")
        return False
    except Exception as e:
        st.error(f"Lỗi khi convert video: {e}")
        return False


def format_statistics(class_counts: Dict[str, int]) -> str:
    """
    Format thống kê thành chuỗi hiển thị
    
    Args:
        class_counts: Dictionary chứa số lượng từng class
        
    Returns:
        Chuỗi formatted
    """
    if not class_counts:
        return "Chưa phát hiện đối tượng nào"
    
    lines = []
    total = sum(class_counts.values())
    lines.append(f"**Tổng số: {total} đối tượng**\n")
    
    for class_name, count in sorted(class_counts.items()):
        lines.append(f"- {class_name}: **{count}**")
    
    return "\n".join(lines)


def download_sample_video(url: str, output_path: str) -> bool:
    """
    Tải video mẫu từ URL
    
    Args:
        url: URL của video
        output_path: Đường dẫn lưu video
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 100000
        else:
            st.error(f"HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Lỗi khi tải video: {e}")
        return False


def check_ffmpeg_installed() -> bool:
    """
    Kiểm tra xem FFmpeg đã được cài đặt chưa
    
    Returns:
        True nếu FFmpeg có sẵn, False nếu không
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_info(video_path: str) -> Optional[Dict]:
    """
    Lấy thông tin về video
    
    Args:
        video_path: Đường dẫn video
        
    Returns:
        Dictionary chứa thông tin video hoặc None nếu lỗi
    """
    import cv2
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration_seconds': 0
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['total_frames'] / info['fps']
        
        cap.release()
        return info
        
    except Exception as e:
        st.error(f"Lỗi khi đọc thông tin video: {e}")
        return None


def validate_video_file(file_path: str) -> bool:
    """
    Kiểm tra file video có hợp lệ không
    
    Args:
        file_path: Đường dẫn file video
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not os.path.exists(file_path):
        return False
    
    if os.path.getsize(file_path) < 1000:
        return False
    
    ext = os.path.splitext(file_path)[1][1:].lower()
    if ext not in config.SUPPORTED_FORMATS:
        return False
    
    return True


def download_video_from_url(url: str, output_path: str) -> bool:
    """
    Tải video từ URL (hỗ trợ YouTube và direct links)
    
    Args:
        url: URL của video
        output_path: Đường dẫn lưu video
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    # Xóa file cũ nếu tồn tại
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Kiểm tra nếu là YouTube
    if "youtube.com" in url or "youtu.be" in url:
        return download_youtube_video(url, output_path)
    else:
        # Direct link
        return download_sample_video(url, output_path)


def download_youtube_video(url: str, output_path: str) -> bool:
    """
    Tải video từ YouTube bằng yt-dlp
    
    Args:
        url: YouTube URL
        output_path: Đường dẫn lưu video
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    try:
        # Import yt-dlp
        try:
            import yt_dlp
        except ImportError:
            st.info("⏳ Đang cài đặt yt-dlp...")
            import subprocess
            subprocess.check_call([
                'pip', 'install', '-q', 'yt-dlp'
            ])
            import yt_dlp
        
        # Cấu hình yt-dlp
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Tải video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Kiểm tra file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
            return True
        else:
            st.error("❌ File tải về không hợp lệ")
            return False
    
    except Exception as e:
        st.error(f"❌ Lỗi khi tải YouTube video: {e}")
        return False


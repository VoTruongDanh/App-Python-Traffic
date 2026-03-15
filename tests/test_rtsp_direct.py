"""
Test RTSP stream directly without processing
To check if the issue is from RTSP source or app
"""
import cv2
import sys
import os

def test_rtsp(rtsp_url):
    """Test RTSP stream with optimal settings"""
    print("=" * 60)
    print("RTSP DIRECT TEST")
    print("=" * 60)
    print(f"URL: {rtsp_url}")
    print()
    
    # Set FFMPEG options for low latency
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay'
    
    # Open with FFMPEG backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Ultra-low latency settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # No buffering
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open RTSP stream")
        print("   Check URL and network connection")
        return
    
    print("✅ RTSP stream opened")
    print()
    print("Settings:")
    print(f"  Buffer size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print()
    print("Press 'q' to quit")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        # Flush buffer - grab multiple times to get latest frame
        for _ in range(2):
            cap.grab()
        
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Failed to read frame")
            continue
        
        frame_count += 1
        
        # Show frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('RTSP Test - Raw Stream', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    else:
        print("Usage: python test_rtsp_direct.py <rtsp_url>")
        print()
        print("Example:")
        print("  python test_rtsp_direct.py rtsp://192.168.1.100:554/stream")
        sys.exit(1)
    
    test_rtsp(rtsp_url)

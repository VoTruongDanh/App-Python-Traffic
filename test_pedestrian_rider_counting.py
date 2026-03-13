"""
Test Pedestrian vs Rider Counting
Kiểm tra xem hệ thống có đếm đúng khi người chuyển từ pedestrian sang rider không
"""
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model_loader import ModelLoader
from sort_tracker import SORTTracker
from video_processor import VideoProcessor
import config


def test_pedestrian_rider_counting():
    """
    Test counting logic:
    - Khi một người vào khung hình là pedestrian
    - Sau đó nhảy lên xe trở thành rider
    - Chỉ nên đếm 1 người, không phải 2
    """
    print("=" * 60)
    print("TEST: Pedestrian vs Rider Counting")
    print("=" * 60)
    print()
    print("Mục đích:")
    print("  - Kiểm tra logic đếm khi classification thay đổi")
    print("  - Pedestrian -> Rider: chỉ đếm 1 người")
    print("  - Track ID giữ nguyên, classification thay đổi")
    print()
    print("Cách test:")
    print("  1. Chạy video có người đi bộ rồi lên xe")
    print("  2. Quan sát label thay đổi từ Green (Pedestrian) -> Orange (Rider)")
    print("  3. Kiểm tra statistics: Pedestrian count hoặc Rider count tăng 1")
    print("  4. Total count chỉ tăng 1, không phải 2")
    print()
    
    # Load models
    print("Loading models...")
    model_loader = ModelLoader()
    
    # Try Train2 model first (has both person and vehicle)
    model_person = model_loader.load_model("Train2/best.pt")
    model_vehicle = model_person  # Same model
    
    if model_person is None:
        print("❌ Cannot load Train2 model")
        print("   Please use a model that can detect both person and vehicle")
        return
    
    print("✓ Model loaded")
    
    # Create tracker
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Create processor
    processor = VideoProcessor(model_person, model_vehicle, tracker)
    processor.using_train2 = True
    
    print()
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print("1. Nhập đường dẫn video hoặc RTSP stream")
    print("2. Quan sát:")
    print("   - Green label = Pedestrian (người đi bộ)")
    print("   - Orange label = Rider (người trên xe)")
    print("3. Kiểm tra statistics ở góc trên:")
    print("   - Pedestrian: X")
    print("   - Rider: Y")
    print("   - Total: X + Y (không bị đếm 2 lần)")
    print()
    print("Press 'q' to quit")
    print("=" * 60)
    print()
    
    # Get video source
    video_path = input("Video path or RTSP URL: ").strip()
    
    if not video_path:
        print("❌ No video path provided")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    print(f"✓ Video opened: {video_path}")
    print()
    print("Processing... (Press 'q' to quit)")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or cannot read frame")
            break
        
        frame_count += 1
        
        # Process frame
        processed_frame, stats = processor.process_frame(frame, resize_scale=100, max_det=30)
        
        # Draw statistics on frame
        y_offset = 30
        cv2.putText(processed_frame, "=== COUNTING TEST ===", (10, y_offset),
                   config.FONT, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(processed_frame, f"Frame: {frame_count}", (10, y_offset),
                   config.FONT, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(processed_frame, f"Total Objects: {stats['total_objects']}", (10, y_offset),
                   config.FONT, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        # Show class counts
        for class_name, count in stats['class_counts'].items():
            color = (0, 255, 0) if class_name == "Pedestrian" else (255, 165, 0)
            cv2.putText(processed_frame, f"{class_name}: {count}", (10, y_offset),
                       config.FONT, 0.6, color, 2)
            y_offset += 25
        
        # Show legend
        y_offset += 10
        cv2.putText(processed_frame, "Legend:", (10, y_offset),
                   config.FONT, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(processed_frame, "Green = Pedestrian", (10, y_offset),
                   config.FONT, 0.5, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(processed_frame, "Orange = Rider/Driver", (10, y_offset),
                   config.FONT, 0.5, (255, 165, 0), 1)
        
        # Display
        cv2.imshow("Pedestrian vs Rider Counting Test", processed_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nStopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("FINAL STATISTICS:")
    print("=" * 60)
    print(f"Total Objects: {stats['total_objects']}")
    for class_name, count in stats['class_counts'].items():
        print(f"  {class_name}: {count}")
    print()
    print("✓ Test completed")
    print()
    print("Kết luận:")
    print("  - Nếu một người chuyển từ Pedestrian -> Rider")
    print("  - Total count chỉ tăng 1 (đếm đúng)")
    print("  - Classification được lưu lại từ lần đầu tiên")
    print("=" * 60)


if __name__ == "__main__":
    test_pedestrian_rider_counting()

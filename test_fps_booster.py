"""
Test FPS Booster
So sánh FPS với và không có FPS Booster
"""
import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_loader import ModelLoader
from sort_tracker import SORTTracker
from video_processor import VideoProcessor
from fps_booster import FPSBooster
import config


def test_without_booster(video_path, duration=30):
    """Test FPS không có booster"""
    print("\n" + "=" * 60)
    print("TEST 1: WITHOUT FPS BOOSTER")
    print("=" * 60)
    
    # Load model
    model_loader = ModelLoader()
    model = model_loader.load_model("Train2/best.pt")
    
    if model is None:
        print("❌ Cannot load model")
        return None
    
    tracker = SORTTracker()
    processor = VideoProcessor(model, model, tracker)
    processor.using_train2 = True
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Process
    frame_count = 0
    start_time = time.time()
    fps_list = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
            continue
        
        frame_start = time.time()
        
        # Process frame (full resolution)
        processed, stats = processor.process_frame(frame, resize_scale=100, max_det=30)
        
        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(fps)
        
        frame_count += 1
        
        # Display
        cv2.putText(processed, f"FPS: {fps:.1f}", (10, 30),
                   config.FONT, 0.7, (0, 255, 0), 2)
        cv2.putText(processed, "WITHOUT BOOSTER", (10, 60),
                   config.FONT, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Test Without Booster", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate stats
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0
    
    print(f"\n✓ Processed {frame_count} frames in {duration}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Min FPS: {min_fps:.2f}")
    print(f"  Max FPS: {max_fps:.2f}")
    
    return {
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'frames': frame_count
    }


def test_with_booster(video_path, duration=30):
    """Test FPS với booster"""
    print("\n" + "=" * 60)
    print("TEST 2: WITH FPS BOOSTER")
    print("=" * 60)
    
    # Load model
    model_loader = ModelLoader()
    model = model_loader.load_model("Train2/best.pt")
    
    if model is None:
        print("❌ Cannot load model")
        return None
    
    tracker = SORTTracker()
    processor = VideoProcessor(model, model, tracker)
    processor.using_train2 = True
    
    # Create FPS Booster
    booster = FPSBooster(target_fps=30, enable_all=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Process
    frame_count = 0
    start_time = time.time()
    fps_list = []
    last_detections = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
            continue
        
        frame_start = time.time()
        
        # Get current FPS
        current_fps = fps_list[-1] if fps_list else 20
        
        # Get resolution scale from booster
        resize_scale = booster.get_resolution_scale(current_fps)
        
        # Check if should process frame
        if booster.should_process_frame(current_fps, frame_count):
            # Process frame with dynamic resolution
            processed, stats = processor.process_frame(frame, resize_scale=resize_scale, max_det=30)
        else:
            # Skip frame, use last result
            processed = frame
        
        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(fps)
        
        frame_count += 1
        
        # Display
        cv2.putText(processed, f"FPS: {fps:.1f}", (10, 30),
                   config.FONT, 0.7, (0, 255, 0), 2)
        cv2.putText(processed, "WITH BOOSTER", (10, 60),
                   config.FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(processed, f"Resolution: {resize_scale}%", (10, 90),
                   config.FONT, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Test With Booster", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate stats
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0
    
    print(f"\n✓ Processed {frame_count} frames in {duration}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Min FPS: {min_fps:.2f}")
    print(f"  Max FPS: {max_fps:.2f}")
    
    # Booster stats
    booster_stats = booster.get_stats()
    print(f"\n  Booster Stats:")
    for key, value in booster_stats.items():
        print(f"    {key}: {value}")
    
    return {
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'frames': frame_count,
        'booster_stats': booster_stats
    }


def main():
    print("=" * 60)
    print("FPS BOOSTER TEST")
    print("=" * 60)
    print()
    print("So sánh FPS với và không có FPS Booster")
    print()
    print("FPS Booster bao gồm:")
    print("  1. Dynamic Resolution Scaling")
    print("  2. Adaptive Frame Skip")
    print("  3. Smart Detection Throttling")
    print()
    
    # Get video path
    video_path = input("Video path or RTSP URL: ").strip()
    
    if not video_path:
        print("❌ No video path provided")
        return
    
    duration = 30  # Test for 30 seconds
    
    # Test without booster
    result1 = test_without_booster(video_path, duration)
    
    if result1 is None:
        return
    
    time.sleep(2)  # Pause between tests
    
    # Test with booster
    result2 = test_with_booster(video_path, duration)
    
    if result2 is None:
        return
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Metric':<20} {'Without Booster':<20} {'With Booster':<20} {'Improvement':<15}")
    print("-" * 75)
    
    avg_improvement = (result2['avg_fps'] - result1['avg_fps']) / result1['avg_fps'] * 100
    print(f"{'Average FPS':<20} {result1['avg_fps']:<20.2f} {result2['avg_fps']:<20.2f} {avg_improvement:>+.1f}%")
    
    min_improvement = (result2['min_fps'] - result1['min_fps']) / result1['min_fps'] * 100
    print(f"{'Min FPS':<20} {result1['min_fps']:<20.2f} {result2['min_fps']:<20.2f} {min_improvement:>+.1f}%")
    
    max_improvement = (result2['max_fps'] - result1['max_fps']) / result1['max_fps'] * 100
    print(f"{'Max FPS':<20} {result1['max_fps']:<20.2f} {result2['max_fps']:<20.2f} {max_improvement:>+.1f}%")
    
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if avg_improvement > 20:
        print(f"✅ FPS Booster SIGNIFICANTLY improved performance (+{avg_improvement:.1f}%)")
    elif avg_improvement > 10:
        print(f"✅ FPS Booster improved performance (+{avg_improvement:.1f}%)")
    elif avg_improvement > 0:
        print(f"⚠️  FPS Booster slightly improved performance (+{avg_improvement:.1f}%)")
    else:
        print(f"❌ FPS Booster did not improve performance ({avg_improvement:.1f}%)")
    
    print()


if __name__ == "__main__":
    main()

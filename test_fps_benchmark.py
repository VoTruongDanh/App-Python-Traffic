"""
FPS Benchmark - Test real inference speed without UI overhead
"""
import cv2
import time
import numpy as np
from model_loader import load_yolo_models, initialize_tracker
from video_processor import VideoProcessor

def benchmark_fps(video_path, duration=30):
    """
    Benchmark FPS without UI overhead
    
    Args:
        video_path: Path to video or stream URL
        duration: Test duration in seconds
    """
    print("=" * 60)
    print("FPS BENCHMARK - No UI Overhead")
    print("=" * 60)
    
    # Load models
    print("\n[1/3] Loading models...")
    model_person, model_vehicle = load_yolo_models(
        best_model_choice="Train1",
        base_model_choice="YOLOv26n (Fastest)",
        custom_model_path=None
    )
    
    # Initialize tracker
    print("[2/3] Initializing tracker...")
    tracker = initialize_tracker("Simple (Fastest)")
    
    # Create processor
    processor = VideoProcessor(model_person, model_vehicle, tracker)
    processor.set_confidence(0.55)
    
    # Open video
    print(f"[3/3] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    print(f"\n{'='*60}")
    print("BENCHMARKING...")
    print(f"{'='*60}\n")
    
    frame_count = 0
    start_time = time.time()
    fps_samples = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_start = time.time()
            processed_frame, stats = processor.process_frame(frame)
            frame_time = time.time() - frame_start
            
            frame_count += 1
            
            # Calculate FPS every second
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                fps_samples.append(fps)
                print(f"FPS: {fps:.1f} | Frame time: {frame_time*1000:.1f}ms | Objects: {stats['total_objects']}")
                
                frame_count = 0
                start_time = time.time()
            
            # Stop after duration
            if len(fps_samples) >= duration:
                break
                
    finally:
        cap.release()
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if fps_samples:
        avg_fps = sum(fps_samples) / len(fps_samples)
        min_fps = min(fps_samples)
        max_fps = max(fps_samples)
        
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Min FPS: {min_fps:.1f}")
        print(f"Max FPS: {max_fps:.1f}")
        print(f"Samples: {len(fps_samples)}")
        
        if avg_fps < 15:
            print("\n⚠️  FPS < 15: Bottleneck is in inference/processing, not UI")
        elif avg_fps >= 20:
            print("\n✅ FPS >= 20: Good performance!")
        else:
            print("\n⚙️  FPS 15-20: Acceptable performance")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default: webcam
        video_path = 0
        print("No video specified, using webcam (0)")
        print("Usage: python test_fps_benchmark.py <video_path_or_url>")
        print()
    
    benchmark_fps(video_path, duration=30)

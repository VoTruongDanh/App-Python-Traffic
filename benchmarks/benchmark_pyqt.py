"""
Benchmark script to compare PyQt5 vs Streamlit performance
"""
import cv2
import time
import numpy as np
from src.inference.model_loader import load_yolo_models, initialize_tracker
from src.processing.video_processor import VideoProcessor


def benchmark_processing(video_source, duration_seconds=10, frame_skip=2, resize_scale=50):
    """
    Benchmark video processing performance
    
    Args:
        video_source: Video file path or camera index
        duration_seconds: How long to run benchmark
        frame_skip: Frame skip parameter
        resize_scale: Resize scale percentage
    """
    print("=" * 60)
    print("  PyQt5 Performance Benchmark")
    print("=" * 60)
    print()
    
    # Load models
    print("[1/4] Loading models...")
    start = time.time()
    model_person, model_vehicle = load_yolo_models()
    tracker = initialize_tracker("Simple (Fastest)")
    processor = VideoProcessor(model_person, model_vehicle, tracker)
    processor.set_confidence(0.4)
    load_time = time.time() - start
    print(f"✓ Models loaded in {load_time:.2f}s")
    print()
    
    # Open video
    print("[2/4] Opening video source...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"✗ Cannot open video source: {video_source}")
        return
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"✓ Video: {width}x{height} @ {fps} FPS")
    print()
    
    # Calculate resize dimensions
    new_width = int(width * resize_scale / 100)
    new_height = int(height * resize_scale / 100)
    print(f"[3/4] Processing settings:")
    print(f"  - Frame skip: {frame_skip}")
    print(f"  - Resize: {resize_scale}% ({new_width}x{new_height})")
    print(f"  - Duration: {duration_seconds}s")
    print()
    
    # Benchmark
    print("[4/4] Running benchmark...")
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    total_inference_time = 0
    total_resize_time = 0
    
    try:
        while time.time() < end_time and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
            
            # Resize
            resize_start = time.time()
            if resize_scale != 100:
                frame = cv2.resize(frame, (new_width, new_height))
            total_resize_time += time.time() - resize_start
            
            # Process
            inference_start = time.time()
            processed_frame, stats = processor.process_frame(frame)
            total_inference_time += time.time() - inference_start
            
            frame_count += 1
            processed_count += 1
            
            # Progress
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = processed_count / elapsed
                print(f"  Processed {processed_count} frames | FPS: {current_fps:.1f}", end='\r')
    
    finally:
        cap.release()
    
    # Results
    total_time = time.time() - start_time
    avg_fps = processed_count / total_time
    avg_inference = (total_inference_time / processed_count) * 1000  # ms
    avg_resize = (total_resize_time / processed_count) * 1000  # ms
    
    print()
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Total frames read:      {frame_count}")
    print(f"Frames processed:       {processed_count}")
    print(f"Total time:             {total_time:.2f}s")
    print(f"Average FPS:            {avg_fps:.1f}")
    print(f"Avg inference time:     {avg_inference:.1f}ms")
    print(f"Avg resize time:        {avg_resize:.1f}ms")
    print()
    
    # Performance rating
    if avg_fps >= 30:
        rating = "🚀 EXCELLENT - Real-time capable!"
    elif avg_fps >= 20:
        rating = "✅ GOOD - Smooth performance"
    elif avg_fps >= 10:
        rating = "⚠️  FAIR - Acceptable for most use cases"
    else:
        rating = "❌ POOR - Consider optimization"
    
    print(f"Performance: {rating}")
    print()
    
    # Recommendations
    if avg_fps < 30:
        print("💡 Optimization suggestions:")
        if resize_scale > 50:
            print("  - Try resize_scale=50 for 4x speedup")
        if frame_skip < 2:
            print("  - Try frame_skip=2 for 3x speedup")
        print("  - Use YOLOv8n instead of YOLOv3")
        print("  - Use Simple tracker instead of DeepSort")
        print()
    
    return {
        'fps': avg_fps,
        'inference_ms': avg_inference,
        'resize_ms': avg_resize,
        'total_time': total_time,
        'frames_processed': processed_count
    }


def compare_configurations(video_source):
    """Compare different configuration presets"""
    print("\n" + "=" * 60)
    print("  CONFIGURATION COMPARISON")
    print("=" * 60)
    print()
    
    configs = [
        {"name": "High Quality", "skip": 0, "resize": 100},
        {"name": "Balanced", "skip": 2, "resize": 75},
        {"name": "Performance", "skip": 2, "resize": 50},
        {"name": "Maximum Speed", "skip": 5, "resize": 25},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 60)
        result = benchmark_processing(
            video_source,
            duration_seconds=5,
            frame_skip=config['skip'],
            resize_scale=config['resize']
        )
        result['config'] = config['name']
        results.append(result)
        time.sleep(1)  # Cool down
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Configuration':<20} {'FPS':<10} {'Inference (ms)':<15} {'Rating'}")
    print("-" * 60)
    
    for result in results:
        fps = result['fps']
        inf = result['inference_ms']
        
        if fps >= 30:
            rating = "🚀 Excellent"
        elif fps >= 20:
            rating = "✅ Good"
        elif fps >= 10:
            rating = "⚠️  Fair"
        else:
            rating = "❌ Poor"
        
        print(f"{result['config']:<20} {fps:<10.1f} {inf:<15.1f} {rating}")
    
    print()


if __name__ == '__main__':
    import sys
    
    # Default to webcam if no argument
    video_source = 0 if len(sys.argv) < 2 else sys.argv[1]
    
    # Try to convert to int (for webcam)
    try:
        video_source = int(video_source)
    except ValueError:
        pass
    
    print(f"Video source: {video_source}")
    print()
    
    # Run single benchmark
    # benchmark_processing(video_source, duration_seconds=10, frame_skip=2, resize_scale=50)
    
    # Or compare configurations
    compare_configurations(video_source)

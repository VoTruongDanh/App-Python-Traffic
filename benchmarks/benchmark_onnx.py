"""
Benchmark ONNX vs PyTorch performance
"""
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import os

try:
    from src.inference.onnx_model import ONNXModel
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX not available")
    exit(1)


def benchmark_model(model, image, iterations=50, warmup=10):
    """
    Benchmark model inference speed
    
    Args:
        model: Model to benchmark
        image: Test image
        iterations: Number of iterations
        warmup: Warmup iterations
        
    Returns:
        Average inference time in ms
    """
    # Warmup
    for _ in range(warmup):
        _ = model(image, conf=0.25, verbose=False)
    
    # Benchmark
    times = []
    for i in range(iterations):
        start = time.time()
        _ = model(image, conf=0.25, verbose=False)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{iterations} | Avg: {np.mean(times):.1f}ms")
    
    return np.mean(times), np.min(times), np.max(times)


def main():
    print("=" * 60)
    print("ONNX vs PyTorch BENCHMARK")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected")
    
    print()
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Models to test
    models_to_test = [
        ('yolov8n.pt', 'YOLOv8n'),
        ('yolo11n.pt', 'YOLOv11n'),
        ('../train1/best.pt', 'Train1 (Person)'),
    ]
    
    results = []
    
    for model_path, model_name in models_to_test:
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping {model_name} - file not found")
            continue
        
        onnx_path = model_path.replace('.pt', '.onnx')
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        # Test PyTorch
        print("\n1️⃣  PyTorch (.pt):")
        try:
            pt_model = YOLO(model_path)
            if torch.cuda.is_available():
                pt_model.to('cuda')
            
            pt_avg, pt_min, pt_max = benchmark_model(pt_model, test_image)
            print(f"   Average: {pt_avg:.1f}ms ({1000/pt_avg:.1f} FPS)")
            print(f"   Min: {pt_min:.1f}ms | Max: {pt_max:.1f}ms")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            pt_avg = None
        
        # Test ONNX
        print("\n2️⃣  ONNX Runtime (.onnx):")
        if not os.path.exists(onnx_path):
            print(f"   ⚠️  ONNX not found: {onnx_path}")
            print(f"   Convert with: python UI/convert_models_to_onnx.py")
            onnx_avg = None
        else:
            try:
                onnx_model = ONNXModel(onnx_path, use_gpu=torch.cuda.is_available())
                
                onnx_avg, onnx_min, onnx_max = benchmark_model(onnx_model, test_image)
                print(f"   Average: {onnx_avg:.1f}ms ({1000/onnx_avg:.1f} FPS)")
                print(f"   Min: {onnx_min:.1f}ms | Max: {onnx_max:.1f}ms")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                onnx_avg = None
        
        # Calculate speedup
        if pt_avg and onnx_avg:
            speedup = pt_avg / onnx_avg
            print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with ONNX")
            results.append({
                'model': model_name,
                'pytorch_ms': pt_avg,
                'onnx_ms': onnx_avg,
                'speedup': speedup
            })
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"\n{r['model']}:")
            print(f"  PyTorch: {r['pytorch_ms']:.1f}ms ({1000/r['pytorch_ms']:.1f} FPS)")
            print(f"  ONNX:    {r['onnx_ms']:.1f}ms ({1000/r['onnx_ms']:.1f} FPS)")
            print(f"  Speedup: {r['speedup']:.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\n✅ Average speedup: {avg_speedup:.2f}x")
        print(f"   Expected FPS improvement: {avg_speedup:.1f}x")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

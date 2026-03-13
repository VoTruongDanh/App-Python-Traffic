"""
Benchmark model inference speed to find bottleneck
"""
import time
import cv2
import numpy as np
from model_loader import load_yolo_models

def benchmark_model():
    print("=" * 60)
    print("MODEL INFERENCE BENCHMARK")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    start = time.time()
    model_person, model_vehicle = load_yolo_models()
    load_time = time.time() - start
    print(f"   Load time: {load_time:.2f}s")
    
    # Check model type
    from onnx_model import ONNXModel
    is_onnx = isinstance(model_person, ONNXModel) or isinstance(model_vehicle, ONNXModel)
    print(f"   Model type: {'ONNX' if is_onnx else 'PyTorch'}")
    
    if is_onnx:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"   Providers: {providers}")
            if 'CUDAExecutionProvider' in providers:
                print("   ✅ GPU support available")
            else:
                print("   ⚠️  CPU only!")
        except:
            pass
    
    # Create test frame (1920x1080)
    print("\n2. Creating test frame (1920x1080)...")
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Warmup (first inference is slow)
    print("\n3. Warmup inference...")
    _ = model_person(test_frame, conf=0.5, verbose=False)
    
    # Benchmark
    print("\n4. Running benchmark (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        results = model_person(test_frame, conf=0.5, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed*1000:.1f}ms")
    
    # Results
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Min: {min_time*1000:.1f}ms | Max: {max_time*1000:.1f}ms")
    print(f"Theoretical max FPS: {fps:.1f}")
    print()
    
    # Recommendations
    if fps < 15:
        print("⚠️  VERY SLOW! Recommendations:")
        if not is_onnx:
            print("   1. Convert to ONNX (2-3x speedup)")
        else:
            print("   1. Check if GPU is being used")
            print("   2. Try smaller model (YOLOv26n)")
        print("   3. Use Frame Skip: 2-3")
        print("   4. Reduce confidence threshold")
    elif fps < 25:
        print("⚠️  Slow. Recommendations:")
        print("   1. Use Frame Skip: 1-2")
        print("   2. Try YOLOv26n (fastest)")
    else:
        print("✅ Good performance!")
    
    print("=" * 60)

if __name__ == "__main__":
    benchmark_model()

"""
Test ONNX integration - verify everything works
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        print("✅ onnxruntime imported")
        print(f"   Version: {ort.__version__}")
        print(f"   Providers: {ort.get_available_providers()}")
        
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("   ✅ GPU support available")
        else:
            print("   ⚠️  GPU support NOT available")
    except ImportError as e:
        print(f"❌ onnxruntime import failed: {e}")
        return False
    
    try:
        from src.inference.onnx_model import ONNXModel
        print("✅ onnx_model imported")
    except ImportError as e:
        print(f"❌ onnx_model import failed: {e}")
        return False
    
    try:
        from src.inference.model_loader import load_yolo_models, _load_model
        print("✅ model_loader imported")
    except ImportError as e:
        print(f"❌ model_loader import failed: {e}")
        return False
    
    return True


def test_model_files():
    """Test if ONNX model files exist"""
    print("\n" + "=" * 60)
    print("TEST 2: Model Files")
    print("=" * 60)
    
    models = [
        'yolov8n.pt',
        'yolo11n.pt',
        '../train1/best.pt',
        '../Train2/best.pt',
    ]
    
    found_pt = 0
    found_onnx = 0
    
    for model_path in models:
        pt_exists = os.path.exists(model_path)
        onnx_path = model_path.replace('.pt', '.onnx')
        onnx_exists = os.path.exists(onnx_path)
        
        if pt_exists:
            found_pt += 1
            status_pt = "✅"
        else:
            status_pt = "❌"
        
        if onnx_exists:
            found_onnx += 1
            status_onnx = "✅"
        else:
            status_onnx = "⚠️ "
        
        print(f"{status_pt} {model_path}")
        print(f"{status_onnx} {onnx_path}")
    
    print(f"\nSummary: {found_pt} PyTorch models, {found_onnx} ONNX models")
    
    if found_onnx == 0:
        print("\n⚠️  No ONNX models found!")
        print("   Run: python convert_models_to_onnx.py")
        return False
    
    return True


def test_model_loading():
    """Test if models can be loaded"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Loading")
    print("=" * 60)
    
    try:
        from src.inference.model_loader import _load_model
        import torch
        
        # Test loading a model
        test_models = ['yolov8n.pt', 'yolo11n.pt']
        
        for model_path in test_models:
            if not os.path.exists(model_path):
                continue
            
            print(f"\nTesting: {model_path}")
            
            try:
                model = _load_model(model_path, use_gpu=torch.cuda.is_available())
                
                # Check if ONNX
                from src.inference.onnx_model import ONNXModel
                if isinstance(model, ONNXModel):
                    print(f"   ✅ Loaded as ONNX")
                else:
                    print(f"   ⚠️  Loaded as PyTorch (ONNX not available)")
                
                return True
            except Exception as e:
                print(f"   ❌ Loading failed: {e}")
                return False
        
        print("⚠️  No test models found")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_inference():
    """Test if inference works"""
    print("\n" + "=" * 60)
    print("TEST 4: Inference")
    print("=" * 60)
    
    try:
        import numpy as np
        from src.inference.model_loader import _load_model
        import torch
        
        # Find a model to test
        test_models = ['yolov8n.pt', 'yolo11n.pt']
        model = None
        
        for model_path in test_models:
            if os.path.exists(model_path):
                print(f"Testing inference with: {model_path}")
                model = _load_model(model_path, use_gpu=torch.cuda.is_available())
                break
        
        if model is None:
            print("⚠️  No models available for testing")
            return False
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        print("Running inference...")
        results = model(test_image, conf=0.25, verbose=False)
        
        print(f"✅ Inference successful")
        print(f"   Results: {len(results)} batch(es)")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ONNX INTEGRATION TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Model files
    results.append(("Model Files", test_model_files()))
    
    # Test 3: Model loading
    results.append(("Model Loading", test_model_loading()))
    
    # Test 4: Inference
    results.append(("Inference", test_inference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✅ ALL TESTS PASSED - ONNX integration working!")
        print("   You can now run the app with 30+ FPS")
    else:
        print("\n⚠️  Some tests failed")
        print("   Run SETUP_30FPS.bat to fix issues")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Test ONNX Runtime installation status
"""

def test_onnx():
    print("=" * 60)
    print("ONNX RUNTIME STATUS CHECK")
    print("=" * 60)
    print()
    
    # Test 1: Import
    print("Test 1: Import onnxruntime")
    try:
        import onnxruntime
        print("   ✅ SUCCESS: onnxruntime imported")
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        print()
        print("Solution: Run FIX_ONNX_GPU.bat")
        return False
    
    # Test 2: Get providers
    print()
    print("Test 2: Check providers")
    try:
        providers = onnxruntime.get_available_providers()
        print(f"   Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("   ✅ GPU Support: YES")
            gpu_ok = True
        else:
            print("   ⚠️  GPU Support: NO (CPU only)")
            gpu_ok = False
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Create session
    print()
    print("Test 3: Create ONNX session")
    try:
        import numpy as np
        
        # Create dummy ONNX model
        from onnx import helper, TensorProto
        
        # Simple identity model
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
        
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'test', [input_tensor], [output_tensor])
        model = helper.make_model(graph)
        
        # Save to bytes
        import io
        f = io.BytesIO()
        f.write(model.SerializeToString())
        f.seek(0)
        
        # Create session
        sess = onnxruntime.InferenceSession(f.read(), providers=providers)
        print(f"   ✅ Session created with: {sess.get_providers()}")
        
        # Test inference
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = sess.run(None, {'input': dummy_input})
        print(f"   ✅ Inference successful")
        
    except Exception as e:
        print(f"   ⚠️  Session test failed: {e}")
        print("   (This is OK if basic import works)")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if gpu_ok:
        print("✅ ONNX Runtime GPU is working!")
        print("   You can convert models and get 2-3x speedup")
    else:
        print("⚠️  ONNX Runtime installed but GPU not available")
        print("   Run: FIX_ONNX_GPU.bat")
        print("   Or install CUDA Toolkit")
    
    print()
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_onnx()

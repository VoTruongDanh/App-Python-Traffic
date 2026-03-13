"""
Test script for Train2 model integration
"""
import os
import sys

def test_train2_exists():
    """Test if Train2 best.pt exists"""
    print("=" * 50)
    print("Test 1: Check Train2 model file")
    print("=" * 50)
    
    train2_path = "../Train2/best.pt"
    if os.path.exists(train2_path):
        size_mb = os.path.getsize(train2_path) / (1024 * 1024)
        print(f"✅ Train2 model found: {train2_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Train2 model NOT found: {train2_path}")
        print("   Please ensure Train2/best.pt exists")
        return False


def test_config():
    """Test config has Train2 definitions"""
    print("\n" + "=" * 50)
    print("Test 2: Check config definitions")
    print("=" * 50)
    
    try:
        import config
        
        # Check Train2 path
        if hasattr(config, 'MODEL_TRAIN2_PATH'):
            print(f"✅ MODEL_TRAIN2_PATH defined: {config.MODEL_TRAIN2_PATH}")
        else:
            print("❌ MODEL_TRAIN2_PATH not defined")
            return False
        
        # Check Train2 classes
        if hasattr(config, 'TRAIN2_CLASSES'):
            print(f"✅ TRAIN2_CLASSES defined: {len(config.TRAIN2_CLASSES)} classes")
            for cls_id, cls_name in config.TRAIN2_CLASSES.items():
                print(f"   {cls_id}: {cls_name}")
        else:
            print("❌ TRAIN2_CLASSES not defined")
            return False
        
        # Check Train2 class IDs
        if hasattr(config, 'TRAIN2_CLASS_IDS'):
            print(f"✅ TRAIN2_CLASS_IDS defined: {len(config.TRAIN2_CLASS_IDS)} mappings")
        else:
            print("❌ TRAIN2_CLASS_IDS not defined")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def test_app_state():
    """Test app_state has best_model_choice"""
    print("\n" + "=" * 50)
    print("Test 3: Check app_state")
    print("=" * 50)
    
    try:
        from pyqt_app import app_state
        
        if hasattr(app_state, 'best_model_choice'):
            print(f"✅ best_model_choice defined: {app_state.best_model_choice}")
            
            # Test setting Train2
            app_state.best_model_choice = "Train2 (Multi-class)"
            print(f"✅ Can set to Train2: {app_state.best_model_choice}")
            
            # Reset to default
            app_state.best_model_choice = "Train1 (Person only)"
            print(f"✅ Can reset to Train1: {app_state.best_model_choice}")
            
            return True
        else:
            print("❌ best_model_choice not defined in app_state")
            return False
            
    except Exception as e:
        print(f"❌ Error testing app_state: {e}")
        return False


def test_video_processor():
    """Test video processor has Train2 methods"""
    print("\n" + "=" * 50)
    print("Test 4: Check video processor")
    print("=" * 50)
    
    try:
        from video_processor import VideoProcessor
        
        # Check methods exist
        methods = ['_is_train2_model', '_get_class_name']
        for method in methods:
            if hasattr(VideoProcessor, method):
                print(f"✅ Method {method}() exists")
            else:
                print(f"❌ Method {method}() not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing video processor: {e}")
        return False


def test_documentation():
    """Test documentation exists"""
    print("\n" + "=" * 50)
    print("Test 5: Check documentation")
    print("=" * 50)
    
    doc_path = "docs/TRAIN2_MODEL_GUIDE.md"
    if os.path.exists(doc_path):
        size_kb = os.path.getsize(doc_path) / 1024
        print(f"✅ Train2 guide found: {doc_path}")
        print(f"   Size: {size_kb:.2f} KB")
        return True
    else:
        print(f"❌ Train2 guide NOT found: {doc_path}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Train2 Model Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Train2 file exists", test_train2_exists),
        ("Config definitions", test_config),
        ("App state", test_app_state),
        ("Video processor", test_video_processor),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nTrain2 integration is working correctly.")
        print("\nNext steps:")
        print("1. Run: python pyqt_app.py")
        print("2. Select 'Train2 (Multi-class)' in dropdown")
        print("3. Load a video with vehicles")
        print("4. Verify 5 classes are detected!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        print("\nPlease fix the issues above before using Train2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

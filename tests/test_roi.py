"""
Simple test script for ROI functionality
Run this to verify ROI feature is working correctly
"""
import sys
import numpy as np
import cv2
from src.tracking.roi_manager import ROIManager


def test_roi_basic():
    """Test basic ROI functionality"""
    print("Testing ROI Manager...")
    
    # Create ROI manager
    roi = ROIManager()
    print("✓ ROI Manager created")
    
    # Test: ROI should be inactive initially
    assert not roi.is_active(), "ROI should be inactive initially"
    print("✓ Initial state correct")
    
    # Add points to create a square ROI (100x100 at position 100,100)
    roi.add_point(100, 100)
    roi.add_point(200, 100)
    roi.add_point(200, 200)
    roi.add_point(100, 200)
    print("✓ Added 4 points")
    
    # Test: ROI should be active now
    assert roi.is_active(), "ROI should be active with 4 points"
    print("✓ ROI is active")
    
    # Test: Object inside ROI
    bbox_inside = [150, 150, 180, 180]  # Center of ROI
    assert roi.is_object_in_roi(bbox_inside), "Object should be inside ROI"
    print("✓ Object inside ROI detected correctly")
    
    # Test: Object outside ROI
    bbox_outside = [250, 250, 280, 280]  # Outside ROI
    assert not roi.is_object_in_roi(bbox_outside), "Object should be outside ROI"
    print("✓ Object outside ROI detected correctly")
    
    # Test: Object partially in ROI (with default 50% threshold)
    bbox_partial = [180, 180, 220, 220]  # Partially overlapping
    result = roi.is_object_in_roi(bbox_partial)
    print(f"✓ Partial overlap handled (result: {result})")
    
    # Test: Threshold adjustment
    roi.set_threshold(0.0)  # Accept any overlap
    assert roi.is_object_in_roi(bbox_partial), "Should accept with 0% threshold"
    print("✓ Threshold 0% works")
    
    roi.set_threshold(1.0)  # Require full overlap
    assert not roi.is_object_in_roi(bbox_partial), "Should reject with 100% threshold"
    print("✓ Threshold 100% works")
    
    # Test: Clear ROI
    roi.clear()
    assert not roi.is_active(), "ROI should be inactive after clear"
    print("✓ Clear ROI works")
    
    print("\n✅ All basic tests passed!")


def test_roi_save_load():
    """Test save/load functionality"""
    print("\nTesting save/load...")
    
    # Create ROI with specific configuration
    roi1 = ROIManager()
    roi1.add_point(50, 50)
    roi1.add_point(150, 50)
    roi1.add_point(150, 150)
    roi1.set_threshold(0.7)
    
    # Get configuration
    config = roi1.get_config()
    print("✓ Configuration exported")
    
    # Create new ROI and load configuration
    roi2 = ROIManager()
    roi2.load_config(config)
    print("✓ Configuration imported")
    
    # Verify configuration matches
    assert len(roi2.roi_points) == 3, "Points should match"
    assert roi2.threshold == 0.7, "Threshold should match"
    assert roi2.is_active(), "ROI should be active"
    print("✓ Configuration matches")
    
    print("\n✅ Save/load tests passed!")


def test_roi_visual():
    """Test visual drawing (creates test image)"""
    print("\nTesting visual drawing...")
    
    # Create test frame
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Create ROI
    roi = ROIManager()
    roi.add_point(100, 100)
    roi.add_point(300, 100)
    roi.add_point(300, 300)
    roi.add_point(100, 300)
    roi.set_threshold(0.5)
    
    # Draw ROI on frame
    frame = roi.draw_roi(frame)
    print("✓ ROI drawn on frame")
    
    # Save test image
    cv2.imwrite("test_roi_visual.png", frame)
    print("✓ Test image saved as 'test_roi_visual.png'")
    
    print("\n✅ Visual tests passed!")


def test_roi_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    roi = ROIManager()
    
    # Test: Less than 3 points
    roi.add_point(100, 100)
    roi.add_point(200, 100)
    assert not roi.is_active(), "ROI should not be active with < 3 points"
    print("✓ Handles < 3 points correctly")
    
    # Test: Invalid bbox
    roi.add_point(200, 200)  # Now 3 points
    invalid_bbox = [100, 100, 100, 100]  # Zero width/height
    result = roi.is_object_in_roi(invalid_bbox)
    print(f"✓ Handles invalid bbox (result: {result})")
    
    # Test: Threshold bounds
    roi.set_threshold(-0.5)  # Should clamp to 0
    assert roi.threshold == 0.0, "Threshold should clamp to 0"
    print("✓ Threshold lower bound works")
    
    roi.set_threshold(1.5)  # Should clamp to 1
    assert roi.threshold == 1.0, "Threshold should clamp to 1"
    print("✓ Threshold upper bound works")
    
    print("\n✅ Edge case tests passed!")


def main():
    """Run all tests"""
    print("=" * 50)
    print("ROI Feature Test Suite")
    print("=" * 50)
    
    try:
        test_roi_basic()
        test_roi_save_load()
        test_roi_visual()
        test_roi_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("\nROI feature is working correctly.")
        print("You can now use it in the main application.")
        print("\nNext steps:")
        print("1. Run: python pyqt_app.py")
        print("2. Load a video or stream")
        print("3. Click 'Draw ROI' and try it out!")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

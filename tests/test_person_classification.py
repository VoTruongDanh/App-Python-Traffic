"""
Test Person Classification Integration
Quick test to verify person classification is working
"""
import cv2
import numpy as np
from src.inference.model_loader import load_yolo_models, initialize_tracker
from src.processing.video_processor import VideoProcessor
from src.core import config

def test_person_classification():
    """Test person classification with a sample video"""
    print("🧪 Testing Person Classification Integration...")
    
    # Load models
    print("📦 Loading models...")
    model_person, model_vehicle = load_yolo_models("YOLOv8n", "Train1")
    tracker = initialize_tracker("SORT")
    
    # Create processor
    processor = VideoProcessor(model_person, model_vehicle, tracker)
    processor.set_confidence(0.5)
    
    print("✅ Models loaded successfully")
    print(f"   Person Classifier IOU threshold: {processor.person_classifier.iou_threshold}")
    print(f"   Using Train2 model: {processor.using_train2}")
    
    # Test with webcam or video file
    test_source = 0  # Webcam
    # test_source = "path/to/video.mp4"  # Or video file
    
    cap = cv2.VideoCapture(test_source)
    if not cap.isOpened():
        print("❌ Cannot open video source")
        return
    
    print(f"📹 Processing video from: {test_source}")
    print("   Press 'q' to quit")
    print("   Watch for:")
    print("   - Green labels = Pedestrians")
    print("   - Orange labels = Riders/Drivers")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, stats = processor.process_frame(frame, resize_scale=100, max_det=20)
        
        # Display
        cv2.imshow("Person Classification Test", processed_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Frame {frame_count}: {stats}")
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Test complete! Processed {frame_count} frames")
    print(f"   Total unique objects: {len(processor.unique_ids)}")
    print(f"   Class counts: {processor.class_counts}")

if __name__ == "__main__":
    test_person_classification()

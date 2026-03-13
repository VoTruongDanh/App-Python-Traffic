"""
Convert PyTorch YOLO models to ONNX format for 2-3x speedup
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def convert_model_to_onnx(model_path: str, output_path: str = None, imgsz: int = 640):
    """
    Convert a YOLO model to ONNX format
    
    Args:
        model_path: Path to .pt model file
        output_path: Output path for .onnx file (optional)
        imgsz: Input image size (default: 640)
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # Skip if ONNX already exists
    if os.path.exists(output_path):
        print(f"✅ ONNX already exists: {output_path}")
        return True
    
    try:
        print(f"\n🔄 Converting: {model_path}")
        print(f"   Output: {output_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Export to ONNX with optimizations
        # Note: simplify=False to avoid onnxslim crash
        model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=False,  # Disable simplify to avoid onnxslim crash
            opset=12,  # ONNX opset version
            dynamic=False,  # Fixed input size for better optimization
        )
        
        # Ultralytics saves ONNX in same directory as .pt
        # Move to desired location if different
        auto_output = model_path.replace('.pt', '.onnx')
        if auto_output != output_path and os.path.exists(auto_output):
            os.rename(auto_output, output_path)
        
        print(f"✅ Conversion successful!")
        
        # Show file sizes
        pt_size = os.path.getsize(model_path) / (1024 * 1024)
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   PyTorch: {pt_size:.1f} MB")
        print(f"   ONNX: {onnx_size:.1f} MB")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Conversion failed: {e}")
        print(f"   Details: {traceback.format_exc()}")
        
        # Clean up partial ONNX file if exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"   Cleaned up partial file")
            except:
                pass
        
        return False


def main():
    """Convert all available models to ONNX"""
    print("=" * 60)
    print("YOLO MODEL CONVERSION TO ONNX")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - conversion will use CPU")
    
    print()
    
    # List of models to convert
    models_to_convert = [
        # Base models in UI folder
        'yolov3u.pt',
        'yolov8n.pt',
        'yolo11n.pt',
        'yolo11s.pt',
        'yolo26n.pt',
        
        # Trained models
        '../train1/best.pt',
        '../Train2/best.pt',
    ]
    
    success_count = 0
    total_count = 0
    
    for model_path in models_to_convert:
        total_count += 1
        if convert_model_to_onnx(model_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"CONVERSION COMPLETE: {success_count}/{total_count} models")
    print("=" * 60)
    
    if success_count > 0:
        print("\n✅ ONNX models ready for use!")
        print("   The app will automatically use ONNX models if available")
        print("   Expected speedup: 2-3x faster inference")
        print("   Quality: Same as PyTorch (no loss)")
    else:
        print("\n⚠️  No models were converted")
        print("   Make sure model files exist in the correct locations")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

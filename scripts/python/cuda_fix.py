"""
CUDA NMS Fix - Force NMS to run on CPU while keeping model on GPU
This patches torchvision.ops.nms to use CPU tensors
"""

import torch
import torchvision.ops

# Save original NMS function
_original_nms = torchvision.ops.nms

def nms_cpu_wrapper(boxes, scores, iou_threshold):
    """
    Wrapper that moves tensors to CPU for NMS, then back to original device
    """
    device = boxes.device
    
    # Move to CPU
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()
    
    # Run NMS on CPU
    result = _original_nms(boxes_cpu, scores_cpu, iou_threshold)
    
    # Move result back to original device
    return result.to(device)

def apply_cuda_fix():
    """
    Apply the CUDA NMS fix by monkey-patching torchvision
    """
    torchvision.ops.nms = nms_cpu_wrapper
    print("✅ Applied CUDA NMS fix - NMS will run on CPU, inference on GPU")

def remove_cuda_fix():
    """
    Remove the fix and restore original NMS
    """
    torchvision.ops.nms = _original_nms
    print("✅ Removed CUDA NMS fix - Using original NMS")

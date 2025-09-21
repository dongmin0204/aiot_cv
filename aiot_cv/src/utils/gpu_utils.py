"""
GPU utilities for CUDA optimization and memory management.
"""

import numpy as np
from typing import Optional, Tuple
import logging

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def setup_cuda_optimization(memory_fraction: float = 0.8, benchmark: bool = True):
    """
    Setup CUDA optimization settings.
    
    Args:
        memory_fraction: GPU memory allocation fraction
        benchmark: Enable cuDNN benchmark for fixed input sizes
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping GPU optimization")
        return False
    
    try:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.backends.cudnn.benchmark = benchmark
        
        device = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"CUDA setup: {device}, {memory_gb:.1f}GB, fraction={memory_fraction}, benchmark={benchmark}")
        return True
    except Exception as e:
        logging.error(f"CUDA setup failed: {e}")
        return False


def gpu_mask_processing(mask: np.ndarray, operations: list = None) -> np.ndarray:
    """
    GPU-accelerated mask processing operations.
    
    Args:
        mask: Input mask (H, W)
        operations: List of operations ['dilate', 'erode', 'open', 'close']
        
    Returns:
        Processed mask
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return mask  # Fallback to CPU
    
    if operations is None:
        operations = ['dilate']
    
    try:
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).cuda().float().unsqueeze(0).unsqueeze(0)
        
        for op in operations:
            if op == 'dilate':
                # 3x3 dilation
                kernel = torch.ones(1, 1, 3, 3).cuda()
                mask_tensor = F.conv2d(mask_tensor, kernel, padding=1)
                mask_tensor = (mask_tensor > 0).float()
            elif op == 'erode':
                # 3x3 erosion (inverse of dilation)
                kernel = torch.ones(1, 1, 3, 3).cuda()
                mask_tensor = F.conv2d(mask_tensor, kernel, padding=1)
                mask_tensor = (mask_tensor == 9).float()  # All 9 neighbors must be 1
        
        # Convert back to numpy
        result = mask_tensor.squeeze().cpu().numpy()
        return (result * 255).astype(np.uint8)
        
    except Exception as e:
        logging.warning(f"GPU mask processing failed: {e}, falling back to CPU")
        return mask


def gpu_depth_filtering(depth: np.ndarray, mask: np.ndarray, 
                       z_min: float = 0.05, z_max: float = 3.0) -> Tuple[np.ndarray, dict]:
    """
    GPU-accelerated depth filtering and statistics.
    
    Args:
        depth: Depth image in meters
        mask: Object mask
        z_min: Minimum valid depth
        z_max: Maximum valid depth
        
    Returns:
        (filtered_depth, stats)
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        # CPU fallback
        valid = (depth > z_min) & (depth < z_max) & (mask > 0)
        valid_depth = depth[valid]
        stats = {
            'median_z': float(np.median(valid_depth)) if len(valid_depth) > 0 else 0.0,
            'valid_ratio': len(valid_depth) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0.0,
            'total_points': len(valid_depth)
        }
        return depth, stats
    
    try:
        # Convert to tensors
        depth_tensor = torch.from_numpy(depth).cuda().float()
        mask_tensor = torch.from_numpy(mask).cuda()
        
        # Apply filters
        valid = (depth_tensor > z_min) & (depth_tensor < z_max) & (mask_tensor > 0)
        valid_depth = depth_tensor[valid]
        
        # Calculate statistics on GPU
        if len(valid_depth) > 0:
            median_z = torch.median(valid_depth).item()
            valid_ratio = len(valid_depth) / torch.sum(mask_tensor > 0).item()
            total_points = len(valid_depth)
        else:
            median_z = 0.0
            valid_ratio = 0.0
            total_points = 0
        
        stats = {
            'median_z': median_z,
            'valid_ratio': valid_ratio,
            'total_points': total_points
        }
        
        # Apply filtering
        filtered_depth = depth_tensor.cpu().numpy()
        filtered_depth[~valid.cpu().numpy()] = 0
        
        return filtered_depth, stats
        
    except Exception as e:
        logging.warning(f"GPU depth filtering failed: {e}, falling back to CPU")
        # CPU fallback
        valid = (depth > z_min) & (depth < z_max) & (mask > 0)
        valid_depth = depth[valid]
        stats = {
            'median_z': float(np.median(valid_depth)) if len(valid_depth) > 0 else 0.0,
            'valid_ratio': len(valid_depth) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0.0,
            'total_points': len(valid_depth)
        }
        return depth, stats


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_info() -> dict:
    """Get GPU memory information."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {'available': False}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        'available': True,
        'allocated_gb': allocated,
        'cached_gb': cached,
        'total_gb': total,
        'free_gb': total - allocated
    }

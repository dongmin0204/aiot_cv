"""
Domain gap bridging utilities for robust reference matching.
Applies transformations to references to match real-world conditions.
"""

import cv2
import numpy as np
from typing import List, Optional


class DomainBridge:
    """Domain gap bridging for reference image augmentation."""
    
    def __init__(self, profile: str = "metal_lowres", seed: int = 123):
        """
        Initialize domain bridge.
        
        Args:
            profile: Augmentation profile ("metal_lowres", "indoor", "outdoor")
            seed: Random seed for reproducible augmentation
        """
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        
    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply domain-specific augmentation to an image.
        
        Args:
            img: Input image (RGB)
            
        Returns:
            Augmented image
        """
        if self.profile == "metal_lowres":
            return self._apply_metal_lowres(img)
        elif self.profile == "indoor":
            return self._apply_indoor(img)
        elif self.profile == "outdoor":
            return self._apply_outdoor(img)
        else:
            return img  # No augmentation
    
    def _apply_metal_lowres(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation for metallic objects in low-res conditions."""
        out = img.copy()
        
        # (1) Brightness/contrast adjustment
        alpha = self.rng.uniform(0.85, 1.20)  # contrast
        beta = self.rng.uniform(-20, 20)      # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        
        # (2) Blur (Gaussian or motion blur)
        if self.rng.random() < 0.5:
            # Gaussian blur
            k = self.rng.integers(3, 5) * 2 + 1
            out = cv2.GaussianBlur(out, (k, k), self.rng.uniform(0.4, 1.0))
        else:
            # Motion blur
            k = 7
            kernel = np.zeros((k, k), np.float32)
            kernel[k//2, :] = 1.0/k
            out = cv2.filter2D(out, -1, kernel)
        
        # (3) JPEG compression artifacts simulation
        if self.rng.random() < 0.6:
            quality = int(self.rng.uniform(40, 70))
            _, encoded = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # (4) Weak sharpening
        if self.rng.random() < 0.5:
            blur = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
        
        # (5) Slight noise
        if self.rng.random() < 0.3:
            noise = self.rng.normal(0, 5, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return out
    
    def _apply_indoor(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation for indoor lighting conditions."""
        out = img.copy()
        
        # Warm/cool lighting shifts
        temp_shift = self.rng.uniform(-15, 15)
        if temp_shift > 0:  # Warm
            out[:, :, 0] = np.clip(out[:, :, 0] + temp_shift, 0, 255)  # R
        else:  # Cool
            out[:, :, 2] = np.clip(out[:, :, 2] - temp_shift, 0, 255)  # B
        
        # Contrast adjustment
        alpha = self.rng.uniform(0.9, 1.1)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=0)
        
        return out
    
    def _apply_outdoor(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation for outdoor lighting conditions."""
        out = img.copy()
        
        # Brightness variation (sun/shadow)
        beta = self.rng.uniform(-30, 30)
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=beta)
        
        # Saturation adjustment
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.rng.uniform(0.8, 1.2), 0, 255)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return out


def apply_domain_bridge(img: np.ndarray, profile: str = "metal_lowres", seed: int = 123) -> np.ndarray:
    """
    Convenience function to apply domain bridging.
    
    Args:
        img: Input image (RGB)
        profile: Augmentation profile
        seed: Random seed
        
    Returns:
        Augmented image
    """
    bridge = DomainBridge(profile, seed)
    return bridge.apply_augmentation(img)

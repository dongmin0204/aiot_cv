#!/usr/bin/env python3
"""
Simple mask generation script for reference images using PIL and scipy.
This script avoids OpenCV dependency issues.
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage

def generate_masks_for_images(image_files, input_dir, output_dir):
    """Generate masks for a list of image files."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_files)} images")
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        # Get corresponding mask filename
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(output_dir, mask_name)
        
        print(f"Processing: {img_file}")
        
        try:
            # Load image
            img = Image.open(img_path)
            w, h = img.size
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Create mask based on image analysis
            mask = create_mask_from_image(img_array)
            
            # Apply morphological operations to clean up mask
            mask = clean_mask(mask)
            
            # Ensure mask is not too small
            mask_area = np.sum(mask > 0)
            if mask_area < (w * h * 0.05):  # If mask is too small
                print(f"  -> Mask too small ({mask_area} pixels), creating center region")
                mask = create_center_mask(w, h)
            
            # Save mask
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(mask_path)
            
            mask_area = np.sum(mask > 0)
            mask_percentage = (mask_area / (w * h)) * 100
            print(f"  -> Created mask: {mask_name} (area: {mask_area} pixels, {mask_percentage:.1f}%)")
            
        except Exception as e:
            print(f"  -> Error processing {img_file}: {e}")
            # Create a fallback mask
            img = Image.open(img_path)
            w, h = img.size
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(mask_path)
            print(f"  -> Created fallback mask: {mask_name}")

def create_mask_from_image(img_array):
    """Create mask from image array using color analysis."""
    
    if len(img_array.shape) == 3:
        # RGB image - analyze colors
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # For hot glue gun (pink objects)
        # Create mask for pink colors
        pink_mask = (
            (r > 180) & (r < 255) &  # High red
            (g > 100) & (g < 200) &  # Medium green  
            (b > 150) & (b < 255)    # High blue
        )
        
        # Include white parts (trigger, nozzle, etc.)
        white_mask = (
            (r > 200) & (g > 200) & (b > 200)
        )
        
        # Combine masks
        mask = (pink_mask | white_mask).astype(np.uint8) * 255
        
    else:
        # Grayscale image - use threshold
        gray = img_array
        # Use adaptive threshold based on image statistics
        threshold = np.mean(gray) * 0.9
        mask = (gray < threshold).astype(np.uint8) * 255
    
    return mask

def clean_mask(mask):
    """Clean mask using morphological operations."""
    
    # Convert to binary
    binary_mask = mask > 0
    
    # Remove small objects
    binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((5,5)))
    
    # Fill holes
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # Smooth the mask
    binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3,3)))
    
    # Convert back to uint8
    return binary_mask.astype(np.uint8) * 255

def create_center_mask(w, h):
    """Create a center region mask as fallback."""
    center_x, center_y = w // 2, h // 2
    mask_size = min(w, h) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    
    y1 = max(0, center_y - mask_size // 2)
    y2 = min(h, center_y + mask_size // 2)
    x1 = max(0, center_x - mask_size // 2)
    x2 = min(w, center_x + mask_size // 2)
    
    mask[y1:y2, x1:x2] = 255
    return mask

def main():
    """Main function to generate masks for hot glue gun images."""
    
    # Define paths
    input_dir = "aiot_cv/data/references/images"
    output_dir = "aiot_cv/data/references/masks"
    
    # Hot glue gun image files
    gluegun_images = [
        "ref_gluegun_01.png",
        "ref_gluegun_02.png", 
        "ref_gluegun_04.png",
        "ref_gluegun_05.png",
        "ref_gluegun_06.png",
        "ref_gluegun_07.png"
    ]
    
    print("Creating masks for hot glue gun reference images...")
    print("Using simple PIL + scipy approach (no OpenCV)")
    
    # Generate masks
    generate_masks_for_images(gluegun_images, input_dir, output_dir)
    
    print("\n✅ Mask generation completed!")

if __name__ == "__main__":
    main()
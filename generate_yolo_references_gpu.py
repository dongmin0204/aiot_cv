#!/usr/bin/env python3
"""
Generate reference images and masks using YOLO GPU model from tool_seg dataset.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    exit(1)

def generate_yolo_references():
    """Generate reference images and masks using YOLO segmentation with GPU."""
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    train_images_dir = "tool_seg/train/images"
    output_dir = "aiot_cv/data/references/screwdriver"
    yolo_weights = "train_results/exp1/weights/last.pt"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    model.to(device)
    
    # Get reference images (ref_01.png ~ ref_44.png)
    ref_images = []
    for i in range(1, 117):  # ref_01.png to ref_44.png
        ref_images.append(f"ref_{i:02d}.png")
    
    print(f"Processing {len(ref_images)} reference images")
    
    successful_count = 0
    
    for i, img_file in enumerate(ref_images):
        img_path = os.path.join(output_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        print(f"Processing {i+1}/{len(ref_images)}: {img_file}")
        
        try:
            # Load image using PIL
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Run YOLO inference
            results = model(image_array, conf=0.3, device=device, verbose=False)
            
            # Find screwdriver detection
            best_detection = None
            best_conf = 0
            
            for result in results:
                if result.masks is not None and len(result.masks) > 0:
                    for j, mask in enumerate(result.masks.data):
                        # Get class name
                        if hasattr(result, 'names') and j < len(result.boxes.cls):
                            class_id = int(result.boxes.cls[j])
                            class_name = result.names.get(class_id, "unknown")
                            
                            # Check if it's screwdriver
                            if 'screwdriver' in class_name.lower():
                                conf = float(result.boxes.conf[j])
                                if conf > best_conf:
                                    best_conf = conf
                                    best_detection = mask.cpu().numpy()
            
            if best_detection is not None:
                # Resize mask to original image size
                h, w = image_array.shape[:2]
                mask_resized = np.array(Image.fromarray(best_detection).resize((w, h), Image.NEAREST))
                
                # Convert to binary mask (0/255)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Generate output filename for mask
                ref_mask_name = f"ref_{i+1:02d}_mask.png"
                ref_mask_path = os.path.join(output_dir, ref_mask_name)
                
                # Save mask (reference image is already saved)
                mask_image = Image.fromarray(mask_binary, mode='L')
                mask_image.save(ref_mask_path)
                
                print(f"  -> Saved: {ref_mask_name} (conf: {best_conf:.3f})")
                successful_count += 1
            else:
                print(f"  -> No screwdriver detection found")
                
        except Exception as e:
            print(f"  -> Error processing {img_file}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {successful_count} mask files")
    print(f"Output directory: {output_dir}")
    
    # List generated mask files
    if os.path.exists(output_dir):
        mask_files = [f for f in sorted(os.listdir(output_dir)) if f.endswith('_mask.png')]
        print(f"Generated mask files: {len(mask_files)}")
        for f in mask_files:
            print(f"  - {f}")

def main():
    print("Generating masks for reference images using YOLO GPU segmentation...")
    generate_yolo_references()
    print("\n✅ Mask generation completed!")

if __name__ == "__main__":
    main()

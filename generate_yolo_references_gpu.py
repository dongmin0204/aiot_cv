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
    
    # Get screwdriver images
    screwdriver_images = [
        "1-combination-screwdriver-set-shafiq-internatonal-original-imag6yk3uzedgskt_jpeg.rf.66b645ae0f4e8fe9d1e1a2244fc869f2.jpg",
        "1-combination-screwdriver-set-shafiq-internatonal-original-imag6yk3uzedgskt_jpeg.rf.9b995cfd7e7707ea3f4b3844b41d1a81.jpg",
        "1-combination-screwdriver-set-shafiq-internatonal-original-imag6yk3uzedgskt_jpeg.rf.ec09c860be97341d516df165a84fe9b4.jpg",
        "1621-8-cutting-plier-and-831-screwdriver-set-taparia-original-imafgj9gm8rceccg_jpeg.rf.47745379f002f7bc9651779a4a3003dd.jpg",
        "1621-8-cutting-plier-and-831-screwdriver-set-taparia-original-imafgj9gm8rceccg_jpeg.rf.4953ccdc1b41bbef51ff9df4a2b82892.jpg",
        "1621-8-cutting-plier-and-831-screwdriver-set-taparia-original-imafgj9gm8rceccg_jpeg.rf.abef03510ab376eaa3dc95e68a2d8edc.jpg",
        "8-set-of-2-starter-kit-screwdriver-2in1-8inch-pliers-hand-tool-original-imag4q3tbgyr2efz_jpeg.rf.22fe99499f567c476b7ffa30a65c6e18.jpg",
        "8-set-of-2-starter-kit-screwdriver-2in1-8inch-pliers-hand-tool-original-imag4q3tbgyr2efz_jpeg.rf.981472097a60cb990243d908efe8afd4.jpg",
        "8-set-of-2-starter-kit-screwdriver-2in1-8inch-pliers-hand-tool-original-imag4q3tbgyr2efz_jpeg.rf.a2bab99ac678dd72c0b755491cbafadb.jpg",
        "Screwdriver-415-_JPEG.rf.05fe524fe035e48dae8a9542f3ac9033.jpg",
        "Screwdriver-415-_JPEG.rf.1a8c31ccd7b0ff7f63c582546b01f044.jpg",
        "Screwdriver-415-_JPEG.rf.cc75490616f138aa51f1755d4f88856c.jpg",
        "Screwdriver-456-_JPEG.rf.00a0d4abe5fe9f3ccd74ca612769be81.jpg",
        "Screwdriver-456-_JPEG.rf.22c174a60702204eb1b17f465cffa93b.jpg",
        "Screwdriver-456-_JPEG.rf.5918bda8849b749d85eef64dbbfcce52.jpg",
        "Screwdriver-465-_JPEG.rf.23dfe0ce352772d8b534e0dc638ba9b5.jpg",
        "Screwdriver-465-_JPEG.rf.3a64e777eb9df17f80718a295da4199b.jpg",
        "Screwdriver-465-_JPEG.rf.4cf6f380e68dfda5036945bb0f1d288c.jpg",
        "Screwdriver-477-_JPEG.rf.6312d94798555218b5d2766fbd5679bb.jpg",
        "Screwdriver-477-_JPEG.rf.99772baa0a5774bf5ecdac3ea4c05b8f.jpg"
    ]
    
    print(f"Processing {len(screwdriver_images)} screwdriver images")
    
    successful_count = 0
    
    for i, img_file in enumerate(screwdriver_images):
        img_path = os.path.join(train_images_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        print(f"Processing {i+1}/{len(screwdriver_images)}: {img_file}")
        
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
                
                # Generate output filenames
                ref_num = i + 1
                ref_img_name = f"ref_{ref_num:02d}.png"
                ref_mask_name = f"ref_{ref_num:02d}_mask.png"
                
                ref_img_path = os.path.join(output_dir, ref_img_name)
                ref_mask_path = os.path.join(output_dir, ref_mask_name)
                
                # Save reference image (RGB format)
                image.save(ref_img_path)
                
                # Save mask
                mask_image = Image.fromarray(mask_binary, mode='L')
                mask_image.save(ref_mask_path)
                
                print(f"  -> Saved: {ref_img_name}, {ref_mask_name} (conf: {best_conf:.3f})")
                successful_count += 1
            else:
                print(f"  -> No screwdriver detection found")
                
        except Exception as e:
            print(f"  -> Error processing {img_file}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {successful_count} reference image pairs")
    print(f"Output directory: {output_dir}")
    
    # List generated files
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        print(f"Generated files: {len(files)}")
        for f in files:
            print(f"  - {f}")

def main():
    print("Generating reference images and masks using YOLO GPU segmentation...")
    generate_yolo_references()
    print("\n✅ Reference generation completed!")

if __name__ == "__main__":
    main()

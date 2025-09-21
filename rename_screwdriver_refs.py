#!/usr/bin/env python3
"""
Rename screwdriver reference images to ref_XX.png format.
"""

import os
import shutil
from pathlib import Path

def rename_screwdriver_refs():
    """Rename screwdriver images to ref_XX.png format."""
    ref_dir = "aiot_cv/data/references/screwdriver"
    
    # Get all screwdriver_* files
    screwdriver_files = []
    for f in os.listdir(ref_dir):
        if f.startswith('screwdriver_') and f.endswith('.png'):
            screwdriver_files.append(f)
    
    # Sort them for consistent ordering
    screwdriver_files.sort()
    
    print(f"Found {len(screwdriver_files)} screwdriver images to rename")
    
    # Find the highest existing ref_ number
    existing_refs = []
    for f in os.listdir(ref_dir):
        if f.startswith('ref_') and f.endswith('.png') and not f.endswith('_mask.png'):
            try:
                num = int(f.replace('ref_', '').replace('.png', ''))
                existing_refs.append(num)
            except ValueError:
                continue
    
    if existing_refs:
        next_num = max(existing_refs) + 1
    else:
        next_num = 1
    
    print(f"Starting numbering from ref_{next_num:03d}.png")
    
    # Rename files
    renamed_count = 0
    for i, old_file in enumerate(screwdriver_files):
        old_path = os.path.join(ref_dir, old_file)
        new_file = f"ref_{next_num + i:03d}.png"
        new_path = os.path.join(ref_dir, new_file)
        
        try:
            shutil.move(old_path, new_path)
            print(f"Renamed: {old_file} -> {new_file}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {old_file}: {e}")
    
    print(f"Successfully renamed {renamed_count}/{len(screwdriver_files)} files")

def main():
    print("Renaming screwdriver reference images...")
    rename_screwdriver_refs()
    print("Done!")

if __name__ == "__main__":
    main()

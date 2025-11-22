import os
import shutil
import pandas as pd
from pathlib import Path

# Read the text mapping file
# Read as raw lines since the image column can have multiple values
with open("JSON-image_17.txt", "r") as f:
    lines = f.readlines()

# Skip header line
lines = lines[1:]

# Define source and destination directories
info_source_dir = "../dataset/info/"
image_source_dir = "../dataset/images/"
destination_dir = "organized_files"

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Create subdirectories for info and images
info_dest = os.path.join(destination_dir, "info")
image_dest = os.path.join(destination_dir, "images")
os.makedirs(info_dest, exist_ok=True)
os.makedirs(image_dest, exist_ok=True)

# Track statistics
copied_info = 0
copied_images = 0
missing_files = []

# Process each line
for line in lines:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        continue

    influencer = parts[0]
    info_file = parts[1]
    # Everything after the second tab is the image list
    image_list_str = "\t".join(parts[2:])

    # Parse the image list (it's in format ['image1.jpg', 'image2.jpg'])
    image_files = eval(image_list_str)

    # Copy info file with influencer prefix
    info_filename = f"{influencer}-{info_file}"
    info_src = os.path.join(info_source_dir, info_filename)
    info_dst = os.path.join(info_dest, info_filename)

    if os.path.exists(info_src):
        shutil.copy2(info_src, info_dst)
        copied_info += 1
    else:
        missing_files.append(f"Info: {info_src}")

    # Copy image files with influencer prefix
    for img_file in image_files:
        img_filename = f"{influencer}-{img_file}"
        img_src = os.path.join(image_source_dir, img_filename)
        img_dst = os.path.join(image_dest, img_filename)

        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
            copied_images += 1
        else:
            missing_files.append(f"Image: {img_src}")

# Print summary
print(f"\n{'='*50}")
print(f"File Organization Complete!")
print(f"{'='*50}")
print(f"Info files copied: {copied_info}")
print(f"Image files copied: {copied_images}")
print(f"Total files copied: {copied_info + copied_images}")

if missing_files:
    print(f"\nWarning: {len(missing_files)} files were not found:")
    for file in missing_files[:10]:  # Show first 10 missing files
        print(f"  - {file}")
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")
else:
    print("\nAll files were successfully copied!")

print(f"\nFiles organized in: {os.path.abspath(destination_dir)}")

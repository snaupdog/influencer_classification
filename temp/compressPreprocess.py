import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Paths defined
base_dir = "/home/m1h1r/Documents/[2] dev/influencer-net"
ssd_dir = "/run/media/m1h1r/04E884E1E884D1FA"
image_dir = os.path.join(base_dir, "image")
mapping_file = os.path.join(base_dir, "data_sampling/smallInfluencers.txt")
output_base = os.path.join(ssd_dir, "compressedPreprocessedImages")

# Define the target image size for EfficientNetV2‑S (224×224)
target_size = (224, 224)


def load_mapping(mapping_path):
    # dictionary for username -> category
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                username = parts[0]
                category = parts[1]
                mapping[username] = category
    return mapping


def process_and_save_image(image_path, username, category):
    """Process an image and save it as a compressed numpy file."""
    # process and save as compressed numpy file
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(img_batch)[0]  # Remove batch dimension
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Create output directory for the category if it doesn't exist
    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save the compressed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(category_dir, os.path.splitext(filename)[0] + ".npz")
    try:
        np.savez_compressed(output_path, preprocessed)
        print(f"Saved compressed preprocessed image to: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def main():
    # return dictionary to map username to category
    mapping = load_mapping(mapping_file)

    # Process each image in the image directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            if "-" not in filename:
                print(f"Filename {filename} does not have the expected '-' separator.")
                continue

            username = filename.split("-")[0]
            if username not in mapping:
                print(f"Username {username} not found in mapping. Skipping {filename}.")
                continue

            category = mapping[username]
            image_path = os.path.join(image_dir, filename)
            process_and_save_image(image_path, username, category)


if __name__ == "__main__":
    main()

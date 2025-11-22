import os
import csv
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# -------------------------------
# PATHS
# -------------------------------
# Image folder relative to script
image_dir = os.path.abspath("../dataset/images")

# Mapping CSV relative to script
mapping_file = os.path.abspath("../dataset/influencers.csv")

# Output folder in the current working directory
output_base = os.path.join(os.getcwd(), "compressedPreprocessedImages")

# EfficientNetV2‑S expects 224x224 images
target_size = (224, 224)


# -------------------------------
# FUNCTIONS
# -------------------------------


def load_mapping(mapping_path):
    """
    Load CSV mapping file with header:
    Username,Category,Followers,Followees,Posts
    Returns a dictionary: username -> category
    """
    mapping = {}
    with open(mapping_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            username = row["Username"]
            category = row["Category"]
            mapping[username] = category
    return mapping


def process_and_save_image(image_path, username, category):
    """Process an image and save it as a compressed numpy file."""
    try:
        # Load image and convert to RGB
        img = Image.open(image_path).convert("RGB")
        # Resize to target size
        img = img.resize(target_size)
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        # Add batch dimension for preprocessing
        img_batch = np.expand_dims(img_array, axis=0)
        # Preprocess for EfficientNetV2
        preprocessed = preprocess_input(img_batch)[0]  # remove batch dimension
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Create category directory if it doesn't exist
    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save as compressed .npz
    filename = os.path.basename(image_path)
    output_path = os.path.join(category_dir, os.path.splitext(filename)[0] + ".npz")
    try:
        np.savez_compressed(output_path, preprocessed)
        print(f"Saved compressed preprocessed image to: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def main():
    # Load CSV mapping
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


# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    main()

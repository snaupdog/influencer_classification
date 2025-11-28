import os
import csv
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# -------------------------------
# PATHS
# -------------------------------
image_dir = os.path.abspath("../dataset/images")
mapping_file = os.path.abspath("../dataset/influencers.csv")
output_base = os.path.join(os.getcwd(), "compressedPreprocessedImages")
target_size = (224, 224)

# -------------------------------
# FUNCTIONS
# -------------------------------


def load_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            username = row["Username"]
            category = row["Category"]
            mapping[username] = category
    return mapping


def process_and_save_image(image_path, username, category):
    """Process an image and return True if successful, False otherwise."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(img_batch)[0]
    except Exception as e:
        return False  # failed processing

    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    output_path = os.path.join(
        category_dir, os.path.splitext(os.path.basename(image_path))[0] + ".npz"
    )
    try:
        np.savez_compressed(output_path, preprocessed)
        return True
    except Exception as e:
        return False


def main():
    mapping = load_mapping(mapping_file)

    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if "-" not in filename:
            skipped_count += 1
            continue

        username = filename.split("-")[0]
        if username not in mapping:
            skipped_count += 1
            continue

        category = mapping[username]
        image_path = os.path.join(image_dir, filename)
        if process_and_save_image(image_path, username, category):
            processed_count += 1
        else:
            skipped_count += 1

        # Print progress
        print(
            f"Processed: {processed_count}, Skipped: {skipped_count}", end="\r"
        )  # overwrite same line

    print(
        f"\nFinished! Total processed: {processed_count}, Total skipped: {skipped_count}"
    )


if __name__ == "__main__":
    main()

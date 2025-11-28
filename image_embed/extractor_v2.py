import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Configuration
BATCH_SIZE = 32

# Get the folder where the current script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "image_feature_extractor.keras")
INPUT_DIR = os.path.join(BASE_DIR, "compressedPreprocessedImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "image_features3")

CLASS_NAMES = [
    "beauty",
    "family",
    "fashion",
    "fitness",
    "food",
    "interior",
    "other",
    "pet",
    "travel",
]

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

# Load feature extractor
feature_extractor = load_model(MODEL_PATH)


def load_npz_file(file_path):
    try:
        data = np.load(file_path)
        image = data[data.files[0]]

        # CHECK FOR NaN VALUES IN LOADED IMAGE
        if np.isnan(image).any():
            print(f"⚠️ WARNING: NaN values found in {file_path}")
            return None

        # CHECK FOR INF VALUES
        if np.isinf(image).any():
            print(f"⚠️ WARNING: Inf values found in {file_path}")
            return None

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # ENSURE PROPER DATA TYPE AND VALUE RANGE
        image = image.astype(np.float32)

        # Apply EfficientNetV2 preprocessing here (not in preprocessing script)
        image = preprocess_input(image)

        # Check if values are in reasonable range after preprocessing
        if np.isnan(image).any():
            print(f"⚠️ WARNING: NaN values after preprocessing in {file_path}")
            return None

        return image
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def process_batch(batch_images, batch_output_paths, feature_extractor):
    try:
        stacked = np.vstack(batch_images)

        # CHECK FOR NaN BEFORE PREDICTION
        if np.isnan(stacked).any():
            print("❌ ERROR: NaN values in stacked batch before prediction!")
            return

        features = feature_extractor.predict(stacked, verbose=0)

        # CHECK FOR NaN IN OUTPUT
        if np.isnan(features).any():
            print("❌ ERROR: Model produced NaN values in output!")
            nan_count = np.isnan(features).sum()
            print(
                f"   Found {nan_count} NaN values out of {features.size} total values"
            )
            return

        for i, output_path in enumerate(batch_output_paths):
            np.savez_compressed(output_path, features=features[i])
    except Exception as e:
        print(f"❌ Error processing batch: {e}")


def extract_features(
    input_dir, output_dir, class_names, feature_extractor, batch_size=32
):
    processed_count = 0
    skipped_count = 0

    for class_name in class_names:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_input_dir):
            print(f"⏭️  Skipping {class_name} (directory not found)")
            continue

        npz_files = [f for f in os.listdir(class_input_dir) if f.endswith(".npz")]
        print(f"\n📁 Processing {class_name}: {len(npz_files)} files")

        batch_images, batch_output_paths = [], []

        for file_name in npz_files:
            input_path = os.path.join(class_input_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(class_output_dir, f"{base_name}_features.npz")

            if os.path.exists(output_path):
                continue

            image = load_npz_file(input_path)
            if image is None:
                skipped_count += 1
                continue

            batch_images.append(image)
            batch_output_paths.append(output_path)

            if len(batch_images) >= batch_size:
                process_batch(batch_images, batch_output_paths, feature_extractor)
                processed_count += len(batch_images)
                batch_images, batch_output_paths = [], []

        if batch_images:
            process_batch(batch_images, batch_output_paths, feature_extractor)
            processed_count += len(batch_images)

    print(
        f"\n✅ Extraction complete! Processed: {processed_count}, Skipped: {skipped_count}"
    )


# Run extraction
extract_features(INPUT_DIR, OUTPUT_DIR, CLASS_NAMES, feature_extractor, BATCH_SIZE)

# Create metadata CSV
metadata = []
for class_name in CLASS_NAMES:
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    if not os.path.exists(class_output_dir):
        continue
    for f in os.listdir(class_output_dir):
        if f.endswith("_features.npz"):
            metadata.append(
                {
                    "class": class_name,
                    "original_filename": f.replace("_features.npz", ""),
                    "feature_path": os.path.join(class_name, f),
                }
            )

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(OUTPUT_DIR, "feature_metadata.csv"), index=False)
print(f"✅ Metadata saved: {len(metadata_df)} features catalogued")

import numpy as np
import pandas as pd
import os

# ------------------------------
# Input / Output paths
# ------------------------------
csv_file = "../text_embed/processed_posts.csv"
image_base_folder = "../image_embed/image_features"
category_csv = "../image_embed/image_features/feature_metadata.csv"
output_base = "combined_feature_vectors"

# ------------------------------
# Validate CSV paths
# ------------------------------
assert os.path.exists(csv_file), f"Text CSV not found: {csv_file}"
assert os.path.exists(category_csv), f"Category metadata not found: {category_csv}"

print("Loading CSVs...")
df = pd.read_csv(csv_file)
category_df = pd.read_csv(category_csv)

print("Loaded text embeddings:", df.shape)
print("Loaded category metadata:", category_df.shape)

# Remove duplicates
category_df = category_df.drop_duplicates(subset=["original_filename"])

# Map from filename → class
category_map = dict(zip(category_df["original_filename"], category_df["class"]))

# Identify text embedding columns
dim_columns = [col for col in df.columns if col.startswith("dim_")]
print("Detected embedding columns:", len(dim_columns))


# ------------------------------
# Helper: find image feature file
# ------------------------------
def find_image_feature_path(base_folder, feature_name):
    for root, dirs, files in os.walk(base_folder):
        if f"{feature_name}.npz" in files:
            return os.path.join(root, f"{feature_name}.npz")
    return None


# ------------------------------
# Helper: safe check & replace NaN/Inf
# ------------------------------
def clean_array(arr):
    try:
        arr = arr.astype(float).reshape(1, -1)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr
    except Exception:
        return None


# ------------------------------
# Counters
# ------------------------------
processed_count = 0
skipped_count = 0

# ------------------------------
# Main loop
# ------------------------------
for idx, row in df.iterrows():
    post_id = row["info_file"]
    print(f"\nProcessing post: {post_id}")

    # ------------------------------
    # Extract and clean text embedding
    # ------------------------------
    text_embedding = clean_array(row[dim_columns].values)
    if text_embedding is None:
        print("❌ Text embedding invalid. Skipping...")
        skipped_count += 1
        continue

    # ------------------------------
    # Clean post id and find image feature
    # ------------------------------
    clean_post_id = post_id.replace(".info", "").replace("_caption", "")
    modified_post_id = clean_post_id + "_features"

    image_feature_path = find_image_feature_path(image_base_folder, modified_post_id)
    if not image_feature_path:
        print(f"❌ Image feature file not found for {modified_post_id}. Skipping...")
        skipped_count += 1
        continue

    image_data = np.load(image_feature_path)
    if "features" not in image_data:
        print(f"❌ Missing 'features' key in {image_feature_path}. Skipping...")
        skipped_count += 1
        continue

    # Clean image embedding
    image_feature = clean_array(image_data["features"])
    if image_feature is None:
        print(f"❌ Image embedding invalid. Skipping...")
        skipped_count += 1
        continue

    # ------------------------------
    # Combine features + ReLU
    # ------------------------------
    combined_features = np.concatenate([image_feature, text_embedding], axis=1)
    combined_features_relu = np.maximum(0, combined_features)

    # ------------------------------
    # Determine category and create folder
    # ------------------------------
    category = category_map.get(clean_post_id, "unknown")
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # Save combined features
    # ------------------------------
    output_path = os.path.join(output_dir, f"{clean_post_id}.npz")
    np.savez_compressed(
        output_path, features=combined_features_relu.flatten(), post_id=post_id
    )
    processed_count += 1
    print(f"✅ Saved combined features to: {output_path}")

# ------------------------------
# Final summary
# ------------------------------
print("\n🎉 All posts processed!")
print(f"✅ Successfully processed posts: {processed_count}")
print(f"⚠️ Skipped posts: {skipped_count}")

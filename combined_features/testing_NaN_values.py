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
# Helper function: safe check for NaN/Inf
# ------------------------------
def is_finite_array(arr):
    """
    Safely check finiteness even if dtype is object/string.
    Returns False if conversion to float fails.
    """
    try:
        arr = arr.astype(float)
        return np.isfinite(arr).all()
    except Exception:
        return False


# ------------------------------
# Find image feature file
# ------------------------------
def find_image_feature_path(base_folder, feature_name):
    for root, dirs, files in os.walk(base_folder):
        if f"{feature_name}.npz" in files:
            return os.path.join(root, f"{feature_name}.npz")
    return None


# ------------------------------
# Main loop
# ------------------------------
for idx, row in df.iterrows():
    post_id = row["info_file"]
    print(f"\nProcessing post: {post_id}")

    # Extract text embedding
    text_embedding = row[dim_columns].values

    # Safely convert text to float array
    try:
        text_embedding = text_embedding.astype(float).reshape(1, -1)
    except Exception:
        print("❌ Text embedding contains non-numeric values. Skipping...")
        continue

    # Clean id
    clean_post_id = post_id.replace(".info", "").replace("_caption", "")
    modified_post_id = clean_post_id + "_features"

    # Load image features
    image_feature_path = find_image_feature_path(image_base_folder, modified_post_id)
    if not image_feature_path:
        print(f"❌ Image feature file not found for {modified_post_id}. Skipping...")
        continue

    image_data = np.load(image_feature_path)

    if "features" not in image_data:
        print(f"❌ Missing 'features' key in {image_feature_path}. Skipping...")
        continue

    image_feature = image_data["features"]

    # Ensure numeric
    try:
        image_feature = image_feature.astype(float).reshape(1, -1)
    except Exception:
        print("❌ Image feature contains non-numeric data. Skipping...")
        continue

    # ------------------------------
    # Check for NaN or Inf in either vector
    # ------------------------------
    text_ok = is_finite_array(text_embedding)
    image_ok = is_finite_array(image_feature)

    if not text_ok or not image_ok:
        print("⚠️ Invalid values detected:")
        print("   → Text embedding has NaN/Inf?", not text_ok)
        print("   → Image embedding has NaN/Inf?", not image_ok)
        print("   Skipping this post.")
        continue

    # Combine and apply ReLU
    combined_features = np.concatenate([image_feature, text_embedding], axis=1)
    combined_features_relu = np.maximum(0, combined_features)

    # Determine category
    category = category_map.get(clean_post_id, "unknown")

    # Output folder
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    # Save
    output_path = os.path.join(output_dir, f"{clean_post_id}.npz")
    np.savez_compressed(
        output_path, features=combined_features_relu.flatten(), post_id=post_id
    )

    print(f"✅ Saved combined features to: {output_path}")

print("\n🎉 All posts processed!")

import numpy as np
import pandas as pd
import os

# Input files
csv_file = "../text_embed/processed_posts.csv"
image_base_folder = "../image_embed/image_features"
category_csv = "../image_embed/image_features/feature_metadata.csv"
output_base = "combined_feature_vectors"

# fake input files


# Load CSVs
df = pd.read_csv(csv_file)
category_df = pd.read_csv(category_csv)

# Remove duplicates if needed
category_df = category_df.drop_duplicates(subset=["original_filename"])

# Map from original_filename to class (category)
category_map = dict(zip(category_df["original_filename"], category_df["class"]))

# Get all text embedding columns
dim_columns = [col for col in df.columns if col.startswith("dim_")]


# Function to search recursively for the .npz file
def find_image_feature_path(base_folder, feature_name):
    for root, dirs, files in os.walk(base_folder):
        if f"{feature_name}.npz" in files:
            return os.path.join(root, f"{feature_name}.npz")
    return None


# Keep track of skipped posts
skipped_posts = []

# Process each row
for idx, row in df.iterrows():
    post_id = row["info_file"]
    print(f"\nProcessing post: {post_id}")

    # Get text embedding
    text_embedding = row[dim_columns].values.reshape(1, -1)

    # Convert post_id to match feature file naming
    base_post_id = post_id.replace(".info", "")  # strip .info
    modified_post_id = f"{base_post_id}_features"  # append _features

    # Find image feature file
    image_feature_path = find_image_feature_path(image_base_folder, modified_post_id)

    if not image_feature_path or not os.path.exists(image_feature_path):
        print(f"❌ Image feature file for {modified_post_id} not found. Skipping...")
        skipped_posts.append((post_id, "missing file"))
        continue

    # Load image features
    try:
        image_data = np.load(image_feature_path)
        if "features" not in image_data:
            print(f"❌ 'features' key missing in {image_feature_path}. Skipping...")
            skipped_posts.append((post_id, "missing features key"))
            continue
        image_feature = image_data["features"].reshape(1, -1)
    except Exception as e:
        print(f"❌ Error loading {image_feature_path}: {e}. Skipping...")
        skipped_posts.append((post_id, f"load error: {e}"))
        continue

    # Combine features
    combined_features = np.concatenate([image_feature, text_embedding], axis=1)

    # Replace NaN and Inf with 0
    combined_features = np.nan_to_num(
        combined_features, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Apply ReLU
    combined_features_relu = np.maximum(0, combined_features)

    # Determine category
    category = category_map.get(base_post_id, "unknown")  # fallback to 'unknown'

    # Create category folder if needed
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    # Save combined features
    output_path = os.path.join(output_dir, f"{base_post_id}.npz")
    np.savez_compressed(
        output_path, features=combined_features_relu.flatten(), post_id=post_id
    )
    print(f"✅ Saved combined features to {output_path}")

# Summary of skipped posts
if skipped_posts:
    print("\n⚠️ Some posts were skipped:")
    for post, reason in skipped_posts:
        print(f" - {post}: {reason}")

print("\n✅ All features processed and saved by category.")

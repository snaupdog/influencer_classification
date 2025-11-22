import numpy as np
import pandas as pd
import os

# ------------------------------
# Input / Output paths
# ------------------------------
csv_file = "../text_embed/processed_posts.csv"
image_base_folder = "../image_embed/image_features"
category_csv = "../image_embed/image_features/feature_metadata.csv"  # replace with your actual filename
output_base = "combined_feature_vectors"

# ------------------------------
# Load CSVs
# ------------------------------
df = pd.read_csv(csv_file)
category_df = pd.read_csv(category_csv)

# Remove duplicates in metadata if needed
category_df = category_df.drop_duplicates(subset=["original_filename"])

# Map from filename to category
category_map = dict(zip(category_df["original_filename"], category_df["class"]))

# Identify text embedding columns
dim_columns = [col for col in df.columns if col.startswith("dim_")]


# ------------------------------
# Helper function to search image features
# ------------------------------
def find_image_feature_path(base_folder, feature_name):
    """
    Recursively search for a .npz file matching feature_name
    """
    for root, dirs, files in os.walk(base_folder):
        if f"{feature_name}.npz" in files:
            return os.path.join(root, f"{feature_name}.npz")
    return None


# ------------------------------
# Process each post
# ------------------------------
for idx, row in df.iterrows():
    post_id = row["info_file"]
    print(f"\nProcessing post: {post_id}")

    # Text embedding
    text_embedding = row[dim_columns].values.reshape(1, -1)

    # Clean post_id: remove .info and replace _caption if present
    clean_post_id = post_id.replace(".info", "").replace("_caption", "")
    modified_post_id = clean_post_id + "_features"  # matches your .npz files

    # Find image feature path
    image_feature_path = find_image_feature_path(image_base_folder, modified_post_id)
    if not image_feature_path or not os.path.exists(image_feature_path):
        print(f"Image feature file for {modified_post_id} not found. Skipping...")
        continue

    # Load image feature
    image_data = np.load(image_feature_path)
    image_feature = image_data["features"].reshape(1, -1)

    # Combine image + text and apply ReLU
    combined_features = np.concatenate([image_feature, text_embedding], axis=1)
    combined_features_relu = np.maximum(0, combined_features)

    # Determine category
    category = category_map.get(clean_post_id, "unknown")

    # Create category folder
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    # Save combined features
    output_path = os.path.join(output_dir, f"{clean_post_id}.npz")
    np.savez_compressed(
        output_path, features=combined_features_relu.flatten(), post_id=post_id
    )

    print(f"Saved combined features to {output_path}")

print("\n✅ All features saved by category.")

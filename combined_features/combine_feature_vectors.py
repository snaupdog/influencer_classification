import numpy as np
import pandas as pd
import os

# Input files
csv_file = "../text_embed/processed_posts.csv"
image_base_folder = "../image_embed/image_features"
category_csv = "../image_embed/image_features/feature_metadata.csv"  # replace with your actual filename
output_base = "combined_feature_vectors"

# Load CSVs
df = pd.read_csv(csv_file)
category_df = pd.read_csv(category_csv)

# Clean duplicates if needed
category_df = category_df.drop_duplicates(subset=["original_filename"])

# Create a map from original_filename to class (category)
category_map = dict(zip(category_df["original_filename"], category_df["class"]))

# Get all dimension columns (text embedding)
dim_columns = [col for col in df.columns if col.startswith("dim_")]


# Function to search recursively for the .npz file
def find_image_feature_path(base_folder, post_id):
    for root, dirs, files in os.walk(base_folder):
        if f"{post_id}.npz" in files:
            return os.path.join(root, f"{post_id}.npz")
    return None


# Iterate over each row in the text vector CSV
for idx, row in df.iterrows():
    post_id = row["info_file"]
    print(f"\nProcessing post: {post_id}")

    text_embedding = row[dim_columns].values.reshape(1, -1)

    # Update post_id to match feature filename format
    modified_post_id = post_id.replace("_caption", "_features")
    image_feature_path = find_image_feature_path(image_base_folder, modified_post_id)

    if not image_feature_path or not os.path.exists(image_feature_path):
        print(f"Image feature file for {modified_post_id} not found. Skipping...")
        continue

    # Load image features
    image_data = np.load(image_feature_path)
    image_feature = image_data["features"].reshape(1, -1)

    # Combine and apply ReLU
    combined_features = np.concatenate([image_feature, text_embedding], axis=1)
    combined_features_relu = np.maximum(0, combined_features)

    # Get clean filename (without _caption)
    clean_post_id = post_id.replace("_caption", "")

    # Get class/category for this post
    category = category_map.get(clean_post_id, "unknown")  # fallback to 'unknown'

    # Create category folder if it doesn't exist
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    # Save the .npz file
    output_path = os.path.join(output_dir, f"{clean_post_id}.npz")
    np.savez_compressed(
        output_path, features=combined_features_relu.flatten(), post_id=post_id
    )

    print(f"Saved combined features to {output_path}")

print("\n✅ All features saved by category.")

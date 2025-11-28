# %%
import os
import numpy as np
import pandas as pd

# Configuration
FEATURE_DIR = "image_featuresv2"  # Change if different
# FEATURE_DIR = "image_features"  # Change if different
CLASS_NAMES = [
    "beauty",
    "family",
]

# Store summary info
summary = []


# Helper function to load npz safely
def load_npz_file(file_path):
    try:
        data = np.load(file_path)
        if len(data.files) == 0:
            return None
        array = data[data.files[0]]
        return array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Iterate over all classes and files
for class_name in CLASS_NAMES:
    class_dir = os.path.join(FEATURE_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"Class directory not found: {class_dir}")
        continue

    for file_name in os.listdir(class_dir):
        if not file_name.endswith("_features.npz"):
            continue

        file_path = os.path.join(class_dir, file_name)
        feature_array = load_npz_file(file_path)

        if feature_array is None:
            summary.append(
                {
                    "file": file_path,
                    "status": "failed_to_load",
                    "shape": None,
                    "nan_count": None,
                    "inf_count": None,
                    "mean": None,
                    "min": None,
                    "max": None,
                    "std": None,
                }
            )
            continue

        # Check for NaNs/Infs
        nan_count = np.isnan(feature_array).sum()
        inf_count = np.isinf(feature_array).sum()

        # Compute stats
        mean_val = np.mean(feature_array)
        min_val = np.min(feature_array)
        max_val = np.max(feature_array)
        std_val = np.std(feature_array)

        status = "ok"
        if nan_count > 0 or inf_count > 0:
            status = "nan_or_inf_detected"
        elif feature_array.size == 0:
            status = "empty_array"

        summary.append(
            {
                "file": file_path,
                "status": status,
                "shape": feature_array.shape,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
                "std": std_val,
            }
        )

# Convert summary to DataFrame
summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(FEATURE_DIR, "feature_check_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Summary saved to {summary_csv}")

# Optional: quick stats
print(summary_df["status"].value_counts())
print(summary_df.head())

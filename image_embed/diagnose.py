import numpy as np
import os
from tqdm import tqdm

# ============================================================================
# DIAGNOSE: Check the actual preprocessed image data
# ============================================================================

INPUT_DIR = "compressedPreprocessedImages"
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


def diagnose_image_data(input_dir, class_names, sample_size=5):
    """Analyze preprocessed image data to find the issue"""

    print("=" * 80)
    print("DIAGNOSING PREPROCESSED IMAGE DATA")
    print("=" * 80)

    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"\n❌ {class_name}: Directory not found")
            continue

        npz_files = [f for f in os.listdir(class_dir) if f.endswith(".npz")]

        if not npz_files:
            print(f"\n❌ {class_name}: No npz files found")
            continue

        print(f"\n{'='*80}")
        print(f"CLASS: {class_name} ({len(npz_files)} files)")
        print(f"{'='*80}")

        # Sample a few files
        sample_files = np.random.choice(
            npz_files, size=min(sample_size, len(npz_files)), replace=False
        )

        class_stats = []

        for i, file_name in enumerate(sample_files):
            file_path = os.path.join(class_dir, file_name)

            try:
                data = np.load(file_path)
                image = data[data.files[0]]
                image = image.astype(np.float32)

                # Analyze this image
                img_min = np.nanmin(image)
                img_max = np.nanmax(image)
                img_mean = np.nanmean(image)
                img_std = np.nanstd(image)
                nan_count = np.isnan(image).sum()
                inf_count = np.isinf(image).sum()

                class_stats.append(
                    {"min": img_min, "max": img_max, "mean": img_mean, "std": img_std}
                )

                status = "✓" if (nan_count == 0 and inf_count == 0) else "❌"

                print(f"\n  {status} File {i+1}: {file_name}")
                print(f"     Shape: {image.shape}")
                print(f"     Range: [{img_min:.4f}, {img_max:.4f}]")
                print(f"     Mean: {img_mean:.4f}, Std: {img_std:.4f}")
                print(f"     NaN count: {nan_count}, Inf count: {inf_count}")

                # Check if values are in expected range (typically 0-1 or -1 to 1)
                if img_min < -100 or img_max > 100:
                    print(f"     ⚠️  WARNING: Values outside typical range [-100, 100]")

            except Exception as e:
                print(f"\n  ❌ File {i+1}: {file_name} - Error: {e}")

        # Print class summary
        if class_stats:
            print(f"\n  CLASS SUMMARY:")
            all_mins = [s["min"] for s in class_stats]
            all_maxs = [s["max"] for s in class_stats]
            all_means = [s["mean"] for s in class_stats]
            all_stds = [s["std"] for s in class_stats]

            print(f"     Overall Min: {np.min(all_mins):.4f}")
            print(f"     Overall Max: {np.max(all_maxs):.4f}")
            print(f"     Mean of Means: {np.mean(all_means):.4f}")
            print(f"     Mean of Stds: {np.mean(all_stds):.4f}")


# Run diagnosis
diagnose_image_data(INPUT_DIR, CLASS_NAMES, sample_size=5)


# ============================================================================
# SOLUTION: Process images with proper normalization
# ============================================================================

print("\n\n" + "=" * 80)
print("RECOMMENDED FIX: Normalize images before feature extraction")
print("=" * 80)


def extract_features_with_preprocessing(
    input_dir, output_dir, class_names, feature_extractor, batch_size=32
):
    """
    Extract features with aggressive preprocessing to handle any data issues
    """
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    def load_npz_file(file_path):
        """Load and preprocess image"""
        try:
            data = np.load(file_path)
            image = data[data.files[0]]
            image = image.astype(np.float32)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def normalize_image_batch(batch_images):
        """Apply multiple normalization strategies"""
        normalized_batch = []

        for img in batch_images:
            # Remove batch dimension temporarily
            img_squeezed = img.squeeze(0) if img.shape[0] == 1 else img

            # Step 1: Handle NaN/Inf in input
            img_squeezed = np.nan_to_num(img_squeezed, nan=0.0, posinf=0.0, neginf=0.0)

            # Step 2: Clip extreme values
            img_squeezed = np.clip(img_squeezed, -1000, 1000)

            # Step 3: Normalize to [-1, 1] range
            img_min = np.min(img_squeezed)
            img_max = np.max(img_squeezed)

            if img_max > img_min:
                img_squeezed = 2 * (img_squeezed - img_min) / (img_max - img_min) - 1

            # Step 4: Apply ImageNet normalization (EfficientNet standard)
            # Assuming input is RGB in [-1, 1], convert to [0, 255] first
            img_squeezed = ((img_squeezed + 1) / 2 * 255).astype(np.uint8)
            img_squeezed = img_squeezed.astype(np.float32)

            # Apply EfficientNet preprocessing
            img_squeezed = preprocess_input(img_squeezed)

            # Add batch dimension back
            img_squeezed = np.expand_dims(img_squeezed, axis=0)
            normalized_batch.append(img_squeezed)

        return normalized_batch

    def process_batch_safe(
        batch_images, batch_file_paths, batch_output_paths, feature_extractor
    ):
        """Process batch with preprocessing"""
        try:
            # Normalize the batch
            normalized_batch = normalize_image_batch(batch_images)
            stacked_batch = np.vstack(normalized_batch)

            # Extract features
            feature_vectors = feature_extractor.predict(stacked_batch, verbose=0)

            # Save results
            for i, (file_path, output_path) in enumerate(
                zip(batch_file_paths, batch_output_paths)
            ):
                features = feature_vectors[i]

                # Final safety check
                if np.isnan(features).any() or np.isinf(features).any():
                    print(
                        f"   Still NaN in {os.path.basename(file_path)} - replacing with zeros"
                    )
                    features = np.zeros_like(features)

                # Normalize output features
                feature_mean = np.mean(features)
                feature_std = np.std(features)
                if feature_std > 1e-7:
                    features = (features - feature_mean) / feature_std

                original_filename = os.path.basename(file_path)
                base_name = os.path.splitext(original_filename)[0]
                np.savez_compressed(output_path, features=features)

            return True
        except Exception as e:
            print(f"Error processing batch: {e}")
            return False

    total_processed = 0
    errors = 0

    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(class_dir):
            continue

        npz_files = [f for f in os.listdir(class_dir) if f.endswith(".npz")]

        if not npz_files:
            continue

        print(f"\nProcessing {len(npz_files)} files from class {class_name}")

        batch_images = []
        batch_file_paths = []
        batch_output_paths = []

        for file_name in tqdm(npz_files, desc=f"Class: {class_name}"):
            file_path = os.path.join(class_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_name = f"{base_name}_features.npz"
            output_path = os.path.join(output_class_dir, output_name)

            if os.path.exists(output_path):
                continue

            image = load_npz_file(file_path)
            if image is None:
                errors += 1
                continue

            batch_images.append(image)
            batch_file_paths.append(file_path)
            batch_output_paths.append(output_path)

            if len(batch_images) >= batch_size:
                process_batch_safe(
                    batch_images,
                    batch_file_paths,
                    batch_output_paths,
                    feature_extractor,
                )
                total_processed += len(batch_images)
                batch_images, batch_file_paths, batch_output_paths = [], [], []

        if batch_images:
            process_batch_safe(
                batch_images, batch_file_paths, batch_output_paths, feature_extractor
            )
            total_processed += len(batch_images)

    print(f"\nCompleted. Processed {total_processed} files with {errors} errors.")


# To use the fixed version:
# from tensorflow.keras.models import load_model
# feature_extractor = load_model("image_feature_extractor.keras")
# extract_features_with_preprocessing(
#     input_dir="compressedPreprocessedImages",
#     output_dir="image_featuresv2",
#     class_names=CLASS_NAMES,
#     feature_extractor=feature_extractor,
#     batch_size=32
# )

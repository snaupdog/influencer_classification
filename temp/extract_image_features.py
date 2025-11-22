# %%
# Import required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Configuration
BATCH_SIZE = 32
MODEL_PATH = '/kaggle/input/feature-extractor/image_feature_extractor.keras' 
INPUT_DIR = '/kaggle/input/image-classification/tfCompressedPreprocessedImages' 
OUTPUT_DIR = '/kaggle/working/image_features' 


CLASS_NAMES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'other', 'pet', 'travel']

os.makedirs(OUTPUT_DIR, exist_ok=True)
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

print(f"Output directories created in {OUTPUT_DIR}")

# %%
def load_feature_extractor(model_path):
    """Load the feature extractor model"""
    print(f"Loading feature extractor from {model_path}")
    feature_extractor = load_model(model_path)
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")
    return feature_extractor

# Load the model
feature_extractor = load_feature_extractor(MODEL_PATH)

# %%
def load_npz_file(file_path):
    """Load an image from a npz file and prepare it for the model"""
    try:
        data = np.load(file_path)
        image = data[data.files[0]]
        # Ensure proper data type and shape
        image = image.astype(np.float32)
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_batch(batch_images, batch_file_paths, batch_output_paths, feature_extractor):
    """Process a batch of images and save their feature vectors"""
    try:
        # Stack all images into a single batch
        stacked_batch = np.vstack(batch_images)
        
        # Extract features
        feature_vectors = feature_extractor.predict(stacked_batch, verbose=0)
        
        # Save each feature vector with original filename
        for i, (file_path, output_path) in enumerate(zip(batch_file_paths, batch_output_paths)):
            # Get original filename from the input path
            original_filename = os.path.basename(file_path)
            base_name = os.path.splitext(original_filename)[0]  # Remove .npz extension
            
            # Save with original filename
            np.savez_compressed(output_path, features=feature_vectors[i])
            
        return True
    except Exception as e:
        print(f"Error processing batch: {e}")
        return False

# %%
def extract_features(input_dir, output_dir, class_names, feature_extractor, batch_size=32):
    """
    Extract features from preprocessed images and save with original filenames
    
    Args:
        input_dir: Directory containing preprocessed images in class folders
        output_dir: Where to save the extracted features
        class_names: List of class names (folder names)
        feature_extractor: Model for feature extraction
        batch_size: Number of images to process at once
    """
    total_processed = 0
    errors = 0
    
    # Initialize results tracking
    results = {
        'class': [],
        'files_processed': [],
        'errors': []
    }
    
    # Process each class
    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue
        
        # Get all npz files in this class directory
        npz_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
        
        if not npz_files:
            print(f"No npz files found in {class_dir}. Skipping.")
            continue
        
        class_processed = 0
        class_errors = 0
        
        print(f"Processing {len(npz_files)} files from class {class_name}")
        
        # Process files in batches for efficiency
        batch_images = []
        batch_file_paths = []
        batch_output_paths = []
        
        for file_name in tqdm(npz_files, desc=f"Class: {class_name}"):
            file_path = os.path.join(class_dir, file_name)
            
            # Use original filename for the output (just add _features)
            base_name = os.path.splitext(file_name)[0]  # Remove .npz extension
            output_name = f"{base_name}_features.npz"
            output_path = os.path.join(output_class_dir, output_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                continue
                
            image = load_npz_file(file_path)
            if image is None:
                errors += 1
                class_errors += 1
                continue
                
            batch_images.append(image)
            batch_file_paths.append(file_path)
            batch_output_paths.append(output_path)
            
            # Process batch when it reaches the batch size
            if len(batch_images) >= batch_size:
                success = process_batch(batch_images, batch_file_paths, batch_output_paths, feature_extractor)
                if success:
                    class_processed += len(batch_images)
                    total_processed += len(batch_images)
                else:
                    class_errors += len(batch_images)
                    errors += len(batch_images)
                
                # Clear batch
                batch_images = []
                batch_file_paths = []
                batch_output_paths = []
        
        # Process any remaining images
        if batch_images:
            success = process_batch(batch_images, batch_file_paths, batch_output_paths, feature_extractor)
            if success:
                class_processed += len(batch_images)
                total_processed += len(batch_images)
            else:
                class_errors += len(batch_images)
                errors += len(batch_images)
        
        # Record results for this class
        results['class'].append(class_name)
        results['files_processed'].append(class_processed)
        results['errors'].append(class_errors)
        
        print(f"Class {class_name}: Processed {class_processed} files with {class_errors} errors")
    
    print(f"Completed feature extraction. Processed {total_processed} files with {errors} errors.")
    return pd.DataFrame(results)

# %%
# Run the feature extraction
results_df = extract_features(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    class_names=CLASS_NAMES,
    feature_extractor=feature_extractor,
    batch_size=BATCH_SIZE
)

# Display results
print("\nExtraction Summary:")
display(results_df)

# Calculate totals
total_processed = results_df['files_processed'].sum()
total_errors = results_df['errors'].sum()
print(f"Total files processed: {total_processed}")
print(f"Total errors: {total_errors}")
print(f"Success rate: {(total_processed - total_errors) / total_processed * 100:.2f}%")

# %%
# Visualize processing statistics
plt.figure(figsize=(12, 6))
plt.bar(results_df['class'], results_df['files_processed'], color='blue', alpha=0.7, label='Processed')
plt.bar(results_df['class'], results_df['errors'], color='red', alpha=0.7, label='Errors')
plt.title('Feature Extraction Results by Class')
plt.xlabel('Class')
plt.ylabel('Number of Files')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Pie chart of overall class distribution
plt.figure(figsize=(10, 10))
plt.pie(results_df['files_processed'], labels=results_df['class'], autopct='%1.1f%%')
plt.title('Distribution of Processed Images by Class')
plt.tight_layout()
plt.show()

# %%
# Check a sample of the extracted feature vectors
def check_feature_vectors():
    # Pick a random class
    random_class = np.random.choice(CLASS_NAMES)
    feature_dir = os.path.join(OUTPUT_DIR, random_class)
    
    # Get a random feature file
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.npz')]
    
    if not feature_files:
        print(f"No feature files found in {feature_dir}")
        return
    
    random_file = np.random.choice(feature_files)
    feature_path = os.path.join(feature_dir, random_file)
    
    # Load and analyze
    feature_data = np.load(feature_path)
    feature_vector = feature_data['features']
    
    print(f"Sample feature vector from class {random_class}:")
    print(f"File: {random_file}")
    print(f"Shape: {feature_vector.shape}")
    print(f"Mean: {feature_vector.mean():.6f}")
    print(f"Std: {feature_vector.std():.6f}")
    print(f"Min: {feature_vector.min():.6f}")
    print(f"Max: {feature_vector.max():.6f}")
    
    # Visualize feature vector
    plt.figure(figsize=(15, 5))
    plt.plot(feature_vector)
    plt.title(f'Feature Vector Visualization: {random_file}')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    # Histogram of values
    plt.figure(figsize=(10, 5))
    plt.hist(feature_vector, bins=50)
    plt.title(f'Feature Vector Distribution: {random_file}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Run the check
check_feature_vectors()

# %%
# Create a metadata file for easy integration with text features
def create_metadata():
    metadata = []
    
    for class_name in CLASS_NAMES:
        feature_dir = os.path.join(OUTPUT_DIR, class_name)
        
        if not os.path.exists(feature_dir):
            continue
            
        feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.npz')]
        
        for feature_file in feature_files:
            # Extract original filename
            original_name = feature_file.replace('_features.npz', '')
            
            metadata.append({
                'class': class_name,
                'original_filename': original_name,
                'feature_path': os.path.join(class_name, feature_file)
            })
    
    # Create DataFrame and save to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(OUTPUT_DIR, 'feature_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Metadata saved to {metadata_path}")
    print(f"Total feature vectors: {len(metadata_df)}")
    return metadata_df

# Create metadata
metadata_df = create_metadata()
display(metadata_df.head())



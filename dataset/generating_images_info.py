import os
import shutil
import pandas as pd

# Load CSV
df = pd.read_csv(
    "JSON-image_17.csv",
    header=None,
    names=["influencer_name", "JSON_PostMetadata", "Image_files"],
)

# Define source and destination directories
info_source_dir = "../actual_dataset/info/"
image_source_dir = "../actual_dataset/images/"
destination_dir = os.getcwd()

# Create subdirectories
info_dest = os.path.join(destination_dir, "info")
image_dest = os.path.join(destination_dir, "images")
os.makedirs(info_dest, exist_ok=True)
os.makedirs(image_dest, exist_ok=True)

# Stats
copied_info = 0
copied_images = 0
missing_files = []

# Process rows
for _, row in df.iterrows():

    influencer = row["influencer_name"]
    info_file = row["JSON_PostMetadata"]

    # ------- FIXED IMAGE LIST PARSER -------
    raw = str(row["Image_files"]).strip()

    if "," in raw:
        image_files = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        image_files = [raw]
    # ----------------------------------------

    # Copy info file
    info_filename = f"{influencer}-{info_file}"
    src_info = os.path.join(info_source_dir, info_filename)
    dst_info = os.path.join(info_dest, info_filename)

    if os.path.exists(src_info):
        shutil.copy2(src_info, dst_info)
        copied_info += 1
    else:
        missing_files.append(f"Missing INFO: {src_info}")

    # Copy all images
    for img in image_files:
        img_filename = f"{influencer}-{img}"
        src_img = os.path.join(image_source_dir, img_filename)
        dst_img = os.path.join(image_dest, img_filename)

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied_images += 1
        else:
            missing_files.append(f"Missing IMAGE: {src_img}")

# Summary report
print("\n" + "=" * 60)
print("File Organization Complete!")
print("=" * 60)
print(f"Info files copied: {copied_info}")
print(f"Image files copied: {copied_images}")
print(f"Total files copied: {copied_info + copied_images}")

if missing_files:
    print(f"\n⚠ Missing files ({len(missing_files)}):")
    for f in missing_files[:15]:
        print("  -", f)
    if len(missing_files) > 15:
        print(f"  ... and {len(missing_files) - 15} more")
else:
    print("\n✅ All files were copied successfully!")

print(f"\nFiles organized in: {os.path.abspath(destination_dir)}")

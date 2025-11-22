import os
import json
import re
import csv
from pathlib import Path

# -----------------------------
# Step 1: Cleaning Utilities
# -----------------------------

emoji_pattern = re.compile(
    "["
    + "\U0001f600-\U0001f64f"  # emoticons
    + "\U0001f300-\U0001f5ff"  # symbols & pictographs
    + "\U0001f680-\U0001f6ff"  # transport & map symbols
    + "\U0001f1e0-\U0001f1ff"  # flags
    + "\U0001f900-\U0001f9ff"  # supplemental symbols
    + "\U0001fa70-\U0001faff"  # extended symbols
    + "\u2600-\u26ff"  # miscellaneous symbols
    + "\u2700-\u27bf"  # dingbats
    + "]+",
    flags=re.UNICODE,
)


def remove_emojis(text):
    return emoji_pattern.sub(r"", text)


def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def clean_caption(text):
    text = remove_hashtags(text)
    text = remove_emojis(text)
    # Preserve all Unicode characters (Korean, Chinese, etc.)
    text = re.sub(r"[^\w\s\u0080-\uFFFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "."


def clean_comment(text):
    # Remove mentions and hashtags
    text = re.sub(r"[#@]\w+", "", text)
    text = remove_emojis(text)
    # Keep all language characters (avoid stripping Korean etc.)
    text = re.sub(r"[^\w\s\u0080-\uFFFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Step 2: Load Influencers.txt
# -----------------------------
def load_influencers(influencers_file):
    influencers = {}
    with open(influencers_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                name, category, followers, following, posts = parts[:5]
                influencers[name.lower()] = {
                    "category": category,
                    "followers": followers,
                    "following": following,
                    "posts": posts,
                }
    print(f"✅ Loaded {len(influencers)} influencers")
    return influencers


# -----------------------------
# Step 3: Process .info Files
# -----------------------------
def extract_post_data(info_dir, influencers, output_csv):
    fieldnames = [
        "info_file",
        "influencer",
        "category",
        "followers",
        "following",
        "posts",
        "caption",
        "cleaned_caption",
        "comments",
        "images",
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []

    for root, _, files in os.walk(info_dir):
        for filename in files:
            if not filename.endswith(".info"):
                continue

            info_path = os.path.join(root, filename)
            influencer_name = filename.split("-")[0].lower()  # get influencer name
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # caption
                captions = [
                    edge["node"]["text"]
                    for edge in data.get("edge_media_to_caption", {}).get("edges", [])
                ]
                caption = captions[0] if captions else ""
                cleaned_caption = clean_caption(caption)

                # comments
                comments_raw = [
                    edge["node"]["text"]
                    for edge in data.get("edge_media_to_parent_comment", {}).get(
                        "edges", []
                    )
                ]
                cleaned_comments = [clean_comment(c) for c in comments_raw if c.strip()]
                comments_joined = " | ".join(cleaned_comments)

                # influencer data
                influencer_data = influencers.get(influencer_name, None)
                if influencer_data:
                    category = influencer_data["category"]
                    followers = influencer_data["followers"]
                    following = influencer_data["following"]
                    posts = influencer_data["posts"]
                else:
                    category = followers = following = posts = "unknown"

                # image matching (same prefix)
                dir_path = os.path.dirname(info_path)
                prefix = filename.replace(".info", "")
                image_matches = [
                    f
                    for f in os.listdir(dir_path)
                    if f.startswith(prefix) and f.endswith((".jpg", ".png"))
                ]
                image_list = ", ".join(image_matches) if image_matches else "none"

                rows.append(
                    {
                        "info_file": filename,
                        "influencer": influencer_name,
                        "category": category,
                        "followers": followers,
                        "following": following,
                        "posts": posts,
                        "caption": caption,
                        "cleaned_caption": cleaned_caption,
                        "comments": comments_joined,
                        "images": image_list,
                    }
                )

                print(f"✅ Processed {filename}")

            except Exception as e:
                print(f"❌ Error with {filename}: {e}")

    # write CSV
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n🎉 Saved {len(rows)} posts to {output_csv}")


# -----------------------------
# Step 4: Main Runner
# -----------------------------
def run_pipeline(root_folder):
    info_dir = os.path.join(root_folder, "info")
    influencers_file = os.path.join(root_folder, "influencers_17.txt")
    output_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "processed_posts.csv"
    )

    print("\n🚀 Starting post CSV pipeline...\n")

    influencers = load_influencers(influencers_file)
    extract_post_data(info_dir, influencers, output_csv)

    print("\n✅ Pipeline finished successfully!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  # text_embed/
    dataset_root = os.path.abspath(
        os.path.join(script_dir, "..", "dataset")
    )  # one up → dataset/

    run_pipeline(dataset_root)

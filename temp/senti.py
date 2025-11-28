import os
import json
import re
import csv
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# TEXT CLEANING (emojis allowed)
# ---------------------------
emoji_pattern = re.compile(
    "[" +
    u"\U0001F600-\U0001F64F" +
    u"\U0001F300-\U0001F5FF" +
    u"\U0001F680-\U0001F6FF" +
    u"\U0001F1E0-\U0001F1FF" +
    u"\U0001F900-\U0001F9FF" +
    u"\U0001FA70-\U0001FAFF" +
    u"\u2600-\u26FF" +
    u"\u2700-\u27BF" +
    "]+", flags=re.UNICODE
)

def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)

def clean_caption(text):
    text = remove_hashtags(text)
    text = emoji_pattern.sub("", text)        # remove emojis only for caption
    text = re.sub(r"[^\w\s\u0080-\uFFFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "."

def clean_comment(text):
    text = re.sub(r'[@#]\w+', "", text)       # remove @ and hashtags
    # keep emojis
    text = re.sub(r"[^\w\s" + emoji_pattern.pattern + "\u0080-\uFFFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# GPU Model
# ---------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(device)
model.eval()

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def batch_scores(comments: List[str]) -> List[float]:
    scores = []
    with torch.no_grad():
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i:i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            probs = F.softmax(out.logits, dim=1)
            scores.extend(probs[:, 2].detach().cpu().tolist())
    return scores


# ---------------------------------------------------
# MAIN PROCESSING (posts + weighted influencer sentiment)
# ---------------------------------------------------
def extract_post_data(info_dir, selected_influencers, output_csv):
    fieldnames = [
        "info_file", "influencer", "caption", "cleaned_caption",
        "comments", "post_sentiment", "images"
    ]

    rows = []
    influencer_sentiments = {}  # {name: {"score": total, "weight": total_w}}

    for root, _, files in os.walk(info_dir):
        for filename in files:
            if not filename.endswith(".info"):
                continue

            influencer = filename.split("-")[0].lower()
            if influencer not in selected_influencers:
                continue

            path = os.path.join(root, filename)

            try:
                data = json.load(open(path, "r", encoding="utf-8"))

                # caption
                caps = [e["node"]["text"] for e in data.get("edge_media_to_caption", {}).get("edges", [])]
                caption = caps[0] if caps else ""
                cleaned_caption = clean_caption(caption)

                # comments
                raw_comments = [
                    e["node"]["text"]
                    for e in data.get("edge_media_to_parent_comment", {}).get("edges", [])
                ]
                cleaned_comments = [clean_comment(c) for c in raw_comments if c.strip()]

                if len(cleaned_comments) == 0:
                    print(f"⏭ Skipped {filename} (0 comments)")
                    continue

                comments_joined = " | ".join(cleaned_comments)

                # sentiment weighted by comment length
                scores = batch_scores(cleaned_comments)
                w_sum = 0
                s_sum = 0

                for c, s in zip(cleaned_comments, scores):
                    w = len(c)
                    w_sum += w
                    s_sum += s * w

                if w_sum == 0:
                    print(f"⏭ Skipped {filename} (empty after cleaning)")
                    continue

                sentiment = s_sum / w_sum

                # images
                prefix = filename.replace(".info", "")
                images = [
                    f for f in os.listdir(root)
                    if f.startswith(prefix) and f.endswith((".jpg", ".png"))
                ]
                images = ", ".join(images) if images else "none"

                # row for main CSV
                rows.append({
                    "info_file": filename,
                    "influencer": influencer,
                    "caption": caption,
                    "cleaned_caption": cleaned_caption,
                    "comments": comments_joined,
                    "post_sentiment": sentiment,
                    "images": images
                })

                # accumulate influencer weights
                if influencer not in influencer_sentiments:
                    influencer_sentiments[influencer] = {"score": 0.0, "weight": 0}

                influencer_sentiments[influencer]["score"] += sentiment * w_sum
                influencer_sentiments[influencer]["weight"] += w_sum

                print("✔ Processed:", filename)

            except Exception as e:
                print("❌ Error:", filename, e)

    # ---------------------
    # Write posts CSV
    # ---------------------
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved posts CSV: {output_csv}")

    # ---------------------
    # Write influencer.csv
    # ---------------------
    infl_csv = os.path.join(os.path.dirname(output_csv), "influencer_sentiments.csv")

    with open(infl_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["influencer", "sentiment_weighted"])

        for name, v in influencer_sentiments.items():
            if v["weight"] == 0:
                avg = 0.5
            else:
                avg = v["score"] / v["weight"]

            writer.writerow([name, round(avg, 4)])

    print("Saved influencer sentiment CSV:", infl_csv)
    print("\nTotal posts processed:", len(rows))


# ---------------------------------------------------
# RUNNER
# ---------------------------------------------------
def run_pipeline():
    info_dir = r"C:\Users\Adithya\Dev\newdataset\info\info"

    # detect influencers from filenames
    influencers = sorted(list({
        f.split("-")[0].lower()
        for f in os.listdir(info_dir)
        if f.endswith(".info")
    }))

    print(f"Found {len(influencers)} influencers.")
    n = int(input("How many influencers do you want to analyze? "))

    selected = influencers[:n]
    print("\nAnalyzing:")
    for s in selected:
        print(" →", s)

    output_csv = r"C:\Users\Adithya\Dev\newdataset\processed_posts.csv"
    extract_post_data(info_dir, selected, output_csv)


if __name__ == "__main__":
    run_pipeline()

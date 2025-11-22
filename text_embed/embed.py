import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ------------------------------
# Configuration
# ------------------------------
input_csv_path = os.path.join(os.getcwd(), "processed_posts.csv")
output_csv_path = input_csv_path  # overwrite same CSV

# ------------------------------
# Load multilingual BERT
# ------------------------------
print("🔹 Loading multilingual BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Using device: {device}")


# ------------------------------
# Embedding function
# ------------------------------
def get_bert_embedding(text):
    if not isinstance(text, str) or not text.strip():
        text = "."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding


# ------------------------------
# Load CSV
# ------------------------------
print("📂 Loading CSV...")
df = pd.read_csv(input_csv_path, encoding="utf-8")

if "cleaned_caption" not in df.columns:
    raise ValueError("❌ The CSV must contain a 'cleaned_caption' column.")

# ------------------------------
# Compute embeddings
# ------------------------------
print(f"🔍 Computing embeddings for {len(df)} cleaned captions...")

embeddings = []
for caption in tqdm(df["cleaned_caption"], desc="Embedding captions"):
    try:
        emb = get_bert_embedding(str(caption))
    except Exception as e:
        print(f"⚠️ Error embedding text: {e}")
        emb = [0.0] * 768
    embeddings.append(emb)

# ------------------------------
# Merge embeddings into DataFrame
# ------------------------------
embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(768)])
df = pd.concat([df, embeddings_df], axis=1)

# ------------------------------
# Save updated CSV
# ------------------------------
print(f"💾 Saving updated CSV with embeddings to: {output_csv_path}")
df.to_csv(output_csv_path, index=False, encoding="utf-8")

print("🏁 Done! Embeddings from cleaned captions have been added successfully.")

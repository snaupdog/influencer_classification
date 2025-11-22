import csv

# Read the influencers from influencers.csv
influencer_usernames = set()
with open("../dataset/influencers.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        influencer_usernames.add(row["Username"].strip())

print(f"Found {len(influencer_usernames)} influencers in influencers.csv")

# Filter JSON-mapping.csv based on influencer usernames
filtered_rows = []
with open("JSON-image_17.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        influencer_name = row["influencer_name"].strip()
        if influencer_name in influencer_usernames:
            filtered_rows.append(row)

print(f"Found {len(filtered_rows)} matching rows in JSON-mapping.csv")

# Write the filtered data to a new file
with open("../dataset/JSON-image_17.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print(f"Created JSON-image_17.csv with {len(filtered_rows)} rows")

# Show statistics by influencer
influencer_counts = {}
for row in filtered_rows:
    name = row["influencer_name"]
    influencer_counts[name] = influencer_counts.get(name, 0) + 1

print("\nPosts per influencer:")
for name, count in sorted(influencer_counts.items()):
    print(f"  {name}: {count} posts")

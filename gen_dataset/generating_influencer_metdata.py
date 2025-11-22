import csv
from collections import defaultdict


def generate_influencers_csv(input_file, output_file, n_per_category):
    """Generate influencers.csv with N influencers per category"""
    categories = defaultdict(list)

    # Read the CSV file
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header line
        for row in reader:
            if len(row) != 5:
                print(f"[WARN] malformed row skipped: {row}")
                continue
            username, category, followers, followees, posts = row
            categories[category].append(row)

    # Select N influencers per category
    final_rows = []
    for category, items in categories.items():
        final_rows.extend(items[:n_per_category])  # take first N

    # Write CSV output
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "Category", "Followers", "Followees", "Posts"])
        writer.writerows(final_rows)

    print(f"[OK] Created {output_file} with {n_per_category} influencers per category.")
    print(f"Categories found: {list(categories.keys())}")

    return [row[0] for row in final_rows]  # Return list of usernames


def filter_json_mapping(influencer_usernames, json_input_file, json_output_file):
    """Filter JSON-image mapping based on influencer usernames"""
    print(f"\n[INFO] Found {len(influencer_usernames)} influencers to filter")

    # Filter JSON-mapping.csv based on influencer usernames
    filtered_rows = []
    with open(json_input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            influencer_name = row["influencer_name"].strip()
            if influencer_name in influencer_usernames:
                filtered_rows.append(row)

    print(f"[OK] Found {len(filtered_rows)} matching rows in {json_input_file}")

    # Write the filtered data to a new file
    with open(json_output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[OK] Created {json_output_file} with {len(filtered_rows)} rows")

    # Show statistics by influencer
    influencer_counts = {}
    for row in filtered_rows:
        name = row["influencer_name"]
        influencer_counts[name] = influencer_counts.get(name, 0) + 1

    print("\nPosts per influencer:")
    for name, count in sorted(influencer_counts.items()):
        print(f"  {name}: {count} posts")


if __name__ == "__main__":
    # Prompt user for input
    try:
        n = int(input("Enter number of influencers per category: "))
        if n <= 0:
            print("[ERROR] Please enter a positive number.")
        else:
            # Step 1: Generate influencers.csv
            influencer_usernames = generate_influencers_csv(
                input_file="influencers_17.csv",
                output_file="../dataset/influencers.csv",
                n_per_category=n,
            )

            # Step 2: Filter JSON-image mapping
            filter_json_mapping(
                influencer_usernames=set(influencer_usernames),
                json_input_file="JSON-image_17.csv",
                json_output_file="../dataset/JSON-image_17.csv",
            )

            print("\n[SUCCESS] All operations completed!")

    except ValueError:
        print("[ERROR] Invalid input. Please enter a valid number.")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

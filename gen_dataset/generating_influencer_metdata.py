import csv
from collections import defaultdict


def generate_influencers_csv(input_file, output_file, n_influencers_per_category):
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
        final_rows.extend(items[:n_influencers_per_category])  # take first N

    # Write CSV output
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "Category", "Followers", "Followees", "Posts"])
        writer.writerows(final_rows)

    print(
        f"[OK] Created {output_file} with {n_influencers_per_category} influencers per category."
    )
    print(f"Categories found: {list(categories.keys())}")

    return [row[0] for row in final_rows]  # Return list of usernames


def filter_json_mapping(
    influencer_usernames, json_input_file, json_output_file, n_posts_per_influencer
):
    """Filter JSON-image mapping based on influencer usernames and limit posts per influencer"""
    print(f"\n[INFO] Found {len(influencer_usernames)} influencers to filter")

    filtered_rows = []
    per_influencer_count = defaultdict(int)

    with open(json_input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            name = row["influencer_name"].strip()

            # Only include influencers selected earlier
            if name in influencer_usernames:

                # Limit to N posts per influencer
                if per_influencer_count[name] < n_posts_per_influencer:
                    filtered_rows.append(row)
                    per_influencer_count[name] += 1

    # Write output
    with open(json_output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[OK] Created {json_output_file} with {len(filtered_rows)} rows")

    print("\nPosts per influencer:")
    for name, count in sorted(per_influencer_count.items()):
        print(f"  {name}: {count} posts")


if __name__ == "__main__":
    try:
        n_influencers_per_category = int(
            input("Enter number of influencers per category: ")
        )
        n_posts_per_influencer = int(input("Enter number of posts per influencer: "))

        if n_influencers_per_category <= 0 or n_posts_per_influencer <= 0:
            print("[ERROR] Please enter positive numbers.")
        else:
            # Step 1: pick influencers
            influencer_usernames = generate_influencers_csv(
                input_file="influencers_17.csv",
                output_file="../dataset/influencers.csv",
                n_influencers_per_category=n_influencers_per_category,
            )

            # Step 2: limit posts per influencer
            filter_json_mapping(
                influencer_usernames=set(influencer_usernames),
                json_input_file="JSON-image_17.csv",
                json_output_file="../dataset/JSON-image_17.csv",
                n_posts_per_influencer=n_posts_per_influencer,
            )

            print("\n[SUCCESS] All operations completed!")

    except ValueError:
        print("[ERROR] Invalid input. Please enter valid numbers.")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

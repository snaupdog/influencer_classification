import csv
from collections import defaultdict


def generate_influencers_csv(input_file, output_file, n_per_category):
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


if __name__ == "__main__":
    # Prompt user for input
    try:
        n = int(input("Enter number of influencers per category: "))
        if n <= 0:
            print("[ERROR] Please enter a positive number.")
        else:
            generate_influencers_csv(
                input_file="influencers_17.csv",
                output_file="../dataset/influencers.csv",
                n_per_category=n,
            )
    except ValueError:
        print("[ERROR] Invalid input. Please enter a valid number.")

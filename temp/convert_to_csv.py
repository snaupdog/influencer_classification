import csv
import ast


def convert_influencers_txt(input_file, output_file):
    rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            rows.append(parts)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[OK] Converted {input_file} → {output_file}")


def convert_json_image_txt(input_file, output_file):
    rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                print(f"[WARN] malformed line skipped: {line}")
                continue

            influencer, metadata_file, image_list_raw = parts

            # Convert "['file.jpg']" → python list ['file.jpg']
            try:
                image_list = ast.literal_eval(image_list_raw)
            except:
                image_list = [image_list_raw]  # fallback

            # Join multiple images as comma-separated string
            image_list_str = ", ".join(image_list)

            rows.append([influencer, metadata_file, image_list_str])

    # Write csv
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["influencer_name", "JSON_PostMetadata", "Image_files"])
        writer.writerows(rows)

    print(f"[OK] Converted {input_file} → {output_file}")


if __name__ == "__main__":
    # Convert influencers
    convert_influencers_txt(
        input_file="influencers_17.txt", output_file="influencers_17.csv"
    )

    # Convert JSON-image file
    convert_json_image_txt(
        input_file="JSON-image_17.txt", output_file="JSON-image_17.csv"
    )

import os
import json
import math
import csv
import matplotlib.pyplot as plt
import shutil

INPUT_DIR = "JSON_Complex"
IMG_INPUT_DIR = "img_complex"
TOP_K = 500

OUTPUT_ALL = "all_tables_metrics.csv"
OUTPUT_TOP = "top500_tables_metrics.csv"

OUTPUT_JSON_DIR = "JSON_Complex_TOP500"
OUTPUT_IMG_DIR = "IMG_Complex_TOP500"

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def flatten_keys(obj, prefix="", depth=1):
    keys = []
    max_depth = depth

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            sub_keys, sub_depth = flatten_keys(v, new_prefix, depth + 1)
            keys.extend(sub_keys)
            max_depth = max(max_depth, sub_depth)

    elif isinstance(obj, list):
        for item in obj[:10]:
            sub_keys, sub_depth = flatten_keys(item, prefix, depth + 1)
            keys.extend(sub_keys)
            max_depth = max(max_depth, sub_depth)

    else:
        keys.append(prefix)

    return keys, max_depth


def analyze_table(data):
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return None

    rows = data
    num_rows = len(rows)

    all_keys = []
    row_key_sets = []

    max_depth = 1

    for row in rows[:50]:
        keys, depth = flatten_keys(row)
        max_depth = max(max_depth, depth)
        all_keys.extend(keys)
        row_key_sets.append(set(keys))

    num_columns = len(set(all_keys))

    unique_counts = [len(k) for k in row_key_sets]
    schema_variance = sum(unique_counts) / (len(unique_counts) + 1e-5)

    score = (
        0.4 * math.log(num_columns + 1) +
        0.3 * max_depth +
        0.2 * math.log(num_rows + 1) +
        0.1 * schema_variance
    )

    return {
        "rows": num_rows,
        "columns": num_columns,
        "depth": max_depth,
        "variance": schema_variance,
        "score": score
    }


results = []

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(INPUT_DIR, fname)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = analyze_table(data)

        if stats:
            stats["file"] = fname
            results.append(stats)

    except Exception:
        continue


results.sort(key=lambda x: x["score"], reverse=True)


with open(OUTPUT_ALL, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "rows", "columns", "depth", "variance", "score"]
    )
    writer.writeheader()
    writer.writerows(results)


top500 = results[:TOP_K]

with open(OUTPUT_TOP, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "rows", "columns", "depth", "variance", "score"]
    )
    writer.writeheader()
    writer.writerows(top500)


scores = [x["score"] for x in results]

plt.figure()
plt.hist(scores, bins=30)
plt.title("Distribution of Table Complexity Scores")
plt.xlabel("Complexity Score")
plt.ylabel("Number of Tables")
plt.savefig("complexity_distribution.png")
plt.show()


copied_json = 0
copied_img = 0
missing_images = []

for item in top500:
    fname = item["file"]

    src_json = os.path.join(INPUT_DIR, fname)
    dst_json = os.path.join(OUTPUT_JSON_DIR, fname)

    try:
        shutil.copy2(src_json, dst_json)
        copied_json += 1
    except Exception:
        continue

    base_name = fname.replace(".json", "")
    if base_name.endswith("_complex"):
        base_name = base_name.replace("_complex", "")

    img_name = base_name + "_complex.jpg"

    src_img = os.path.join(IMG_INPUT_DIR, img_name)
    dst_img = os.path.join(OUTPUT_IMG_DIR, img_name)

    if os.path.exists(src_img):
        try:
            shutil.copy2(src_img, dst_img)
            copied_img += 1
        except Exception:
            pass
    else:
        missing_images.append(img_name)


print(f"Saved {len(results)} tables total")
print(f"Saved top {TOP_K} tables")
print(f"Copied JSON: {copied_json}")
print(f"Copied IMG: {copied_img}")
print(f"Missing images: {len(missing_images)}")


if missing_images:
    with open("missing_images.txt", "w") as f:
        for name in missing_images:
            f.write(name + "\n")
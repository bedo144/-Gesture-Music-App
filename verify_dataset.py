import csv
import os
from collections import Counter


DATASET_FILE = "data/handnote_dataset.csv"


# ==========================================
# Check file
# ==========================================

if not os.path.exists(DATASET_FILE):
    print("ERROR: Dataset file not found.")
    exit()


# ==========================================
# Read dataset
# ==========================================

with open(DATASET_FILE, "r", newline="") as file:

    reader = csv.DictReader(file)
    rows = list(reader)


# ==========================================
# Basic information
# ==========================================

print("=" * 50)
print("HandNote AI Dataset Verification")
print("=" * 50)

print(f"Total samples: {len(rows)}")


if len(rows) == 0:
    print("ERROR: Dataset is empty.")
    exit()


# ==========================================
# Count classes
# ==========================================

hand_counts = Counter()
label_counts = Counter()
combination_counts = Counter()


for row in rows:

    hand = row["hand_side"]
    label = row["label"]

    hand_counts[hand] += 1
    label_counts[label] += 1

    combination_counts[
        f"{hand} -> {label}"
    ] += 1


# ==========================================
# Feature check
# ==========================================

feature_columns = [
    column
    for column in reader.fieldnames
    if column.startswith("feature_")
]


missing_values = 0

for row in rows:

    for feature in feature_columns:

        if row[feature] == "":
            missing_values += 1


# ==========================================
# Print results
# ==========================================

print("\nFeatures")
print("-" * 30)

print(f"Features/sample: {len(feature_columns)}")
print(f"Missing values : {missing_values}")


print("\nHand Distribution")
print("-" * 30)

for hand, count in hand_counts.items():

    print(f"{hand}: {count}")


print("\nClass Distribution")
print("-" * 30)

for label, count in label_counts.items():

    print(f"{label}: {count}")


print("\nHand + Class")
print("-" * 30)

for combination, count in combination_counts.items():

    print(f"{combination}: {count}")


# ==========================================
# Final validation
# ==========================================

if (
    len(feature_columns) == 63
    and missing_values == 0
):

    print("\n✅ DATASET STRUCTURE IS CORRECT")

else:

    print("\n❌ DATASET HAS A PROBLEM")


print("=" * 50)
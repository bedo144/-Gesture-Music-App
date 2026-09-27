import pandas as pd
import numpy as np


# ==========================================
# Configuration
# ==========================================

DATASET_FILE = "data/handnote_dataset.csv"


# ==========================================
# Load dataset
# ==========================================

print("=" * 60)
print("HandNote AI - Dataset Analysis")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)


# ==========================================
# Basic information
# ==========================================

print("\n1. DATASET INFORMATION")
print("-" * 40)

print(f"Total samples: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ==========================================
# Check hand distribution
# ==========================================

print("\n2. HAND DISTRIBUTION")
print("-" * 40)

print(df["hand_side"].value_counts())


# ==========================================
# Check class distribution
# ==========================================

print("\n3. CLASS DISTRIBUTION")
print("-" * 40)

print(df["label"].value_counts())


# ==========================================
# Hand + Class distribution
# ==========================================

print("\n4. HAND + CLASS DISTRIBUTION")
print("-" * 40)

print(
    df.groupby(
        ["hand_side", "label"]
    ).size()
)


# ==========================================
# Feature information
# ==========================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]

print("\n5. FEATURE INFORMATION")
print("-" * 40)

print(f"Number of features: {len(feature_columns)}")


# ==========================================
# Missing values
# ==========================================

print("\n6. MISSING VALUES")
print("-" * 40)

missing_values = df[feature_columns].isnull().sum().sum()

print(f"Missing feature values: {missing_values}")


# ==========================================
# Duplicate samples
# ==========================================

print("\n7. DUPLICATE SAMPLES")
print("-" * 40)

duplicates = df.duplicated(
    subset=feature_columns
).sum()

print(f"Duplicate feature rows: {duplicates}")


# ==========================================
# Feature statistics
# ==========================================

print("\n8. FEATURE RANGE")
print("-" * 40)

features = df[feature_columns].values

print(
    f"Minimum value: {features.min():.4f}"
)

print(
    f"Maximum value: {features.max():.4f}"
)

print(
    f"Mean value: {features.mean():.4f}"
)

print(
    f"Standard deviation: {features.std():.4f}"
)


# ==========================================
# Final check
# ==========================================

print("\n9. FINAL CHECK")
print("-" * 40)

if (
    len(feature_columns) == 63
    and missing_values == 0
):

    print("✅ Dataset has 63 features per sample")
    print("✅ No missing feature values")
    print("✅ Dataset analysis completed")

else:

    print("❌ Dataset needs attention")


print("\n" + "=" * 60)
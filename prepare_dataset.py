import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# Configuration
# ==========================================

DATASET_FILE = "data/handnote_dataset.csv"


# ==========================================
# Load dataset
# ==========================================

print("=" * 60)
print("HandNote AI - Dataset Preparation")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)


# ==========================================
# Feature columns
# ==========================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]


print("\nTotal samples:", len(df))
print("Features:", len(feature_columns))


# ==========================================
# Function to prepare one hand
# ==========================================

def prepare_hand_data(data, hand_name):

    print("\n" + "=" * 60)
    print(f"Preparing {hand_name} Hand Dataset")
    print("=" * 60)

    # --------------------------------------
    # Features
    # --------------------------------------

    X = data[feature_columns]

    # --------------------------------------
    # Labels
    # --------------------------------------

    y = data["label"]


    # ======================================
    # First split
    # Train = 70%
    # Temporary = 30%
    # ======================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


    # ======================================
    # Second split
    #
    # Validation = 15%
    # Test = 15%
    # ======================================

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )


    # ======================================
    # Print information
    # ======================================

    print("\nSamples:")
    print("Total      :", len(data))
    print("Training   :", len(X_train))
    print("Validation :", len(X_val))
    print("Testing    :", len(X_test))


    print("\nClass distribution:")

    print("\nTraining:")
    print(y_train.value_counts().sort_index())

    print("\nValidation:")
    print(y_val.value_counts().sort_index())

    print("\nTesting:")
    print(y_test.value_counts().sort_index())


    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# LEFT HAND
# ==========================================

left_data = df[
    df["hand_side"] == "Left"
].copy()


left_data = left_data.reset_index(drop=True)


left_results = prepare_hand_data(
    left_data,
    "Left"
)


# ==========================================
# RIGHT HAND
# ==========================================

right_data = df[
    df["hand_side"] == "Right"
].copy()


right_data = right_data.reset_index(drop=True)


right_results = prepare_hand_data(
    right_data,
    "Right"
)


# ==========================================
# Final message
# ==========================================

print("\n" + "=" * 60)
print("✅ DATASET PREPARATION COMPLETED")
print("=" * 60)
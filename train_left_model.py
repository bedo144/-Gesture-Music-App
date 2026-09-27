import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ==========================================
# Configuration
# ==========================================

DATASET_FILE = "data/handnote_dataset.csv"

MODEL_FILE = "left_hand_model.keras"


# ==========================================
# Load dataset
# ==========================================

print("=" * 60)
print("HandNote AI - Left Hand Model Training")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)


# ==========================================
# Select Left Hand only
# ==========================================

left_data = df[
    df["hand_side"] == "Left"
].copy()

left_data = left_data.reset_index(drop=True)


print("\nTotal Left Hand samples:")
print(len(left_data))


# ==========================================
# Get features
# ==========================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]


X = left_data[feature_columns].values


# ==========================================
# Get labels
# ==========================================

y_text = left_data["label"].values


# ==========================================
# Encode labels
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)


print("\nClasses:")
print(label_encoder.classes_)


# ==========================================
# Train / Validation / Test split
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ==========================================
# Build Neural Network
# ==========================================

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(63,)
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        7,
        activation="softmax"
    )
])


# ==========================================
# Compile model
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# Show model
# ==========================================

print("\nModel Architecture:")
model.summary()


# ==========================================
# Train
# ==========================================

print("\nStarting training...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=16,

    verbose=1
)


# ==========================================
# Evaluate
# ==========================================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ==========================================
# Save model
# ==========================================

model.save(MODEL_FILE)


print("\nModel saved to:")

print(MODEL_FILE)


print("\n" + "=" * 60)
print("✅ LEFT HAND MODEL TRAINING COMPLETED")
print("=" * 60)
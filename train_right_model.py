import os

# ============================================================
# REDUCE TENSORFLOW LOG MESSAGES
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "data/handnote_dataset.csv"
MODEL_FILE = "right_hand_model.keras"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("HandNote AI - Right Hand Model Training")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Total dataset samples: {len(df)}")


# ============================================================
# 2. SELECT RIGHT HAND DATA
# ============================================================

right_data = df[
    df["hand_side"] == "Right"
].copy()

right_data = right_data.reset_index(drop=True)

print(
    f"Total Right Hand samples: {len(right_data)}"
)


# ============================================================
# 3. GET FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]

print(
    f"Number of features: {len(feature_columns)}"
)


# ============================================================
# 4. CREATE INPUT FEATURES
# ============================================================

X = right_data[
    feature_columns
].values.astype(np.float32)


# ============================================================
# 5. CREATE LABELS
# ============================================================

y_text = right_data[
    "label"
].values


# ============================================================
# 6. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

print("\nClasses:")

for index, class_name in enumerate(
    label_encoder.classes_
):
    print(
        f"{index}: {class_name}"
    )


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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


print("\nDataset Split")
print("-" * 40)

print(
    f"Training   : {len(X_train)}"
)

print(
    f"Validation : {len(X_val)}"
)

print(
    f"Testing    : {len(X_test)}"
)


# ============================================================
# 8. BUILD NEURAL NETWORK
# ============================================================

print("\nBuilding neural network...")


model = Sequential([

    Input(shape=(63,)),

    Dense(
        128,
        activation="relu"
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
        5,
        activation="softmax"
    )
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 10. SHOW MODEL
# ============================================================

print("\nModel Architecture")

print("-" * 40)

model.summary()


# ============================================================
# 11. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(

    MODEL_FILE,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting training...")

print("-" * 60)


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=16,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test,

    verbose=0
)


print("\n" + "=" * 60)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# 15. TRAINING SUMMARY
# ============================================================

print("\nTraining Summary")

print("-" * 40)

print(
    f"Epochs completed: {len(history.history['loss'])}"
)

print(
    f"Best validation accuracy: "
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 16. FINISH
# ============================================================

print("\n" + "=" * 60)

print(
    "🔥 RIGHT HAND MODEL TRAINING COMPLETED"
)

print("=" * 60)

print(
    f"Model saved as: {MODEL_FILE}"
)

print("=" * 60)
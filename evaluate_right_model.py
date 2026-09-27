import os

# Reduce TensorFlow log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "data/handnote_dataset.csv"
MODEL_FILE = "right_hand_model.keras"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("HandNote AI - Right Hand Model Evaluation")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print(
    f"Total dataset samples: {len(df)}"
)


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
# 3. GET FEATURES
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
# 4. CREATE X
# ============================================================

X = right_data[
    feature_columns
].values.astype(np.float32)


# ============================================================
# 5. CREATE Y
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
# 7. SAME DATA SPLIT USED DURING TRAINING
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
# 8. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print(
    "✅ Model loaded successfully"
)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

probabilities = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 10. GET PREDICTED CLASSES
# ============================================================

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)

print(
    f"TEST ACCURACY: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")

print("-" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        digits=4
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix")

print("-" * 60)

matrix = confusion_matrix(
    y_test,
    predictions
)

confusion_df = pd.DataFrame(
    matrix,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

print(
    confusion_df
)


# ============================================================
# 14. INDIVIDUAL PREDICTIONS
# ============================================================

print("\nIndividual Test Predictions")

print("-" * 60)

for i in range(len(X_test)):

    actual_class = label_encoder.inverse_transform(
        [y_test[i]]
    )[0]

    predicted_class = label_encoder.inverse_transform(
        [predictions[i]]
    )[0]

    confidence = probabilities[i][
        predictions[i]
    ]

    if actual_class == predicted_class:
        status = "✅"
    else:
        status = "❌"

    print(
        f"{i + 1:02d}. "
        f"Actual: {actual_class:<7} | "
        f"Predicted: {predicted_class:<7} | "
        f"Confidence: {confidence * 100:6.2f}% "
        f"{status}"
    )


# ============================================================
# 15. PREDICTION SUMMARY
# ============================================================

correct = np.sum(
    y_test == predictions
)

incorrect = np.sum(
    y_test != predictions
)


print("\nPrediction Summary")

print("-" * 60)

print(
    f"Correct predictions  : {correct}"
)

print(
    f"Incorrect predictions: {incorrect}"
)

print(
    f"Total test samples   : {len(y_test)}"
)


# ============================================================
# 16. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

if accuracy >= 0.90:

    print(
        "🔥 EXCELLENT: Right Hand model accuracy is 90%+."
    )

elif accuracy >= 0.80:

    print(
        "✅ GOOD: Right Hand model is learning well."
    )

else:

    print(
        "⚠️ MODEL NEEDS IMPROVEMENT."
    )


print("=" * 60)

print(
    "✅ RIGHT HAND MODEL EVALUATION COMPLETED"
)

print("=" * 60)
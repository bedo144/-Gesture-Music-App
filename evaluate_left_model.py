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
MODEL_FILE = "left_hand_model.keras"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("HandNote AI - Left Hand Model Evaluation")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Total dataset samples: {len(df)}")


# ============================================================
# 2. SELECT LEFT HAND DATA
# ============================================================

left_data = df[
    df["hand_side"] == "Left"
].copy()

left_data = left_data.reset_index(drop=True)

print(f"Total Left Hand samples: {len(left_data)}")


# ============================================================
# 3. GET FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]

print(f"Number of features: {len(feature_columns)}")


# ============================================================
# 4. CREATE X
# ============================================================

X = left_data[
    feature_columns
].values.astype(np.float32)


# ============================================================
# 5. CREATE Y
# ============================================================

y_text = left_data[
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

print(
    label_encoder.classes_
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
# 8. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("✅ Model loaded successfully")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions on test data...")

probabilities = model.predict(

    X_test,

    verbose=0
)


# ============================================================
# 10. GET PREDICTED CLASS
# ============================================================

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 11. CALCULATE ACCURACY
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

report = classification_report(

    y_test,

    predictions,

    target_names=label_encoder.classes_,

    digits=4
)

print(report)


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


print(confusion_df)


# ============================================================
# 14. SHOW INDIVIDUAL TEST PREDICTIONS
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

    status = "✅" if actual_class == predicted_class else "❌"

    print(
        f"{i + 1:02d}. "
        f"Actual: {actual_class:<2} | "
        f"Predicted: {predicted_class:<2} | "
        f"Confidence: {confidence * 100:6.2f}% "
        f"{status}"
    )


# ============================================================
# 15. COUNT CORRECT / INCORRECT
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
        "🔥 EXCELLENT: Left Hand model accuracy is above 90%."
    )

elif accuracy >= 0.80:

    print(
        "✅ GOOD: Left Hand model is learning well."
    )

else:

    print(
        "⚠️ MODEL NEEDS IMPROVEMENT."
    )


print("=" * 60)

print(
    "✅ LEFT HAND MODEL EVALUATION COMPLETED"
)

print("=" * 60)
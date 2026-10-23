
# ============================================================
# Diabetes Prediction - Model Training Pipeline
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "diabetes_training_data.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "diabetes"

# 1. Load and clean dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)
print("Original dataset columns:")
print(df.columns.tolist())

print("\nDataset shape:", df.shape)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Standardize common HbA1c column variants
if "hba1c_level" not in df.columns and "hba1c" in df.columns:
    df = df.rename(columns={"hba1c": "hba1c_level"})

required = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hba1c_level",
    "blood_glucose_level", TARGET
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].copy()
df = df.drop_duplicates()

# Standardize categorical text
for col in ["gender", "smoking_history"]:
    df[col] = df[col].apply(
        lambda value: value.strip()
        if isinstance(value, str) else value
    )

# Convert numeric columns safely
numeric_cols = [
    "age", "hypertension", "heart_disease",
    "bmi", "hba1c_level", "blood_glucose_level"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing target labels
df = df.dropna(subset=[TARGET])

# Convert target into 0/1
target_values = df[TARGET].astype(str).str.strip().str.lower()
target_map = {
    "0": 0, "1": 1,
    "no": 0, "yes": 1,
    "false": 0, "true": 1,
    "no diabetes": 0, "diabetes": 1
}

mapped_target = target_values.map(target_map)

if mapped_target.isna().any():
    raise ValueError(
        "Target must contain recognized binary diabetes labels. "
        f"Found: {target_values.unique().tolist()}"
    )

df[TARGET] = mapped_target.astype(int)

if df[TARGET].nunique() != 2:
    raise ValueError("Dataset must contain both target classes: 0 and 1.")

# Remove rows missing categorical values only if needed?
# Pipeline imputes missing categorical and numerical feature values.

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = ["gender", "smoking_history"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 2. Split BEFORE fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 4. Build and train Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 5. Evaluate on held-out test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
report = classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["No diabetes", "Diabetes"],
    output_dict=True,
    zero_division=0
)

# 6. Save model and evaluation data
joblib.dump(model, ARTIFACT_DIR / "diabetes_model.joblib")

eval_df = X_test.copy()
eval_df["actual"] = y_test.to_numpy()
eval_df["predicted"] = y_pred
eval_df.to_csv(ARTIFACT_DIR / "evaluation.csv", index=False)

df.to_csv(ARTIFACT_DIR / "cleaned_dataset.csv", index=False)

metadata = {
    "features": X.columns.tolist(),
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "categories": {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    },
    "ranges": {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max())
        }
        for col in numeric_cols
        if X_train[col].notna().any()
    },
    "class_distribution": {
        str(int(k)): int(v)
        for k, v in y.value_counts().sort_index().items()
    },
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test))
}

with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("=" * 55)
print("MODEL TRAINING COMPLETED")
print("=" * 55)
print(f"Dataset rows after cleaning: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(cm)
print(f"\nSaved model and reports to: {ARTIFACT_DIR}")

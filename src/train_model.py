from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


# ======================= PATH SETTINGS ======================= #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "nonprofits_train.csv"
VAL_PATH = DATA_DIR / "nonprofits_val.csv"
TEST_PATH = DATA_DIR / "nonprofits_test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "baseline_logreg.pkl"

# Your chosen target column
TARGET_COL = "Name_Length_Bin_medium"


# ======================= LOADING HELPERS ===================== #

def load_split(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load split CSV and separate features/target."""
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y


# ======================= PREPROCESSOR ======================== #

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features."""
    # Infer column types
    numeric_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: List[str] = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


# ======================= MODEL PIPELINE ====================== #

def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Create full preprocessing + model pipeline."""
    preprocessor = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    return model


# ======================= EVALUATION ========================== #

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "val",
) -> None:
    """Print metrics for a given split."""
    y_pred = model.predict(X)

    # Some baselines will be non-probabilistic; handle gracefully
    try:
        y_proba = model.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, y_proba)
    except Exception:
        y_proba = None
        roc = None

    acc = accuracy_score(y, y_pred)

    print(f"\n===== Evaluation on {split_name} set =====")
    print(f"Accuracy: {acc:.4f}")
    if roc is not None:
        print(f"ROC-AUC : {roc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification report:")
    print(classification_report(y, y_pred))

# =================== FEATURE IMPORTANCE ====================== #

def show_feature_importance(model: Pipeline, X_sample: pd.DataFrame, top_n: int = 20) -> None:
    """
    Extract and print feature importances (LogReg coefficients)
    mapped back to real feature names after preprocessing.
    """
    print("\n===== Feature Importance =====")

    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    clf: LogisticRegression = model.named_steps["classifier"]

    # Extract feature names
    numeric_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = onehot.get_feature_names_out(cat_cols)

    feature_names = list(numeric_cols) + list(cat_feature_names)
    coefs = clf.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    print(f"\nTop {top_n} most influential features:")
    print(importance.head(top_n).to_string(index=False))

# ======================= MAIN SCRIPT ========================= #

def main() -> None:
    print("Loading data...")
    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)
    X_test, y_test = load_split(TEST_PATH)

    print("Building model pipeline...")
    model = build_model_pipeline(X_train)

    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    # Evaluate on val and test
    evaluate_model(model, X_val, y_val, split_name="val")
    evaluate_model(model, X_test, y_test, split_name="test")
        # Show feature importance
    show_feature_importance(model, X_train)


    # Save model
    print(f"\nSaving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()

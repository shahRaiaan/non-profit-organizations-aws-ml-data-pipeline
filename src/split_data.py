from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ======================= PATH SETTINGS ======================= #

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_PATH = DATA_DIR / "processed" / "nonprofits_features.csv"

TRAIN_PATH = DATA_DIR / "processed" / "nonprofits_train.csv"
VAL_PATH   = DATA_DIR / "processed" / "nonprofits_val.csv"
TEST_PATH  = DATA_DIR / "processed" / "nonprofits_test.csv"

# Target column (make Tsure this EXACT name exists in nonprofits_features.csv)
TARGET_COL = "Name_Length_Bin_medium"


# ======================= LOADING ============================= #

def load_feature_data(path: Path = FEATURE_PATH) -> pd.DataFrame:
    """Load feature-engineered data & drop index-like unnamed columns."""
    df = pd.read_csv(path)

    # Remove auto-index column if present
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    return df


# ======================= SPLITTING LOGIC ===================== #

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split df into train / validation / test sets.
    Final proportions ≈ Train 60% / Val 20% / Test 20%
    """

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found. Columns: {list(df.columns)}")

    # Features (X) + Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off TEST
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Now split train_val → train + val
    val_fraction = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=random_state,
    )

    # Reattach target columns
    train_df = X_train.copy()
    train_df[target_col] = y_train

    val_df = X_val.copy()
    val_df[target_col] = y_val

    test_df = X_test.copy()
    test_df[target_col] = y_test

    return train_df, val_df, test_df


# ======================= SAVE RESULTS ======================== #

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nSaved:")
    print(f"  TRAIN → {TRAIN_PATH}")
    print(f"  VAL   → {VAL_PATH}")
    print(f"  TEST  → {TEST_PATH}")


# ======================= MAIN ENTRY ========================== #

def main():
    print(f"Loading data from: {FEATURE_PATH}")
    df = load_feature_data()

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target column: {TARGET_COL}")

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))

    train_df, val_df, test_df = split_data(df)

    print("\nSplit shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ======================= PATH SETTINGS ======================= #

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Input cleaned data csv (pipe-separated)
CLEANED_DATA_PATH = DATA_DIR / "processed" / "nonprofits_clean.csv"
# Output feature engineered data csv
FEATURE_ENGINEERED_DATA_PATH = DATA_DIR / "processed" / "nonprofits_features.csv"

# ======================= LOADING ================== #
def load_cleaned_data(file_path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned nonprofit data.
    Assumes:
    - all data cleaning already applied
    """
    df = pd.read_csv(file_path, sep="|")
    return df

# ====================== FEATURE SPLITTING ===================== #
def add_feature_splitting(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature splitting transformations."""
    df = df.copy()

    # 1) Name based Splits
    if "Name" in df.columns:
        name_str = df["Name"].astype(str)
        df["Name_Length"] = name_str.str.len()
        df["Name_Word_Count"] = name_str.str.split().str.len()

    # Classification code prefix
    if "Classification Code" in df.columns:
        code_str = df["Classification Code"].astype(str)
        df["Class_Prefix"] = code_str.str[0:2]
    return df

# ======================= BINNING =========================== #
def add_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Add binning transformations for specified columns.
    - Name_Length --> short/medium/long buckets
    - if name is missing, this step is skipped
    """

    df = df.copy()
    if "Name_Length" in df.columns:
        df["Name_Length_Bin"] = pd.cut(
            df["Name_Length"],
            bins=[-1, 10, 30, np.inf],
            labels=["short", "medium", "long"]
        )
        print("[binning] Added Name_Length_Bin")
    return df

# ======================= FREQUENCY + LOG TRANSFORM ======================= #

def add_frequency_and_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "City" in df.columns:
        city_counts = df["City"].value_counts()
        df["city_freq"] = df["City"].map(city_counts)

    if "State" in df.columns:
        state_counts = df["State"].value_counts()
        df["state_freq"] = df["State"].map(state_counts)

    if "city_freq" in df.columns:
        df["city_freq_log"] = np.log1p(df["city_freq"])

    if "state_freq" in df.columns:
        df["state_freq_log"] = np.log1p(df["state_freq"])

    return df


# ======================= NORMALIZATION & SCALING ======================= #

def add_normalization_and_scaling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "name_length" in df.columns:
        mm_scaler = MinMaxScaler()
        df["name_length_norm"] = mm_scaler.fit_transform(df[["name_length"]])
        print("[normalize] Added name_length_norm")

    numeric_candidates = [
        "name_length",
        "name_word_count",
        "city_freq",
        "state_freq",
        "city_freq_log",
        "state_freq_log",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"[scale] Standard-scaled numeric columns: {numeric_cols}")

    return df

# ======================= ENCODING CATEGORIZATION =========================== #
def add_encoding_categorization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple one-hot encoding to low-cardinality categoricals:
    - State
    - Country
    - Class_Prefix
    - Name_Length_Bin

    EIN remains as identifier and is not encoded
    """
    df = df.copy()
    cat_candidates = ["State", "Country", "Class_Prefix", "Name_Length_Bin"]
    categorical_cols = [c for c in cat_candidates if c in df.columns]

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"[encoding] One-hot encoded categorical columns: {categorical_cols}")
    
    return df

# ======================= MAIN PIPELINE =========================== #
def run_feature_engineering() -> pd.DataFrame:
    """Feature-engineering pipeline (assumes input is already clean)
     1) Load cleaned data
     2) Feature Splitting
     3) Binning
     4) Frequency + Log Transform
     5) Normalization + Scaling
     6) Encoding + Categorization
     7) Save to disk
    """

    print("\n>>>> Starting feature_engineering.py <<<<")
    print("[step] Loading cleaned data..." )
    df = load_cleaned_data()
    print("[step] Applying feature engineering transformations..." )
    df = add_feature_splitting(df)
    df = add_binning(df)
    df = add_frequency_and_log(df)
    df = add_normalization_and_scaling(df)
    df = add_encoding_categorization(df)

    # Save to disk
    FEATURE_ENGINEERED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_ENGINEERED_DATA_PATH, index=False)
    print(f"[step] Saved feature-engineered data to {FEATURE_ENGINEERED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_feature_engineering()

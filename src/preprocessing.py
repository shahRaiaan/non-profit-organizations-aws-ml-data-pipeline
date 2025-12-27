from pathlib import Path
import pandas as pd

# ======================= PATH SETTINGS ======================= #

# Root data directory (auto-resolves relative to project)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RAW_DATA_PATH = DATA_DIR / "raw.txt"  # Input raw file
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "nonprofits_clean.csv"  # Output cleaned file


# ======================= LOAD DATA ============================ #

def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw nonprofit data (TAB separated, no header)."""
    df = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=["EIN", "Name", "City", "State", "Country", "Classification Code"],
        dtype={"EIN": str}
    )
    return df


# ======================= CLEANING RULES ======================== #

def clean_nonprofits(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data cleaning rules column-wise."""
    df = df.copy()

    # 1) Normalize whitespace for all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 2) Drop rows with missing EIN or Nonprofit Name
   
    df["EIN"] = df["EIN"].astype(str).str.replace(r"\D", "", regex=True)
    df = df[df["EIN"].str.len() == 9]
    df = df.dropna(subset=["EIN", "Name"])

    # 3) Impute City
    df["City"] = df["City"].fillna("Unknown_City")

    # 4) Impute State
    df["State"] = df["State"].fillna("Unknown_State")

    # 5) Impute Classification Code
    df["Classification Code"] = df["Classification Code"].fillna(
        "Missing_Classification_Code"
    )

    # 6) Deduplicate EIN (keep first)
    df = df.drop_duplicates(subset=["EIN"], keep="first")

    return df


# ======================= SAVE RESULT =========================== #

def save_processed_data(df: pd.DataFrame, file_path: Path = PROCESSED_DATA_PATH) -> None:
    """Save cleaned data to file (TAB-separated output)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, sep="|", index=False)
    print(f"\n✔ Data preprocessing complete.\n✔ Saved to: {file_path.resolve()}\n")


# ======================= MAIN EXECUTION ======================== #

def main() -> None:
    print("\n>>>> Starting preprocessing.py <<<<")
    raw_df = load_raw_data()
    cleaned_df = clean_nonprofits(raw_df)
    save_processed_data(cleaned_df)


if __name__ == "__main__":
    main()

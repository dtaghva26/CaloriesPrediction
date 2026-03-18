# pipeline/data.py
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# -------------------------------
# Data container (clean structure)
# -------------------------------
@dataclass
class Dataset:
    features: pd.DataFrame
    target: pd.Series


# -------------------------------
# Load raw dataset
# -------------------------------
def load_ds(path: str) -> pd.DataFrame:
    """Load raw dataset from CSV"""
    return pd.read_csv(path)


# -------------------------------
# Cleaning logic
# -------------------------------
def clean_ds(df: pd.DataFrame, save_path: str) -> None:
    """
    Clean raw dataset and save processed version
    """

    df = df.copy()

    # Drop missing values
    df.dropna(inplace=True)

    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]

    # Separate target column
    TARGET_COLUMN = "calories"

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    # Identify categorical columns (excluding target)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target if accidentally included
    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)

    # One-hot encode categorical features
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save cleaned dataset
    df.to_csv(save_path, index=False)


# -------------------------------
# Load cleaned dataset
# -------------------------------
def load_cleaned_ds(path: str) -> Dataset:
    """
    Load cleaned dataset and split into features/target
    """

    df = pd.read_csv(path)

    # IMPORTANT: adjust target column name
    TARGET_COLUMN = "calories"

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return Dataset(features=X, target=y)
# ----------------------------
# Data
# ----------------------------
def load_and_prepare_data():
    print("📥 Loading raw data...")
    dataset = load_ds("dataset/raw/archive/calories.csv")

    print("🧹 Cleaning data...")
    clean_ds(dataset, "dataset/preprocessed/calories_clean.csv")

    print("📦 Loading cleaned data...")
    dataset = load_cleaned_ds("dataset/preprocessed/calories_clean.csv")

    X, y = dataset.features, dataset.target

    print(f"Dataset shape: {X.shape}")
    return X, y
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
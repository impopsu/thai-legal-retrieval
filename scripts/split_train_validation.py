import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


project_root = Path(__file__).resolve().parents[1]
train_path = project_root / "data/raw/train-00000-of-00001.parquet"

df = pd.read_parquet(train_path)


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


train_df.to_parquet(
    project_root / "data/processed/train_retrieval.parquet",
    index=False
)

val_df.to_parquet(
    project_root / "data/processed/validation_retrieval.parquet",
    index=False
)


print("Original:", len(df))

print("Train:", len(train_df))

print("Validation:", len(val_df))
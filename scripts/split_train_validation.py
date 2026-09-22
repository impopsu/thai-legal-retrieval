import pandas as pd

from sklearn.model_selection import train_test_split


train_path = "../data/raw/train-00000-of-00001.parquet"

df = pd.read_parquet(train_path)


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


train_df.to_parquet(
    "../data/processed/train_retrieval.parquet",
    index=False
)

val_df.to_parquet(
    "../data/processed/validation_retrieval.parquet",
    index=False
)


print("Original:", len(df))

print("Train:", len(train_df))

print("Validation:", len(val_df))
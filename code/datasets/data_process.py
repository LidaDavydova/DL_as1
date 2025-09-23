import os
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "../../data/raw/raw_data.csv")
TRAIN_FILE = os.path.join(BASE_DIR, "../../data/processed/train_data.csv")
TEST_FILE = os.path.join(BASE_DIR, "../../data/processed/test_data.csv")

TARGET_COLUMN = "Wine"  # column with class labels

df = pd.read_csv(RAW_FILE)

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# Remove outliers (percentile-based)
numeric_cols = df.select_dtypes(include=['float', 'int']).columns.drop(TARGET_COLUMN)
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Split into train/test (stratified)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN]
)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Stage 1 complete: train_data.csv and test_data.csv created in {os.path.dirname(TRAIN_FILE)}")

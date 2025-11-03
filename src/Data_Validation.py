import os
import pandas as pd

def load_csv_safe(path):
    """Load CSV if it exists, otherwise print an error."""
    if not os.path.exists(path):
        print(f" File not found: {path}")
        return None
    print(f" Loaded: {path}")
    return pd.read_csv(path)

def check_missing(df, name):
    """Count missing values."""
    missing = df.isnull().sum().sum()
    print(f" Missing values in {name}: {missing}")
    if missing > 0:
        print(df.isnull().sum())

def check_duplicates(df, name):
    """Check for duplicate rows."""
    dup = df.duplicated().sum()
    print(f" Duplicates in {name}: {dup}")

def compare_columns(train_df, test_df):
    """Compare column names and order between train/test."""
    train_cols = train_df.columns.tolist()
    test_cols = test_df.columns.tolist()
    if train_cols == test_cols:
        print(" Train and Test have matching columns.")
    else:
        print("Column mismatch detected!")
        print("Train-only columns:", set(train_cols) - set(test_cols))
        print("Test-only columns:", set(test_cols) - set(train_cols))

def validate_ranges(df):
    """Basic sanity checks on numeric columns."""
    print("\n Range checks:")
    print("ApplicantIncome:", df["ApplicantIncome"].describe())
    print("LoanAmount:", df["LoanAmount"].describe())

def main():
    # Adjust to your folder setup
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

    train_path = os.path.join(CLEAN_DIR, "loan_cleaned_train.csv")
    test_path = os.path.join(CLEAN_DIR, "loan_cleaned_test.csv")

    print(" Starting Data Validation...\n")

    train_df = load_csv_safe(train_path)
    test_df = load_csv_safe(test_path)
    if train_df is None or test_df is None:
        return

    compare_columns(train_df, test_df)
    check_missing(train_df, "Train")
    check_missing(test_df, "Test")
    check_duplicates(train_df, "Train")
    check_duplicates(test_df, "Test")
    validate_ranges(train_df)

    print("\n Data validation complete. Ready for modeling!")

if __name__ == "__main__":
    main()
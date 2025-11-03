# Loan Eligibility Data Cleaner
# This script cleans both Train and Test data files consistently.
#
# How to use:
#   python src/Clean_Loan_Data.py
#
# Requirements:
#   - Ensure 'loan-train.csv' and 'loan-test.csv' exist under data/raw/
#   - The script will output cleaned files to data/cleaned/

import os
import pandas as pd
import numpy as np


# ------------------------------
# 1. Compute fill values (from training data)
# ------------------------------
def compute_fill_values(df):
    fill_values = {
        "categorical_modes": {
            "Gender": df["Gender"].mode()[0],
            "Married": df["Married"].mode()[0],
            "Dependents": df["Dependents"].mode()[0],
            "Self_Employed": df["Self_Employed"].mode()[0],
        },
        "numeric_medians": {
            "LoanAmount": df["LoanAmount"].median(),
            "Loan_Amount_Term": df["Loan_Amount_Term"].median(),
        },
        "credit_mode": df["Credit_History"].mode()[0],
    }
    return fill_values


# ------------------------------
# 2. Clean dataset function
# ------------------------------
def clean_dataset(df, fill_values):
    df = df.copy()

    # Fill missing (Categorical)
    for col, mode in fill_values["categorical_modes"].items():
        df[col] = df[col].fillna(mode)

    # Fill missing (Numeric)
    for col, median in fill_values["numeric_medians"].items():
        df[col] = df[col].fillna(median)

    # Fill missing Credit History
    df["Credit_History"] = df["Credit_History"].fillna(fill_values["credit_mode"])

    # Standardize categorical values
    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
            df[col] = df[col].fillna(fill_values["categorical_modes"].get(col, "Unknown"))

    # Convert 'Dependents' (e.g., '3+' → 3)
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({"3+": "3"}).astype(int)

    # Ensure numeric columns are truly numeric
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cap outliers using IQR
    def cap_outliers(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return series.clip(lower, upper)

    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        if col in df.columns:
            df[col] = cap_outliers(df[col])

    # Drop Loan_ID (not predictive)
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)
    
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


# ------------------------------
# 3. Main function
# ------------------------------
def main():
    # Automatically locate project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

    # Input and output file paths
    train_path = os.path.join(RAW_DIR, "loan-train.csv")
    test_path = os.path.join(RAW_DIR, "loan-test.csv")
    cleaned_train_path = os.path.join(CLEAN_DIR, "loan_cleaned_train.csv")
    cleaned_test_path = os.path.join(CLEAN_DIR, "loan_cleaned_test.csv")

    print(" Loading training data...")
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")

    print(" Computing imputation values...")
    fill_values = compute_fill_values(train_df)

    print(" Cleaning training data...")
    train_clean = clean_dataset(train_df, fill_values)

    print(" Cleaning test data...")
    test_df = pd.read_csv(test_path)
    test_clean = clean_dataset(test_df, fill_values)

    # Validation summary
    print("\n Missing values summary:")
    print(f"Training missing: {train_clean.isna().sum().sum()}")
    print(f"Test missing: {test_clean.isna().sum().sum()}")

    # Save cleaned files
    os.makedirs(CLEAN_DIR, exist_ok=True)
    train_clean.to_csv(cleaned_train_path, index=False)
    test_clean.to_csv(cleaned_test_path, index=False)

    print("\n Cleaning complete!")
    print(f"Saved cleaned files to: {CLEAN_DIR}")


# ------------------------------
# 4. Entry point
# ------------------------------
if __name__ == "__main__":
    main()
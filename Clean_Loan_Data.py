#Loan Elegibilty Data Cleaner
#This script will be used to clean both Train and Test data files.

#How to use:
# "python Clean_Loan_Data.py"
# Requirements: Ensure both 'Loan-Train.csv' and 'Loan-Test.csv' exist in the same folder as this script.

import pandas as pd
import numpy as np

#Computing modes and medians to fill empty or missing values 
#(using training dataset for modes and medians and applying it 
#to test dataset in order to ensure there isn't data leaks.)
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

def clean_dataset(df, fill_values):
    df = df.copy()

    # Fill missing (Categorical) values 
    for col, mode in fill_values["categorical_modes"].items():
        df[col] = df[col].fillna(mode)
    
    # Fill missing (Numeric) values
    for col, median in fill_values["numeric_medians"].items():
        df[col] = df[col].fillna(median)

    # Fill missing credit history
    df["Credit_History"] = df["Credit_History"].fillna(fill_values["credit_mode"])

    # Clean string categories to ensure consistency
    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
            df[col] = df[col].fillna(fill_values["categorical_modes"].get(col, "Unknown"))

    # Convert 'Dependents' column to integer only (e.g., "3+" to "3" 
    # and that is actually the only special value in that field -> '3+')
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({"3+": "3"}).astype(int)

    # Ensure numeric columns are numeric (as in not strings or characters)
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_Histroy"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")

    # Cap outliers (IQR - Interquartile Range) for income and loan amount
    def cap_outliers(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return series.clip(lower, upper)
    
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        if col in df.columns:
            df[col] = cap_outliers(df[col])

    # Drop ID column as it doesn't really have a use in Eligibilty Prediction
    if "Loan_ID" in df.columns:
        df.drop(columns = ["Loan_ID"], inplace = True)
    
    return df

def main():
    # Load Datasets
    train_path = "loan-train.csv"
    test_path = "loan-test.csv"

    print("Testing data..")
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")

    print("Computing imputation values from training data..")
    fill_values = compute_fill_values(train_df)

    print("Cleaning training data..")
    train_clean = clean_dataset(train_df, fill_values)

    print("Loading and cleaning test data using same values..")
    test_df = pd.read_csv(test_path)
    test_clean = clean_dataset(test_df, fill_values)

    # Validate
    print("\nMissing values check:")
    print("Traing missing: ", train_clean.isna().sum().sum())
    print("Test missing: ", test_clean.isna().sum().sum())

    # Save cleaned datasets
    train_clean.to_csv("loan_cleaned_train.csv", index=False)
    test_clean.to_csv("loan_cleaned_test.csv", index=False)

    print("\n Cleaning Complete :D")
    print("Saved: loan_cleaned_test.csv and loan_cleaned_test.csv")

if __name__ == "__main__":
    main()
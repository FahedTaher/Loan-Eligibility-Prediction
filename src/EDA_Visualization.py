"""
EDA & Visualization Script for Loan Eligibility Prediction
==========================================================
This script performs comprehensive exploratory data analysis:
- Feature distributions (raw vs cleaned)
- Correlation analysis
- Target relationships
- Class imbalance analysis
- Demographic breakdowns
- Visual comparisons showing data transformations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color palette
COLORS = sns.color_palette("husl", 8)


def load_data():
    """Load raw and cleaned training data."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(BASE_DIR, "data", "raw", "loan-train.csv")
    cleaned_path = os.path.join(BASE_DIR, "data", "cleaned", "loan_cleaned_train.csv")
    
    raw_df = pd.read_csv(raw_path)
    cleaned_df = pd.read_csv(cleaned_path)
    
    print(f"Raw data shape: {raw_df.shape}")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    return raw_df, cleaned_df


def create_output_dir():
    """Create output directory for plots."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, "reports", "eda_plots")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_feature_distributions_comparison(raw_df, cleaned_df, output_dir):
    """Plot distributions of numeric features: raw vs cleaned."""
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Raw data
        raw_data = raw_df[col].dropna()
        ax.hist(raw_data, bins=50, alpha=0.6, label='Raw', color='red', density=True)
        
        # Cleaned data
        cleaned_data = cleaned_df[col].dropna()
        ax.hist(cleaned_data, bins=50, alpha=0.6, label='Cleaned', color='blue', density=True)
        
        ax.set_title(f'{col} Distribution: Raw vs Cleaned', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_feature_distributions_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Feature distributions comparison")


def plot_missing_data_comparison(raw_df, cleaned_df, output_dir):
    """Visualize missing data in raw vs cleaned datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw data missing values
    raw_missing = raw_df.isnull().sum()
    raw_missing = raw_missing[raw_missing > 0].sort_values(ascending=False)
    ax1.barh(raw_missing.index, raw_missing.values, color='red', alpha=0.7)
    ax1.set_title('Missing Values in Raw Data', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Count of Missing Values')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Cleaned data missing values
    cleaned_missing = cleaned_df.isnull().sum()
    cleaned_missing = cleaned_missing[cleaned_missing > 0].sort_values(ascending=False)
    if len(cleaned_missing) > 0:
        ax2.barh(cleaned_missing.index, cleaned_missing.values, color='blue', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'No Missing Values', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    ax2.set_title('Missing Values in Cleaned Data', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Count of Missing Values')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_missing_data_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Missing data comparison")


def plot_outlier_comparison(raw_df, cleaned_df, output_dir):
    """Compare outliers before and after cleaning using boxplots."""
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        data_to_plot = [
            raw_df[col].dropna(),
            cleaned_df[col].dropna()
        ]
        
        bp = ax.boxplot(data_to_plot, tick_labels=['Raw', 'Cleaned'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.6)
        
        ax.set_title(f'{col} - Outlier Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_outlier_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outlier comparison")


def plot_correlation_heatmap(df, output_dir, suffix=''):
    """Plot correlation heatmap for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Loan_Status' in df.columns:
        # Convert Loan_Status to numeric for correlation
        df_plot = df.copy()
        df_plot['Loan_Status'] = df_plot['Loan_Status'].map({'Y': 1, 'N': 0})
        numeric_cols = [col for col in numeric_cols if col in df_plot.columns]
    else:
        df_plot = df.copy()
    
    corr_matrix = df_plot[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Heatmap{suffix}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'04_correlation_heatmap{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Correlation heatmap{suffix}")


def plot_target_relationships(df, output_dir, suffix=''):
    """Plot relationships between features and target variable."""
    if 'Loan_Status' not in df.columns:
        print("WARNING: Loan_Status not found, skipping target relationships")
        return
    
    # Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Class distribution pie chart
    ax = axes[0, 0]
    loan_status_counts = df['Loan_Status'].value_counts()
    ax.pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax.set_title('Loan Status Distribution', fontsize=12, fontweight='bold')
    
    # 2. Numeric features vs target
    ax = axes[0, 1]
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    for col in numeric_cols:
        if col in df.columns:
            df_grouped = df.groupby('Loan_Status')[col].mean()
            ax.bar([f"{idx}_{col[:3]}" for idx in df_grouped.index], 
                   df_grouped.values, alpha=0.7, label=col[:10])
    ax.set_title('Average Numeric Features by Loan Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Value')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Categorical features vs target
    ax = axes[1, 0]
    categorical_cols = ["Gender", "Married", "Education", "Property_Area"]
    for col in categorical_cols:
        if col in df.columns:
            crosstab = pd.crosstab(df[col], df['Loan_Status'], normalize='index') * 100
            crosstab.plot(kind='bar', ax=ax, alpha=0.7)
            ax.set_title('Categorical Features vs Loan Status', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)')
            ax.legend(title='Loan Status')
            ax.tick_params(axis='x', rotation=45)
            break  # Show first categorical feature
    
    # 4. Credit History vs Loan Status
    ax = axes[1, 1]
    if 'Credit_History' in df.columns:
        credit_crosstab = pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index') * 100
        credit_crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Credit History vs Loan Status', fontsize=12, fontweight='bold')
        ax.set_xlabel('Credit History (0=No, 1=Yes)')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'05_target_relationships{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Target relationships{suffix}")


def plot_class_imbalance(df, output_dir, suffix=''):
    """Analyze and visualize class imbalance."""
    if 'Loan_Status' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    ax = axes[0]
    counts = df['Loan_Status'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax.set_title('Loan Status Class Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Loan Status')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom')
    
    # Imbalance ratio
    ax = axes[1]
    imbalance_ratio = counts['Y'] / counts['N'] if 'N' in counts.index else 0
    ax.text(0.5, 0.6, f'Class Imbalance Ratio\n(Y/N): {imbalance_ratio:.2f}', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if imbalance_ratio > 1.5 or imbalance_ratio < 0.67:
        ax.text(0.5, 0.3, 'WARNING: Imbalanced Dataset\n(Consider resampling)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                color='red', fontweight='bold')
    else:
        ax.text(0.5, 0.3, 'OK: Balanced Dataset', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                color='green', fontweight='bold')
    
    ax.axis('off')
    ax.set_title('Class Balance Assessment', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'06_class_imbalance{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Class imbalance analysis{suffix}")


def plot_demographic_breakdown(df, output_dir, suffix=''):
    """Plot demographic breakdowns."""
    if 'Loan_Status' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gender breakdown
    ax = axes[0, 0]
    if 'Gender' in df.columns:
        gender_loan = pd.crosstab(df['Gender'], df['Loan_Status'])
        gender_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Gender', fontsize=12, fontweight='bold')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=0)
    
    # Married status
    ax = axes[0, 1]
    if 'Married' in df.columns:
        married_loan = pd.crosstab(df['Married'], df['Loan_Status'])
        married_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Marital Status', fontsize=12, fontweight='bold')
        ax.set_xlabel('Married')
        ax.set_ylabel('Count')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=0)
    
    # Education
    ax = axes[1, 0]
    if 'Education' in df.columns:
        edu_loan = pd.crosstab(df['Education'], df['Loan_Status'])
        edu_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Education', fontsize=12, fontweight='bold')
        ax.set_xlabel('Education')
        ax.set_ylabel('Count')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=45)
    
    # Property Area
    ax = axes[1, 1]
    if 'Property_Area' in df.columns:
        area_loan = pd.crosstab(df['Property_Area'], df['Loan_Status'])
        area_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Property Area', fontsize=12, fontweight='bold')
        ax.set_xlabel('Property Area')
        ax.set_ylabel('Count')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'07_demographic_breakdown{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Demographic breakdown{suffix}")


def plot_feature_distributions_by_target(df, output_dir, suffix=''):
    """Plot distributions of numeric features stratified by target."""
    if 'Loan_Status' not in df.columns:
        return
    
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, col in enumerate(numeric_cols):
        if col not in df.columns:
            continue
        
        ax = axes[idx]
        
        for status in ['Y', 'N']:
            data = df[df['Loan_Status'] == status][col].dropna()
            ax.hist(data, bins=30, alpha=0.6, label=f'Status={status}', density=True)
        
        ax.set_title(f'{col} Distribution by Loan Status', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'08_feature_distributions_by_target{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Feature distributions by target{suffix}")


def plot_additional_insights(df, output_dir, suffix=''):
    """Plot additional insights and patterns."""
    if 'Loan_Status' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dependents vs Loan Status
    ax = axes[0, 0]
    if 'Dependents' in df.columns:
        dependents_loan = pd.crosstab(df['Dependents'], df['Loan_Status'], normalize='index') * 100
        dependents_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Approval Rate by Number of Dependents', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Dependents')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=0)
    
    # Self Employed vs Loan Status
    ax = axes[0, 1]
    if 'Self_Employed' in df.columns:
        self_emp_loan = pd.crosstab(df['Self_Employed'], df['Loan_Status'], normalize='index') * 100
        self_emp_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Self Employment', fontsize=12, fontweight='bold')
        ax.set_xlabel('Self Employed')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=0)
    
    # Loan Amount Term vs Loan Status
    ax = axes[1, 0]
    if 'Loan_Amount_Term' in df.columns:
        term_loan = df.groupby(['Loan_Amount_Term', 'Loan_Status']).size().unstack(fill_value=0)
        term_loan_pct = term_loan.div(term_loan.sum(axis=1), axis=0) * 100
        term_loan_pct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Loan Amount Term', fontsize=12, fontweight='bold')
        ax.set_xlabel('Loan Amount Term (days)')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=45)
    
    # Combined Income vs Loan Status
    ax = axes[1, 1]
    if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['TotalIncome_binned'] = pd.cut(df['TotalIncome'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        income_loan = pd.crosstab(df['TotalIncome_binned'], df['Loan_Status'], normalize='index') * 100
        income_loan.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title('Loan Status by Total Income (Binned)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Total Income Category')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Loan Status')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'09_additional_insights{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Additional insights{suffix}")


def generate_summary_statistics(raw_df, cleaned_df, output_dir):
    """Generate and save summary statistics."""
    summary = []
    
    summary.append("=" * 80)
    summary.append("EDA SUMMARY STATISTICS")
    summary.append("=" * 80)
    summary.append("\n")
    
    summary.append("RAW DATA STATISTICS:")
    summary.append("-" * 80)
    summary.append(f"Shape: {raw_df.shape}")
    summary.append(f"Missing values: {raw_df.isnull().sum().sum()}")
    summary.append(f"Duplicates: {raw_df.duplicated().sum()}")
    summary.append("\nMissing values by column:")
    summary.append(str(raw_df.isnull().sum()[raw_df.isnull().sum() > 0]))
    
    summary.append("\n\nCLEANED DATA STATISTICS:")
    summary.append("-" * 80)
    summary.append(f"Shape: {cleaned_df.shape}")
    summary.append(f"Missing values: {cleaned_df.isnull().sum().sum()}")
    summary.append(f"Duplicates: {cleaned_df.duplicated().sum()}")
    
    if 'Loan_Status' in cleaned_df.columns:
        summary.append("\n\nTARGET VARIABLE:")
        summary.append("-" * 80)
        summary.append(str(cleaned_df['Loan_Status'].value_counts()))
        summary.append(f"\nClass ratio (Y/N): {cleaned_df['Loan_Status'].value_counts()['Y'] / cleaned_df['Loan_Status'].value_counts()['N']:.2f}")
    
    summary.append("\n\nNUMERIC FEATURES SUMMARY:")
    summary.append("-" * 80)
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    summary.append(str(cleaned_df[numeric_cols].describe()))
    
    summary_text = "\n".join(summary)
    
    # Save to file
    summary_path = os.path.join(output_dir, '00_summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n✓ Saved: Summary statistics to {summary_path}")


def main():
    """Main function to run all EDA visualizations."""
    print("\n" + "=" * 80)
    print("EDA & VISUALIZATION FOR LOAN ELIGIBILITY PREDICTION")
    print("=" * 80 + "\n")
    
    # Load data
    raw_df, cleaned_df = load_data()
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}\n")
    
    # Generate summary statistics
    generate_summary_statistics(raw_df, cleaned_df, output_dir)
    
    # 1. Feature distributions comparison (raw vs cleaned)
    plot_feature_distributions_comparison(raw_df, cleaned_df, output_dir)
    
    # 2. Missing data comparison
    plot_missing_data_comparison(raw_df, cleaned_df, output_dir)
    
    # 3. Outlier comparison
    plot_outlier_comparison(raw_df, cleaned_df, output_dir)
    
    # 4. Correlation heatmap (cleaned data)
    plot_correlation_heatmap(cleaned_df, output_dir, suffix='_cleaned')
    
    # 5. Target relationships (cleaned data)
    plot_target_relationships(cleaned_df, output_dir, suffix='_cleaned')
    
    # 6. Class imbalance analysis
    plot_class_imbalance(cleaned_df, output_dir, suffix='_cleaned')
    
    # 7. Demographic breakdown
    plot_demographic_breakdown(cleaned_df, output_dir, suffix='_cleaned')
    
    # 8. Feature distributions by target
    plot_feature_distributions_by_target(cleaned_df, output_dir, suffix='_cleaned')
    
    # 9. Additional insights
    plot_additional_insights(cleaned_df, output_dir, suffix='_cleaned')
    
    print("\n" + "=" * 80)
    print("EDA VISUALIZATION COMPLETE!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


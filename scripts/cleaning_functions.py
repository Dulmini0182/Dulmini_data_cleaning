import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load hotel bookings dataset from CSV"""
    return pd.read_csv(filepath)

def analyze_missing_values(df):
    """Print missing value report and show heatmap"""
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    }).sort_values(by='Percentage', ascending=False)
    print("Missing Value Report:")
    print(missing_report[missing_report['Missing Values'] > 0])

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

def handle_missing_values(df):
    """Handle missing values in children, country, agent, company"""
    df['children'].fillna(0, inplace=True)
    df['country'].fillna(df['country'].mode()[0], inplace=True)
    df['agent'].fillna(0, inplace=True)
    df['agent'] = df['agent'].astype(int)
    df['company'].fillna(0, inplace=True)
    df['company'] = df['company'].astype(int)
    return df

def remove_duplicates(df):
    """Remove exact duplicate rows"""
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records found: {duplicate_count}")
    df = df.drop_duplicates()
    return df

def treat_outliers_adr(df):
    """Cap outliers in 'adr' using IQR method"""
    Q1 = df['adr'].quantile(0.25)
    Q3 = df['adr'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[df['adr'] > upper_bound]
    print(f"Outliers in 'adr': {outliers.shape[0]}")

    df['adr'] = np.where(df['adr'] > upper_bound, upper_bound, df['adr'])
    return df

def fix_inconsistencies(df):
    """Fix date format and remove zero guest rows"""
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                        df['arrival_date_month'] + '-' +
                                        df['arrival_date_day_of_month'].astype(str), errors='coerce')

    df = df[~((df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0))]
    df['country'] = df['country'].str.upper()

    assert (df['adults'] + df['children'] + df['babies'] > 0).all(), "Some bookings have zero guests!"
    return df

def validate_data(df):
    """Print arrival date range and adr range"""
    print("Arrival date range:", df['arrival_date'].min(), "to", df['arrival_date'].max())
    print(f"ADR range: {df['adr'].min()} to {df['adr'].max()}")

def save_cleaned_data(df, filename='hotel_bookings_cleaned.csv'):
    """Save cleaned dataframe to CSV"""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to {filename}")

if __name__ == '__main__':
    df = load_data('hotel_bookings.csv')
    analyze_missing_values(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = treat_outliers_adr(df)
    df = fix_inconsistencies(df)
    validate_data(df)
    save_cleaned_data(df)

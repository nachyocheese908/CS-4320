"""
Load and verify Kaggle Premier League data for Assignment 1 Part B
"""

import pandas as pd
import os
import sys

def find_data_file():
    """Find the Kaggle CSV file in data/raw"""
    data_dir = 'data/raw'
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return None
    
    # List all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None
    
    print(f"Found CSV files: {csv_files}")
    
    # Try to find the most likely Premier League file
    pl_keywords = ['premier', 'epl', 'pl_', '_pl', 'english', '2020', '2021', 'season']
    
    for file in csv_files:
        file_lower = file.lower()
        if any(keyword in file_lower for keyword in pl_keywords):
            return os.path.join(data_dir, file)
    
    # Return the first CSV if no Premier League file found
    return os.path.join(data_dir, csv_files[0])

def load_and_inspect():
    """Load the data and provide inspection output""" 
    # Find the data file
    data_path = find_data_file()
    
    if not data_path:
        print("ERROR: No data file found. Please ensure your Kaggle CSV is in data/raw/")
        return None
    
    print(f"\n1. Loading data from: {data_path}")
    
    try:
        # Load the CSV
        df = pd.read_csv(data_path)
        print(f"   ✓ Successfully loaded data")
        print(f"   ✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Basic inspection
        print(f"\n2. Data Structure:")
        print(f"   Columns ({len(df.columns)} total):")
        for i, col in enumerate(df.columns[:15], 1):  # Show first 15 columns
            print(f"     {i:2d}. {col}")
        
        if len(df.columns) > 15:
            print(f"     ... and {len(df.columns) - 15} more columns")
        
        print(f"\n3. Sample Data (first 3 rows):")
        # Try to show interesting columns
        interesting_cols = []
        possible_cols = ['player', 'name', 'team', 'club', 'position', 'pos', 
                        'goals', 'assists', 'age', 'minutes', 'value', 'market_value']
        
        for col in possible_cols:
            for df_col in df.columns:
                if col in df_col.lower():
                    interesting_cols.append(df_col)
        
        # Show up to 5 interesting columns
        display_cols = interesting_cols[:5] if interesting_cols else df.columns[:5]
        print(df[display_cols].head(10).to_string())
        
        print(f"\n4. Data Types:")
        print(df.dtypes.value_counts())
        
        print(f"\n5. Missing Values Summary:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False).head(10))
        
        print(f"\n6. Basic Statistics for Numeric Columns:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols[:5]].describe().to_string())  # First 5 numeric columns
        
        
    
        return df
        
    except Exception as e:
        print(f"   ✗ Error loading data: {type(e).__name__}: {e}")
        return None

if __name__ == "__main__":
    df = load_and_inspect()
    
    # Save a sample for the assignment report
    if df is not None:
        os.makedirs('outputs', exist_ok=True)
        sample_path = 'outputs/data_sample.txt'
        with open(sample_path, 'w') as f:
            f.write("Data Sample (first 5 rows):\n")
            f.write("=" * 50 + "\n")
            f.write(df.head().to_string())
            f.write(f"\n\nDataset shape: {df.shape}\n")
            f.write(f"File size: {os.path.getsize(find_data_file()) / 1024:.1f} KB\n")
        print(f"\nSample saved to: {sample_path}")
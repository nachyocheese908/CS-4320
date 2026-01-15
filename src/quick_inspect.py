"""
Quick inspection script for Assignment 1 Part B
"""

import pandas as pd

def main():
    
    # Load the raw data
    df = pd.read_csv('data/raw/rawPLData.csv')
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} players × {df.shape[1]} features")
    
    # Remove duplicate columns for display
    df_display = df.loc[:, ~df.columns.duplicated()]
    
    print(f"\nKey columns available:")
    key_cols = [
        'Name', 'Club', 'Position', 'Nationality', 'Age',
        'Appearances', 'Goals', 'Assists', 'Clean sheets'
    ]
    
    for col in key_cols:
        if col in df_display.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (not found)")
    
    print(f"\nSample of players (showing key stats):")
    sample = df_display.head(10)[['Name', 'Club', 'Position', 'Age', 'Goals', 'Assists']]
    print(sample.to_string(index=False))
    
    print(f"\nPosition distribution:")
    if 'Position' in df_display.columns:
        pos_counts = df_display['Position'].value_counts().head(5)
        for pos, count in pos_counts.items():
            print(f"  {pos}: {count} players")
    
    print(f"\nThis dataset contains {df.shape[0]} Premier League players")
    print(f"with {df.shape[1]} performance metrics for the 2020-2021 season.")

if __name__ == "__main__":
    main()
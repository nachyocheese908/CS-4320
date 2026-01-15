"""
Clean data loader for Premier League capstone project
Handles duplicate columns and provides better inspection
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_clean_data():
    """Load the Premier League data with proper cleaning"""
    
    data_path = Path('data/raw/rawPLData.csv')
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return None
    
    print("=" * 70)
    print("PREMIER LEAGUE DATA LOADER WITH CLEANING")
    print("=" * 70)
    
    # Load the data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Identify duplicate columns
    print("\n1. Checking for duplicate columns...")
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    
    if duplicate_columns:
        print(f"   Found duplicate columns: {duplicate_columns}")
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"   Removed duplicates. New shape: {df.shape}")
    else:
        print("   No duplicate columns found")
    
    # Clean column names
    print("\n2. Cleaning column names...")
    original_columns = df.columns.tolist()
    df.columns = [col.strip().replace(' ', '_').replace('%', 'pct') for col in df.columns]
    print(f"   Example rename: '{original_columns[0]}' -> '{df.columns[0]}'")
    
    # Basic data cleaning
    print("\n3. Performing basic data cleaning...")
    
    # Convert percentage columns to numeric (0-1 range)
    pct_columns = [col for col in df.columns if 'pct' in col.lower() or 'accuracy' in col.lower()]
    for col in pct_columns:
        if col in df.columns:
            # Remove % signs and convert to float
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
    
    # Handle empty strings and convert to NaN
    df.replace(['', ' ', '-', '--'], np.nan, inplace=True)
    
    # Create position categories
    print("\n4. Creating position categories...")
    if 'Position' in df.columns:
        df['Position_Category'] = df['Position'].apply(categorize_position)
        position_counts = df['Position_Category'].value_counts()
        print(f"   Position distribution:")
        for pos, count in position_counts.items():
            print(f"     {pos}: {count} players")
    
    # Create some derived features
    print("\n5. Creating derived features...")
    
    # Goals per appearance
    if 'Goals' in df.columns and 'Appearances' in df.columns:
        df['Goals_per_Appearance'] = df['Goals'] / df['Appearances'].replace(0, np.nan)
    
    # Assist rate
    if 'Assists' in df.columns and 'Appearances' in df.columns:
        df['Assists_per_Appearance'] = df['Assists'] / df['Appearances'].replace(0, np.nan)
    
    # Save cleaned data
    print("\n6. Saving cleaned data...")
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = output_dir / 'premier_league_cleaned.csv'
    df.to_csv(clean_path, index=False)
    print(f"   Cleaned data saved to: {clean_path}")
    
    return df

def categorize_position(position):
    """Categorize positions into broader groups"""
    if pd.isna(position):
        return 'Unknown'
    
    position = str(position).lower()
    
    if 'goalkeeper' in position:
        return 'Goalkeeper'
    elif 'defender' in position or 'back' in position:
        return 'Defender'
    elif 'midfielder' in position or 'midfield' in position:
        return 'Midfielder'
    elif 'forward' in position or 'striker' in position or 'winger' in position:
        return 'Forward'
    else:
        return 'Other'

def analyze_dataset(df):
    """Provide comprehensive analysis of the dataset"""
    
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total players: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Column categories
    print("\nColumn Categories:")
    basic_info = [col for col in df.columns if col in ['Name', 'Club', 'Position', 'Nationality', 'Age', 'Jersey_Number']]
    performance = [col for col in df.columns if any(x in col.lower() for x in ['goals', 'assists', 'shots', 'passes', 'tackles', 'saves'])]
    defensive = [col for col in df.columns if any(x in col.lower() for x in ['clearance', 'intercept', 'block', 'tackle', 'save'])]
    offensive = [col for col in df.columns if any(x in col.lower() for x in ['goal', 'shot', 'assist', 'chance', 'cross'])]
    
    print(f"  Basic Info: {len(basic_info)} columns")
    print(f"  Performance: {len(performance)} columns")
    print(f"  Defensive: {len(defensive)} columns")
    print(f"  Offensive: {len(offensive)} columns")
    
    # Show sample players by position
    print("\nSample Players by Position:")
    if 'Position_Category' in df.columns:
        for category in df['Position_Category'].unique():
            sample = df[df['Position_Category'] == category].head(1)
            if not sample.empty:
                player_name = sample.iloc[0]['Name'] if 'Name' in df.columns else 'Unknown'
                print(f"  {category}: {player_name}")
    
    # Missing value analysis
    print("\nMissing Value Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Show columns with most missing values
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Top 10 columns with most missing values:")
    print(missing_df.head(10).to_string())
    
    # Position-specific missing patterns
    print("\nPosition-Specific Missing Patterns (expected):")
    gk_cols = [col for col in df.columns if any(x in col.lower() for x in ['save', 'catch', 'punch', 'claim'])]
    if gk_cols:
        non_gk_missing = df[df['Position_Category'] != 'Goalkeeper'][gk_cols].isnull().mean().mean() * 100
        print(f"  Goalkeeper-specific columns: {len(gk_cols)} columns")
        print(f"  Missing for non-GKs: {non_gk_missing:.1f}% (expected)")
    
    # Basic statistics
    print("\nKey Statistics:")
    
    if 'Age' in df.columns:
        print(f"  Age range: {df['Age'].min():.0f} - {df['Age'].max():.0f} (avg: {df['Age'].mean():.1f})")
    
    if 'Goals' in df.columns:
        top_scorer = df.loc[df['Goals'].idxmax()] if df['Goals'].max() > 0 else None
        if top_scorer is not None:
            print(f"  Top scorer: {top_scorer['Name']} with {top_scorer['Goals']} goals")
    
    if 'Assists' in df.columns:
        top_assister = df.loc[df['Assists'].idxmax()] if df['Assists'].max() > 0 else None
        if top_assister is not None:
            print(f"  Top assister: {top_assister['Name']} with {top_assister['Assists']} assists")
    
    # Club distribution
    if 'Club' in df.columns:
        print(f"\nClub Distribution:")
        club_counts = df['Club'].value_counts()
        print(f"  Total clubs: {len(club_counts)}")
        print(f"  Largest squad: {club_counts.idxmax()} ({club_counts.max()} players)")
        print(f"  Smallest squad: {club_counts.idxmin()} ({club_counts.min()} players)")

def save_analysis_report(df, output_path='outputs/dataset_analysis.txt'):
    """Save detailed analysis to file"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("PREMIER LEAGUE DATASET ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"- Total players: {len(df)}\n")
        f.write(f"- Total features: {len(df.columns)}\n")
        f.write(f"- Data shape: {df.shape}\n\n")
        
        f.write("Column Summary:\n")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            non_null = df[col].notnull().sum()
            pct = (non_null / len(df) * 100)
            f.write(f"{i:3d}. {col:<30} {str(dtype):<10} {non_null:>4} ({pct:5.1f}%)\n")
        
        f.write("\nSample Data (first 5 rows):\n")
        f.write(df.head().to_string())
        f.write("\n\n")
        
        f.write("Position Distribution:\n")
        if 'Position_Category' in df.columns:
            pos_counts = df['Position_Category'].value_counts()
            for pos, count in pos_counts.items():
                f.write(f"  {pos}: {count} players\n")
    
    print(f"\nAnalysis report saved to: {output_path}")

if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data()
    
    if df is not None:
        # Analyze the dataset
        analyze_dataset(df)
        
        # Save analysis report
        save_analysis_report(df)
        
        print("\n" + "=" * 70)
        print("CLEANING COMPLETE")
        print("=" * 70)
        print("✓ Duplicate columns removed")
        print("✓ Column names standardized")
        print("✓ Position categories created")
        print("✓ Derived features added")
        print("✓ Cleaned data saved to data/processed/")
        print("✓ Analysis report generated")
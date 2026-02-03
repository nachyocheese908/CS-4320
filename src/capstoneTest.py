import pandas as pd
import numpy as np
import os
from pathlib import Path

# ===== CONFIGURATION =====
# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# File paths using project root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "SoccerDataFinal.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


CSV_PATH = RAW_DATA_PATH  # ✅ CORRECT - uses the full path
TARGET_COL = 'Vals'
SEED = 4320
EXCLUDES = [TARGET_COL, 'Wages','CrdY','CrdR','Player']

def split_indices(n: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """Deterministic split using a seeded permutation (same idea as lecture)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx



def main():
    df = pd.read_csv(CSV_PATH)

    identifiers = df[['Player']].copy()
    
    for idx, num in enumerate(df['Vals']):
        if num[-1] == 'm': #for million vals
            new = num[:-1]
            new_flt = float(new)
            true_val = int(new_flt * 1000000)
            df.at[idx, 'Vals'] = true_val
        else: #for k values like 400k
            new = num[:-1]
            true_val = int(new) * 1000
            df.at[idx, 'Vals'] = true_val

    for idx, num in enumerate(df['Min']):
        if ',' in num:
            new = num.replace(',','')
            true_val = int(new)
            df.at[idx, 'Min'] = true_val
        else:
            true_val = int(new)
            df.at[idx, 'Min'] = true_val
    df['Min'] = pd.to_numeric(df['Min'], errors='coerce')

    # Quick fix (add this during preprocessing):
    df['Pos'] = df['Pos'].str.strip()
        
    print(df)

    #split
    train_idx, val_idx, test_idx = split_indices(len(df), SEED)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    #store identifiers
    train_identifiers = identifiers.iloc[train_idx].copy()
    val_identifiers = identifiers.iloc[val_idx].copy()
    test_identifiers = identifiers.iloc[test_idx].copy()
    print(f"\nStored identifiers for {len(train_identifiers)} train, {len(val_identifiers)} val, {len(test_identifiers)} test players")

    #separate target
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val   = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test  = test_df[TARGET_COL].to_numpy(dtype=float)

    #choose features
    X_train = train_df.drop(columns=[c for c in EXCLUDES if c in train_df.columns])
    X_val   = val_df.drop(columns=[c for c in EXCLUDES if c in val_df.columns])
    X_test  = test_df.drop(columns=[c for c in EXCLUDES if c in test_df.columns])

    #Identify numeric vs categorical
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    scaling_params = {}
    
    for col in numeric_cols:
        # Compute statistics from TRAIN data only
        mean_val = X_train[col].mean()
        std_val = X_train[col].std()
        
        # Store for application to all sets
        scaling_params[col] = {
            'mean': mean_val,
            'std': std_val
        }
        
        print(f"   {col}:")
        print(f"     Mean = {mean_val:.2f}")
        print(f"     Std  = {std_val:.2f}")
        print(f"     Range before scaling: [{X_train[col].min():.2f}, {X_train[col].max():.2f}]")

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    for col in numeric_cols:
        if col in scaling_params:
            mean_val = scaling_params[col]['mean']
            std_val = scaling_params[col]['std']
            
            # Avoid division by zero (if std = 0, feature is constant)
            if std_val > 0:
                # Standardization: (x - mean) / std
                X_train_scaled[col] = (X_train[col] - mean_val) / std_val
                X_val_scaled[col] = (X_val[col] - mean_val) / std_val
                X_test_scaled[col] = (X_test[col] - mean_val) / std_val
                
                # Report scaling effect
                train_min_after = X_train_scaled[col].min()
                train_max_after = X_train_scaled[col].max()
                print(f"   {col}: Scaled range = [{train_min_after:.2f}, {train_max_after:.2f}]")
                
                # Check for extreme values (potential outliers)
                extreme_count = ((X_train_scaled[col].abs() > 3).sum() + 
                                 (X_val_scaled[col].abs() > 3).sum() + 
                                 (X_test_scaled[col].abs() > 3).sum())
                if extreme_count > 0:
                    print(f"     ⚠ Found {extreme_count} |z-score| > 3 (potential outliers)")
            else:
                print(f"   {col}: WARNING - Standard deviation is 0 (constant feature)")
                print(f"     Setting all values to 0 (mean-centered)")
                X_train_scaled[col] = 0
                X_val_scaled[col] = 0
                X_test_scaled[col] = 0

    X_train = X_train_scaled
    X_val = X_val_scaled
    X_test = X_test_scaled
    
    print("\n✓ Numeric feature scaling complete!")
    print("  All numeric features now have mean ≈ 0 and std ≈ 1 (on training data)")
    print("  Same transformation applied to validation and test sets")

    category_maps = {}
    
    for col in cat_cols:
        # Get unique categories from TRAIN only (sorted for deterministic order)
        # Exclude NaN since we already imputed them
        unique_cats = X_train[col].dropna().unique()
        sorted_cats = np.sort(unique_cats)  # Sort for consistent ordering
        
        category_maps[col] = {
            'categories': sorted_cats.tolist(),
            'category_to_idx': {cat: idx for idx, cat in enumerate(sorted_cats)}
        }
        
        print(f"   {col}: Found {len(sorted_cats)} categories")
        if len(sorted_cats) <= 10:  # Don't print too many
            print(f"     Categories: {sorted_cats.tolist()}")
        else:
            print(f"     First 5 categories: {sorted_cats[:5].tolist()}...")
    
    # --- APPLY ONE-HOT ENCODING TO ALL SETS ---
    print("\n2. Creating one-hot encoded columns...")
    print("   Using SAME category scheme for all sets (fitted on train only!)")
    
    def one_hot_encode(df, category_maps, prefix_sep='_'):
        """Manual one-hot encoding using fitted category maps"""
        encoded_dfs = []
        
        # Keep original numeric columns (already scaled)
        numeric_df = df[numeric_cols].copy()
        encoded_dfs.append(numeric_df)
        
        # One-hot encode each categorical column
        for col in cat_cols:
            if col in category_maps:
                categories = category_maps[col]['categories']
                cat_to_idx = category_maps[col]['category_to_idx']
                
                # Create one-hot columns for this feature
                for category in categories:
                    col_name = f"{col}{prefix_sep}{category}"
                    
                    # Check if value equals this category
                    # Handle unseen categories (not in our map) -> all zeros
                    is_this_category = df[col].apply(
                        lambda x: 1 if x == category else 0
                    )
                    encoded_dfs.append(pd.DataFrame({col_name: is_this_category}))
                
                # Check for unseen categories in this dataset
                unique_vals = df[col].unique()
                unseen_cats = [val for val in unique_vals if val not in cat_to_idx]
                
                if len(unseen_cats) > 0:
                    print(f"     ⚠ {col}: Found {len(unseen_cats)} unseen categories in this set")
                    print(f"        Unseen: {unseen_cats}")
                    print(f"        These will map to all-zeros (no hot encoding)")
        
        # Combine all columns
        result = pd.concat(encoded_dfs, axis=1)
        return result
    
    # Apply encoding to all sets
    X_train_encoded = one_hot_encode(X_train, category_maps)
    X_val_encoded = one_hot_encode(X_val, category_maps)
    X_test_encoded = one_hot_encode(X_test, category_maps)

    X_train = X_train_encoded
    X_val = X_val_encoded
    X_test = X_test_encoded

    #reattach identifiers
    train_final = pd.concat([train_identifiers.reset_index(drop=True), 
                            X_train.reset_index(drop=True), 
                            pd.Series(y_train, name=TARGET_COL)], axis=1)
    
    val_final = pd.concat([val_identifiers.reset_index(drop=True), 
                          X_val.reset_index(drop=True), 
                          pd.Series(y_val, name=TARGET_COL)], axis=1)
    
    test_final = pd.concat([test_identifiers.reset_index(drop=True), 
                           X_test.reset_index(drop=True), 
                           pd.Series(y_test, name=TARGET_COL)], axis=1)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_identifiers': train_identifiers,
        'val_identifiers': val_identifiers,
        'test_identifiers': test_identifiers,
        'train_full': train_final,
        'val_full': val_final,
        'test_full': test_final,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols,
        'category_maps': category_maps,
        'scaling_params': scaling_params
    }

if __name__ == '__main__':
    main()
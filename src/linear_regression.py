import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
import warnings
warnings.filterwarnings('ignore')

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

df = pd.read_csv(CSV_PATH)

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
df['Vals'] = pd.to_numeric(df['Vals'], errors='coerce')

SCALE_FACTOR = 1_000_000
df[TARGET_COL] = df[TARGET_COL] / SCALE_FACTOR


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

identifiers = df[['Player']].copy()

y = df[TARGET_COL].to_numpy(dtype=np.float64)

X_df = df.drop(columns=[TARGET_COL,'Wages','CrdY','CrdR','Player'])

print(f"Dataset shape: {df.shape}")
print(f"Features shape: {X_df.shape}")
print(f"Target shape: {y.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("===After Column Drops===")
print(f"Dataset shape: {X_df.shape}")
print(f"Features shape: {X_df.shape}")
print(f"Target shape: {y.shape}")
print("\nFirst few rows:")
print(X_df.head())
print("\nData types:")
print(X_df.dtypes)
print("\nMissing values:")
print(X_df.isnull().sum())

test_size = 0.15
val_size  = 0.15
random_state = SEED

# 1) split test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_df, y,
    test_size=test_size,
    random_state=random_state,
)

# 2) compute val fraction relative to trainval
val_fraction = val_size / (1.0 - test_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_fraction,
    random_state=random_state,
)

print("sizes:", len(X_train), len(X_val), len(X_test))

numeric_features = [c for c in X_df.columns if is_numeric_dtype(X_df[c])]
categorical_features = [c for c in X_df.columns if c not in numeric_features]

print("numeric:", numeric_features)
print("categorical:", categorical_features)

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ],
    remainder="drop",
    sparse_threshold=0  # ← ADD THIS LINE
)

X_train_p = pre.fit_transform(X_train)
X_val_p   = pre.transform(X_val)
#X_test_p  = pre.transform(X_test)

# convert to NumPy float arrays
X_train_p = np.asarray(X_train_p, dtype=np.float64)
X_val_p   = np.asarray(X_val_p, dtype=np.float64)
#X_test_p  = np.asarray(X_test_p, dtype=np.float64)

print("X_train_p shape:", X_train_p.shape)

def add_bias_column(X: np.ndarray):
    n_samples = X.shape[0]
    ones = np.ones((n_samples, 1), dtype=X.dtype)
    return np.hstack([ones, X])

def predict(X: np.ndarray, w: np.ndarray):
    X_b = add_bias_column(X)
    return X_b @ w

def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    y_hat = predict(X, w)
    n_samples = len(y)
    return np.sum((y_hat - y) ** 2) / n_samples

def mse_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    X_b = add_bias_column(X)
    n_samples = len(y)
    y_hat = X_b @ w
    error = y_hat - y
    gradient = (2 / n_samples) * (X_b.T @ error)
    return gradient


n_features = X_train_p.shape[1]
n_weights = n_features + 1  # +1 for bias

# Initialize weights with small random values
np.random.seed(random_state)
w = np.random.randn(n_weights) * 0.01

lr = 0.01
epochs = 3000

train_losses = []
val_losses = []

for epoch in range(epochs):
    # 1) compute gradient on training
    grad = mse_gradient(X_train_p, y_train, w)

    # 2) update
    w = w - lr * grad

    # 3) track losses (OPTIONAL but recommended)
    train_losses.append(mse_loss(X_train_p, y_train, w))
    val_losses.append(mse_loss(X_val_p, y_val, w))

    # 4) occasionally print progress
    if (epoch+1) % 100 == 0: print(epoch+1, f'Train: {train_losses[-1]}', f'Val: {val_losses[-1]}')

plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.close()

print("\n" + "="*60)
print("Final evaluation on test set (isolated until now)...")

# Transform test set using the preprocessor fitted on training data
X_test_processed = pre.transform(X_test)
X_test_p = np.asarray(X_test_processed, dtype=np.float64)

# Calculate test metrics
test_mse = mse_loss(X_test_p, y_test, w)
y_test_pred = predict(X_test_p, w)

SCALE_FACTOR = 1_000_000  # Make sure this matches what you used above

# Metrics in scaled units (millions)
test_mse_scaled = test_mse  # Already in squared millions
test_rmse_scaled = np.sqrt(test_mse_scaled)  # In millions
test_mae_scaled = np.mean(np.abs(y_test_pred - y_test))  # In millions

# Metrics in original units (dollars)
test_mse_original = test_mse_scaled * (SCALE_FACTOR ** 2)
test_rmse_original = test_rmse_scaled * SCALE_FACTOR
test_mae_original = test_mae_scaled * SCALE_FACTOR

# ============ CONTINUE WITH YOUR EXISTING CODE ============
# Calculate R-squared (same regardless of scaling)
y_test_mean = np.mean(y_test)
ss_res = np.sum((y_test - y_test_pred) ** 2)
ss_tot = np.sum((y_test - y_test_mean) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "-" * 50)
print("TEST SET PERFORMANCE METRICS")
print("-" * 50)
print("\n--- In Millions (Scaled) ---")
print(f"MSE:  {test_mse_scaled:.4f} (million$²)")
print(f"RMSE: {test_rmse_scaled:.4f} million$")
print(f"MAE:  {test_mae_scaled:.4f} million$")

print("\n--- In Original Dollars ---")
print(f"MSE:  {test_mse_original:.0f}")
print(f"RMSE: ${test_rmse_original:,.0f}")
print(f"MAE:  ${test_mae_original:,.0f}")

print(f"\nR²: {r2:.4f}")
print("-" * 50)

# Compare with baseline (predicting mean)
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mse = np.mean((baseline_pred - y_test) ** 2)
improvement_over_baseline = (baseline_mse - test_mse) / baseline_mse * 100
print(f"\nBaseline (predicting training mean): MSE = {baseline_mse:.4f} (million$²)")
print(f"Our model improves by: {improvement_over_baseline:.1f}%")

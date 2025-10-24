"""
Data Preparation Pipeline
Loads gold/crypto data, creates synthetic fear/greed index, engineers features, splits data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def load_data():
    """Load raw gold_crypto.csv dataset"""
    print("Loading datasets...")

    try:
        df = pd.read_csv('datasets/gold_crypto.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Error: datasets/gold_crypto.csv not found. "
            "Please ensure the file exists in the datasets/ directory."
        )
    except Exception as e:
        raise Exception(f"Error reading gold_crypto.csv: {str(e)}")

    # Validate required columns
    required_cols = ['Date', 'Bitcoin (USD)', 'Gold (USD per oz)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate data is not empty
    if len(df) == 0:
        raise ValueError("Error: Dataset is empty")

    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(f"Error parsing dates: {str(e)}")

    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


def create_synthetic_index(df):
    """
    Create synthetic fear/greed index (0-100)
    BTC outperformance = Greed, Gold outperformance = Fear
    """
    print("\nCreating synthetic fear/greed index...")

    df = df.copy()

    # Extract BTC and Gold prices
    try:
        btc = df['Bitcoin (USD)']
        gold = df['Gold (USD per oz)']
    except KeyError as e:
        raise KeyError(f"Required price column not found: {str(e)}")

    # Validate price data
    if btc.isna().all() or gold.isna().all():
        raise ValueError("Error: All price values are NaN")

    if (btc <= 0).any() or (gold <= 0).any():
        print("⚠ Warning: Found non-positive prices. This may cause issues.")

    # Calculate 7-day returns
    btc_return_7d = btc.pct_change(7)
    gold_return_7d = gold.pct_change(7)

    # Compute momentum spread (BTC outperformance)
    momentum_spread = btc_return_7d - gold_return_7d

    # Normalize to 0-100 using rolling percentile rank (252-day window = 1 year)
    # Use min_periods=30 to start calculating after first month
    fear_greed_index = momentum_spread.rolling(window=252, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 1 else np.nan,
        raw=False
    )

    df['fear_greed'] = fear_greed_index

    # Print statistics
    print(f"Fear/Greed Index Statistics:")
    print(df['fear_greed'].describe())
    print(f"Non-null values: {df['fear_greed'].notna().sum()} / {len(df)}")

    return df


def engineer_features(df):
    """
    Create 13 features from BTC/Gold prices + fear/greed index
    NO DATA LEAKAGE: All features use only past data
    """
    print("\nEngineering features...")

    df = df.copy()

    btc = df['Bitcoin (USD)']
    gold = df['Gold (USD per oz)']
    fear_greed = df['fear_greed']

    # ===== AUTOREGRESSIVE FEATURES (3) =====
    df['fear_greed_lag1'] = fear_greed.shift(1)
    df['fear_greed_lag3'] = fear_greed.shift(3)
    df['fear_greed_lag7'] = fear_greed.shift(7)

    # ===== PRICE MOMENTUM FEATURES (4) =====
    df['btc_return_1d'] = btc.pct_change(1)
    df['btc_return_7d'] = btc.pct_change(7)
    df['gold_return_1d'] = gold.pct_change(1)
    df['gold_return_7d'] = gold.pct_change(7)

    # ===== CROSS-ASSET FEATURES (2) =====
    # Use lagged prices to avoid leakage
    btc_lag1 = btc.shift(1)
    gold_lag1 = gold.shift(1)
    df['btc_gold_ratio'] = btc_lag1 / gold_lag1
    df['ratio_change_7d'] = df['btc_gold_ratio'].pct_change(7)

    # ===== VOLATILITY FEATURES (1) =====
    # 7-day rolling std of BTC returns
    df['btc_vol_7d'] = df['btc_return_1d'].rolling(window=7, min_periods=2).std()

    # ===== TREND FEATURES (2) =====
    # Price / 7-day MA ratio (use shifted MA to avoid leakage)
    btc_ma7 = btc.shift(1).rolling(window=7, min_periods=1).mean()
    gold_ma7 = gold.shift(1).rolling(window=7, min_periods=1).mean()
    df['btc_ma7_ratio'] = btc / btc_ma7
    df['gold_ma7_ratio'] = gold / gold_ma7

    # ===== TEMPORAL FEATURES (1) =====
    df['day_of_week'] = df['Date'].dt.dayofweek

    # ===== TARGET VARIABLE =====
    # Predict NEXT day's fear/greed index
    df['fear_greed_next'] = fear_greed.shift(-1)

    # Drop rows with NaN in target or features
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)

    print(f"Dropped {initial_len - final_len} rows with NaN values")
    print(f"Final dataset: {final_len} rows")
    print(f"Features: 13 + 1 target + Date column")

    # Verify no data leakage: all feature columns should exist
    feature_cols = [
        'fear_greed_lag1', 'fear_greed_lag3', 'fear_greed_lag7',
        'btc_return_1d', 'btc_return_7d', 'gold_return_1d', 'gold_return_7d',
        'btc_gold_ratio', 'ratio_change_7d',
        'btc_vol_7d',
        'btc_ma7_ratio', 'gold_ma7_ratio',
        'day_of_week'
    ]

    for col in feature_cols:
        assert col in df.columns, f"Missing feature: {col}"

    print(f"✓ All 13 features created successfully")

    return df


def split_chronologically(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data chronologically into train/val/test sets
    """
    print("\nSplitting data chronologically...")

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size:train_size + val_size].copy()
    test = df.iloc[train_size + val_size:].copy()

    print(f"Train: {len(train)} rows ({train['Date'].min()} to {train['Date'].max()})")
    print(f"Val:   {len(val)} rows ({val['Date'].min()} to {val['Date'].max()})")
    print(f"Test:  {len(test)} rows ({test['Date'].min()} to {test['Date'].max()})")

    # Ensure splits are non-overlapping
    assert train['Date'].max() < val['Date'].min(), "Train-Val overlap detected"
    assert val['Date'].max() < test['Date'].min(), "Val-Test overlap detected"

    print("✓ Chronological split verified (no overlap)")

    return train, val, test


def main():
    """Main data preparation pipeline"""
    print("=" * 70)
    print("Data Preparation Pipeline - Synthetic Fear/Greed Index Predictor")
    print("=" * 70)

    try:
        # Step 1: Load data
        df = load_data()

        # Step 2: Create synthetic index
        df = create_synthetic_index(df)

        # Step 3: Engineer features
        df = engineer_features(df)

        # Validate we have enough data
        if len(df) < 100:
            raise ValueError(f"Error: Only {len(df)} samples remaining after feature engineering. Need at least 100.")

        # Step 4: Split chronologically
        train, val, test = split_chronologically(df)

        # Validate splits
        if len(train) < 50 or len(val) < 10 or len(test) < 10:
            raise ValueError("Error: Insufficient data in one or more splits")

        # Step 5: Save processed datasets
        print("\nSaving processed datasets...")
        Path('datasets').mkdir(exist_ok=True)

        try:
            train.to_csv('datasets/train.csv', index=False)
            val.to_csv('datasets/val.csv', index=False)
            test.to_csv('datasets/test.csv', index=False)
        except Exception as e:
            raise IOError(f"Error saving datasets: {str(e)}")

        print(f"✓ Saved datasets/train.csv ({len(train)} rows)")
        print(f"✓ Saved datasets/val.csv ({len(val)} rows)")
        print(f"✓ Saved datasets/test.csv ({len(test)} rows)")

        # Summary
        print("\n" + "=" * 70)
        print("Pipeline Complete!")
        print("=" * 70)
        print("\nData Summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  Train:         {len(train)} ({len(train)/len(df)*100:.1f}%)")
        print(f"  Validation:    {len(val)} ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test:          {len(test)} ({len(test)/len(df)*100:.1f}%)")
        print(f"\nFeatures: 13")
        print(f"Target:   fear_greed_next (0-100 scale)")
        print("\nNext step: Run 'python src/train_model.py' to train the model")

    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed!")
        print(f"   {str(e)}")
        print("\nPlease check the error message above and fix the issue.")
        raise


if __name__ == '__main__':
    main()

"""
Model Training Pipeline
Trains LightGBM model to predict synthetic fear/greed index with validation-based early stopping.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)


def load_train_val():
    """Load training and validation datasets"""
    print("Loading datasets...")

    try:
        train = pd.read_csv('datasets/train.csv')
        val = pd.read_csv('datasets/val.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error: Dataset file not found: {str(e)}\n"
            "Please run 'python src/prepare_data.py' first to generate datasets."
        )
    except Exception as e:
        raise Exception(f"Error reading datasets: {str(e)}")

    # Validate datasets are not empty
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Error: Train or validation dataset is empty")

    print(f"Train: {len(train)} rows")
    print(f"Val:   {len(val)} rows")

    return train, val


def prepare_features(df):
    """
    Separate features from target
    Returns: X (features), y (target)
    """
    # Define feature columns (exclude Date and target)
    feature_cols = [
        'fear_greed_lag1', 'fear_greed_lag3', 'fear_greed_lag7',
        'btc_return_1d', 'btc_return_7d', 'gold_return_1d', 'gold_return_7d',
        'btc_gold_ratio', 'ratio_change_7d',
        'btc_vol_7d',
        'btc_ma7_ratio', 'gold_ma7_ratio',
        'day_of_week'
    ]

    # Verify all features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    y = df['fear_greed_next'].copy()

    return X, y


def train_lgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM model with early stopping
    """
    print("\nTraining LightGBM model...")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Validate sufficient training data
    if len(X_train) < 50:
        raise ValueError(f"Error: Insufficient training data ({len(X_train)} samples). Need at least 50.")

    # Check for NaN or infinite values
    if X_train.isna().any().any() or X_val.isna().any().any():
        raise ValueError("Error: NaN values found in features")

    if np.isinf(X_train.values).any() or np.isinf(X_val.values).any():
        raise ValueError("Error: Infinite values found in features")

    # Initialize model with conservative hyperparameters
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Train with early stopping
    callbacks = [
        early_stopping(stopping_rounds=50, verbose=True),
        log_evaluation(period=100)
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=callbacks
    )

    print(f"\nTraining stopped at iteration: {model.best_iteration_}")
    print(f"Best validation RMSE: {model.best_score_['valid_0']['rmse']:.4f}")

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    print("\n" + "=" * 70)
    print("Validation Metrics:")
    print("=" * 70)
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")

    # Check if model learned anything useful
    if r2 < 0:
        print("\n⚠ WARNING: R² is negative (model worse than mean baseline)")
    elif r2 < 0.1:
        print("\n⚠ WARNING: R² is very low (weak predictive power)")
    else:
        print(f"\n✓ Model trained successfully (R² = {r2:.4f})")

    return model


def save_model(model, filepath):
    """Save trained model to disk"""
    try:
        Path('models').mkdir(exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        print(f"\n✓ Model saved to {filepath}")
    except Exception as e:
        raise IOError(f"Error saving model: {str(e)}")


def display_feature_importance(model, feature_cols):
    """Display top 10 most important features"""
    print("\n" + "=" * 70)
    print("Top 10 Most Important Features:")
    print("=" * 70)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:25s} {row['importance']:10.2f}")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("Model Training Pipeline - Fear/Greed Index Predictor")
    print("=" * 70)

    try:
        # Step 1: Load data
        train, val = load_train_val()

        # Step 2: Prepare features
        X_train, y_train = prepare_features(train)
        X_val, y_val = prepare_features(val)

        # Step 3: Train model
        model = train_lgbm(X_train, y_train, X_val, y_val)

        # Step 4: Save model
        model_path = 'models/feargreed_model.pkl'
        save_model(model, model_path)

        # Step 5: Display feature importance
        display_feature_importance(model, X_train.columns.tolist())

        # Summary
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nModel saved to: {model_path}")
        print("\nNext step: Run 'python src/evaluate_model.py' to evaluate on test set")

    except Exception as e:
        print(f"\n❌ ERROR: Training failed!")
        print(f"   {str(e)}")
        print("\nPlease check the error message above and fix the issue.")
        raise


if __name__ == '__main__':
    main()

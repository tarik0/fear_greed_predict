"""
GRU Model Evaluation Pipeline
Loads trained GRU model and evaluates performance on test set.
Generates metrics and visualization plots.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FearGreedGRU(nn.Module):
    """
    GRU-based neural network for Fear/Greed Index prediction.
    Must match the architecture from train_model.py.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Apply batch normalization to each timestep
        x = x.reshape(-1, features)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, features)

        # GRU forward pass
        _, hidden = self.gru(x)

        # Take last layer's hidden state
        x = hidden[-1]

        # Output layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x.squeeze(-1)


def load_model(filepath='models/feargreed_model.pt'):
    """Load trained GRU model and scaler from disk"""
    print(f"Loading model from {filepath}...")

    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Model file not found at {filepath}\n"
            "Please run 'python src/train_model.py' first to train the model."
        )
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    # Reconstruct model
    config = checkpoint['model_config']
    model = FearGreedGRU(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']

    # Get configuration
    seq_len = config['seq_len']
    feature_cols = checkpoint['feature_cols']

    print("✓ Model loaded successfully")
    print(f"  Architecture: GRU (hidden={config['hidden_dim']}, layers={config['num_layers']})")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features: {len(feature_cols)}")

    return model, scaler, seq_len, feature_cols


def load_test_data():
    """Load test dataset"""
    print("\nLoading test dataset...")

    try:
        test = pd.read_csv('datasets/test.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Error: datasets/test.csv not found.\n"
            "Please run 'python src/prepare_data.py' first to generate datasets."
        )

    if len(test) == 0:
        raise ValueError("Error: Test dataset is empty")

    print(f"✓ Test: {len(test)} rows")
    print(f"  Date range: {test['Date'].min()} to {test['Date'].max()}")

    return test


def create_sequences(features, seq_len):
    """Create sequences for prediction"""
    sequences = []
    for i in range(len(features) - seq_len):
        sequences.append(features[i:i + seq_len])
    return np.array(sequences, dtype=np.float32)


def prepare_test_data(test_df, scaler, seq_len, feature_cols):
    """
    Prepare test data for GRU inference.
    - Extract features and scale them
    - Create sequences
    - Get aligned targets and dates
    """
    print(f"\nPreparing test data with sequence length {seq_len}...")

    # Extract features and targets
    X_test = test_df[feature_cols].values
    y_test = test_df['fear_greed_next'].values
    dates = pd.to_datetime(test_df['Date'])

    # Scale features using the saved scaler
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_sequences = create_sequences(X_test_scaled, seq_len)

    # Align targets and dates (skip first seq_len samples)
    y_aligned = y_test[seq_len:]
    dates_aligned = dates.iloc[seq_len:].reset_index(drop=True)

    print(f"  Test sequences: {len(X_sequences)}")

    return X_sequences, y_aligned, dates_aligned


def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculate percentage of correct directional predictions (up/down)
    """
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)

    correct_direction = np.sign(y_true_diff) == np.sign(y_pred_diff)
    direction_accuracy = np.mean(correct_direction) * 100

    return direction_accuracy


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics
    """
    print("\n" + "=" * 70)
    print("Evaluating Model on Test Set")
    print("=" * 70)

    # Convert to tensor and make predictions
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    # Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MAPE (handle near-zero values)
    mask = np.abs(y_test) > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = float('nan')

    # SMAPE (more stable for near-zero values)
    smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)) * 100

    # Calculate direction accuracy
    direction_acc = calculate_direction_accuracy(y_test, y_pred)

    # Print metrics
    print("\nTest Set Metrics:")
    print(f"  RMSE:               {rmse:.4f}")
    print(f"  MAE:                {mae:.4f}")
    print(f"  R² Score:           {r2:.4f}")
    print(f"  MAPE:               {mape:.2f}% (unstable with near-zero values)")
    print(f"  SMAPE:              {smape:.2f}% (stable alternative)")
    print(f"  Direction Accuracy: {direction_acc:.2f}%")

    # Interpretation
    print("\nInterpretation:")
    if direction_acc > 55:
        print("  ✓ Direction accuracy > 55% (model has predictive signal)")
    elif direction_acc > 50:
        print("  ≈ Direction accuracy ~50% (weak signal)")
    else:
        print("  ✗ Direction accuracy < 50% (worse than random)")

    if smape < 30:
        print(f"  ✓ SMAPE < 30% (acceptable error for financial index)")
    else:
        print(f"  ⚠ SMAPE ≥ 30% (high error)")

    if r2 > 0.5:
        print(f"  ✓ R² > 0.5 (model explains significant variance)")
    elif r2 > 0:
        print(f"  ≈ R² > 0 (better than mean baseline)")
    else:
        print(f"  ✗ R² < 0 (worse than mean baseline)")

    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape': smape,
        'direction_acc': direction_acc
    }


def plot_predictions(dates, y_true, y_pred, filename='predictions.png'):
    """
    Create prediction visualization (last 90 days)
    """
    print(f"\nGenerating visualization...")

    try:
        # Plot last 90 days for better visibility
        n_days = min(90, len(dates))
        dates_plot = dates[-n_days:]
        y_true_plot = y_true[-n_days:]
        y_pred_plot = y_pred[-n_days:]

        plt.figure(figsize=(14, 6))

        # Plot actual vs predicted
        plt.plot(dates_plot, y_true_plot, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
        plt.plot(dates_plot, y_pred_plot, label='Predicted (GRU)', color='#A23B72', linewidth=2, alpha=0.8, linestyle='--')

        # Add fear/greed zones
        plt.axhspan(0, 20, alpha=0.1, color='red', label='Extreme Fear')
        plt.axhspan(20, 40, alpha=0.05, color='red')
        plt.axhspan(40, 60, alpha=0.05, color='gray', label='Neutral')
        plt.axhspan(60, 80, alpha=0.05, color='green')
        plt.axhspan(80, 100, alpha=0.1, color='green', label='Extreme Greed')

        plt.title('Fear/Greed Index - Actual vs GRU Predicted (Last 90 Days)', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Fear/Greed Index (0-100)', fontsize=12)
        plt.ylim(-5, 105)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        Path('outputs').mkdir(exist_ok=True)
        output_path = f'outputs/{filename}'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Plot saved to {output_path}")

    except Exception as e:
        raise IOError(f"Error generating plot: {str(e)}")


def save_evaluation_report(results, test_date_range, n_samples):
    """
    Save evaluation metrics to text file
    """
    print("\nSaving evaluation report...")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Fear/Greed Index Predictor - GRU Model Test Set Evaluation")
    report_lines.append("=" * 70)
    report_lines.append(f"\nModel: GRU (Gated Recurrent Unit)")
    report_lines.append(f"Test Period: {test_date_range}")
    report_lines.append(f"Test Samples: {n_samples}")
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Metrics:")
    report_lines.append("=" * 70)
    report_lines.append(f"  RMSE:               {results['rmse']:.4f}")
    report_lines.append(f"  MAE:                {results['mae']:.4f}")
    report_lines.append(f"  R² Score:           {results['r2']:.4f}")
    report_lines.append(f"  MAPE:               {results['mape']:.2f}% (unstable with near-zero values)")
    report_lines.append(f"  SMAPE:              {results['smape']:.2f}% (stable alternative)")
    report_lines.append(f"  Direction Accuracy: {results['direction_acc']:.2f}%")
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Metric Definitions:")
    report_lines.append("=" * 70)
    report_lines.append("  RMSE:  Root Mean Squared Error (lower is better)")
    report_lines.append("  MAE:   Mean Absolute Error (lower is better)")
    report_lines.append("  MAPE:  Mean Absolute Percentage Error")
    report_lines.append("         WARNING: Unstable when y_true approaches zero!")
    report_lines.append("  SMAPE: Symmetric MAPE - more stable for near-zero values")
    report_lines.append("  R²:    Coefficient of Determination (1.0 = perfect fit)")
    report_lines.append("  Direction Accuracy: % of correct up/down predictions")
    report_lines.append("                     (>50% = better than random)")
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Success Criteria:")
    report_lines.append("=" * 70)

    # Check success criteria
    success_checks = []
    if results['direction_acc'] > 50:
        success_checks.append("✓ Direction accuracy > 50% (beats random)")
    else:
        success_checks.append("✗ Direction accuracy ≤ 50% (no signal)")

    if results['smape'] < 30:
        success_checks.append("✓ SMAPE < 30% (acceptable error)")
    else:
        success_checks.append("✗ SMAPE ≥ 30% (high error)")

    if results['r2'] > 0.5:
        success_checks.append("✓ R² > 0.5 (strong explanatory power)")
    elif results['r2'] > 0:
        success_checks.append("≈ R² > 0 (better than mean baseline)")
    else:
        success_checks.append("✗ R² ≤ 0 (worse than mean baseline)")

    for check in success_checks:
        report_lines.append(f"  {check}")

    report_text = '\n'.join(report_lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    output_path = 'outputs/evaluation.txt'
    with open(output_path, 'w',encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to {output_path}")


def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("GRU Model Evaluation Pipeline - Fear/Greed Index Predictor")
    print("=" * 70)

    try:
        # Step 1: Load model
        model, scaler, seq_len, feature_cols = load_model('models/feargreed_model.pt')

        # Step 2: Load test data
        test = load_test_data()

        # Step 3: Prepare test data
        X_test, y_test, dates = prepare_test_data(test, scaler, seq_len, feature_cols)

        # Step 4: Evaluate model
        results = evaluate_model(model, X_test, y_test)

        # Step 5: Plot predictions
        plot_predictions(dates, results['y_true'], results['y_pred'])

        # Step 6: Save evaluation report
        test_date_range = f"{dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}"
        save_evaluation_report(results, test_date_range, len(y_test))

        # Summary
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        print("\nOutput files:")
        print("  - outputs/evaluation.txt (metrics summary)")
        print("  - outputs/predictions.png (visualization)")
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR: Evaluation failed!")
        print(f"   {str(e)}")
        print("\nPlease check the error message above and fix the issue.")
        raise


if __name__ == '__main__':
    main()

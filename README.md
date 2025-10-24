# Gold/Crypto Fear-Greed Index Predictor

Machine learning model to predict next-day synthetic Fear & Greed Index (0-100) based on BTC/Gold price dynamics using LightGBM.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Step 1: Prepare data and create synthetic fear/greed index
python src/prepare_data.py

# Step 2: Train LightGBM model
python src/train_model.py

# Step 3: Evaluate on test set
python src/evaluate_model.py
```

## Project Structure
```
gold_crypto_predict/
├── datasets/              # Raw and processed data
│   ├── gold_crypto.csv   # Raw input data
│   ├── train.csv         # Training set (generated)
│   ├── val.csv           # Validation set (generated)
│   └── test.csv          # Test set (generated)
├── models/               # Trained model
│   └── feargreed_model.pkl
├── outputs/              # Evaluation results and plots
│   ├── evaluation.txt    # Metrics summary
│   └── predictions.png   # Visualization
├── src/                  # Source code
│   ├── prepare_data.py   # Data preprocessing & feature engineering
│   ├── train_model.py    # Model training with early stopping
│   └── evaluate_model.py # Model evaluation & visualization
├── CLAUDE.md             # Detailed technical documentation
└── requirements.txt      # Python dependencies
```

## What It Does

1. **Creates Synthetic Index** - Fear/Greed index from BTC vs Gold momentum spread (7-day returns)
2. **Feature Engineering** - 13 features including lag values, momentum, volatility, cross-asset ratios
3. **Model Training** - Single LightGBM regressor with validation-based early stopping
4. **Evaluation** - Tests on chronologically split data with multiple metrics

## Expected Output

- **Processed CSVs**: `datasets/train.csv`, `val.csv`, `test.csv` (~2669/572/573 rows)
- **Trained Model**: `models/feargreed_model.pkl`
- **Metrics Report**: `outputs/evaluation.txt`
- **Visualization**: `outputs/predictions.png` (last 90 days)

## Actual Model Performance (Test Set)

```
RMSE:                15.41  (Target: 10-20) ✓
MAE:                 12.17  (Target: 8-15) ✓
R² Score:            0.72   (Target: 0.2-0.5) ✓✓
MAPE:                79.33% (Unstable due to near-zero values)
SMAPE:               37.56% (Stable alternative)
Direction Accuracy:  50.17% (Weak signal)
```

**Note on MAPE:** The high MAPE is due to the metric's instability when target values approach zero (test set contains 7 values < 1). The model's R², RMSE, and MAE indicate strong performance. SMAPE is a more reliable percentage-based metric for this use case.

### Prediction Visualization

![Fear/Greed Index Predictions - Last 90 Days](outputs/predictions.png)

*The model captures the overall trend of the fear/greed index, showing strong correlation with actual values (R²=0.72). Predictions are shown for the last 90 days of the test set (Dec 2023 - Jul 2025).*

## Index Interpretation

- **0-20**: Extreme Fear (Gold strongly outperforming BTC)
- **20-40**: Fear (Gold outperforming)
- **40-60**: Neutral (Balanced)
- **60-80**: Greed (BTC outperforming)
- **80-100**: Extreme Greed (BTC strongly outperforming)

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Synthetic index design rationale
- Feature engineering (no data leakage)
- Model architecture and hyperparameters
- Best practices and limitations
- Troubleshooting guide

## Disclaimer

**For educational purposes only.** This synthetic fear/greed index is a simplified proxy based solely on BTC/Gold dynamics and should NOT be used for actual trading decisions. Past performance does not guarantee future results.

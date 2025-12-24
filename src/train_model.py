"""
GRU Model Training Pipeline
Trains a Gated Recurrent Unit (GRU) neural network to predict synthetic fear/greed index.
Replaces LightGBM with deep learning for better temporal pattern capture.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
SEQ_LEN = 7  # Sequence length (days of history)
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
GRAD_CLIP = 1.0

# Features to use for sequences (exclude lag features - GRU learns temporal patterns)
SEQUENCE_FEATURES = [
    'fear_greed',
    'btc_return_1d', 'btc_return_7d',
    'gold_return_1d', 'gold_return_7d',
    'btc_gold_ratio', 'ratio_change_7d',
    'btc_vol_7d',
    'btc_ma7_ratio', 'gold_ma7_ratio'
]

# Device configuration (force CPU for compatibility with older GPUs)
DEVICE = torch.device('cpu')


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for creating sequences from time-series data.
    Each sample is a sequence of `seq_len` timesteps used to predict the next value.
    """
    def __init__(self, features, targets, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.X, self.y = self._create_sequences(features, targets)

    def _create_sequences(self, features, targets):
        """Create sliding window sequences"""
        X, y = [], []
        for i in range(len(features) - self.seq_len):
            X.append(features[i:i + self.seq_len])
            y.append(targets[i + self.seq_len])  # Predict next day
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class FearGreedGRU(nn.Module):
    """
    GRU-based neural network for Fear/Greed Index prediction.

    Architecture:
    - BatchNorm on input features
    - Single-layer GRU
    - Dropout + Dense layers for output
    """
    def __init__(self, input_dim=len(SEQUENCE_FEATURES), hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
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
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        Returns:
            Output tensor of shape (batch,)
        """
        batch_size, seq_len, features = x.shape

        # Apply batch normalization to each timestep
        x = x.reshape(-1, features)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, features)

        # GRU forward pass
        _, hidden = self.gru(x)  # hidden: (num_layers, batch, hidden_dim)

        # Take last layer's hidden state
        x = hidden[-1]  # (batch, hidden_dim)

        # Output layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x.squeeze(-1)


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

    if len(train) == 0 or len(val) == 0:
        raise ValueError("Error: Train or validation dataset is empty")

    print(f"Train: {len(train)} rows")
    print(f"Val:   {len(val)} rows")

    return train, val


def prepare_data_for_gru(train_df, val_df, feature_cols=SEQUENCE_FEATURES):
    """
    Prepare data for GRU training.
    - Extract features and targets
    - Fit scaler on training data
    - Create sequence datasets
    """
    print(f"\nPreparing data with {len(feature_cols)} features...")
    print(f"Sequence length: {SEQ_LEN} days")

    # Verify all features exist
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df['fear_greed_next'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['fear_greed_next'].values

    # Check for NaN or infinite values
    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError("Error: NaN values found in features")
    if np.isinf(X_train).any() or np.isinf(X_val).any():
        raise ValueError("Error: Infinite values found in features")

    # Fit scaler on training data only (prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create sequence datasets
    train_dataset = SequenceDataset(X_train_scaled, y_train, SEQ_LEN)
    val_dataset = SequenceDataset(X_val_scaled, y_val, SEQ_LEN)

    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, scaler


def evaluate_on_loader(model, loader, criterion, device=DEVICE):
    """Evaluate model on a DataLoader"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)


def train_gru(train_loader, val_loader, device=DEVICE):
    """
    Train GRU model with early stopping and learning rate scheduling.
    """
    print("\n" + "=" * 70)
    print("Training GRU Model")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Hidden dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}")
    print(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")

    # Initialize model
    model = FearGreedGRU(
        input_dim=len(SEQUENCE_FEATURES),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print("\nTraining started...")

    for epoch in range(MAX_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping (RNN best practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        val_loss, val_preds, val_targets = evaluate_on_loader(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improved = "✓"
        else:
            patience_counter += 1
            improved = ""

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or improved:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"LR={current_lr:.2e} {improved}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Final evaluation on validation set
    val_loss, val_preds, val_targets = evaluate_on_loader(model, val_loader, criterion, device)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    mae = mean_absolute_error(val_targets, val_preds)
    r2 = r2_score(val_targets, val_preds)

    # MAPE (handle near-zero values)
    mask = np.abs(val_targets) > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((val_targets[mask] - val_preds[mask]) / val_targets[mask])) * 100
    else:
        mape = float('nan')

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

    return model, history


def save_model(model, scaler, filepath='models/feargreed_model.pt'):
    """Save trained model and scaler to disk"""
    try:
        Path('models').mkdir(exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': len(SEQUENCE_FEATURES),
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'seq_len': SEQ_LEN
            },
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'feature_cols': SEQUENCE_FEATURES
        }

        torch.save(checkpoint, filepath)
        print(f"\n✓ Model saved to {filepath}")

    except Exception as e:
        raise IOError(f"Error saving model: {str(e)}")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("GRU Model Training Pipeline - Fear/Greed Index Predictor")
    print("=" * 70)

    try:
        # Step 1: Load data
        train_df, val_df = load_train_val()

        # Step 2: Prepare data for GRU
        train_loader, val_loader, scaler = prepare_data_for_gru(train_df, val_df)

        # Step 3: Train model
        model, history = train_gru(train_loader, val_loader)

        # Step 4: Save model
        model_path = 'models/feargreed_model.pt'
        save_model(model, scaler, model_path)

        # Summary
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nModel saved to: {model_path}")
        print(f"Training epochs: {len(history['train_loss'])}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print("\nNext step: Run 'python src/evaluate_model.py' to evaluate on test set")

    except Exception as e:
        print(f"\n❌ ERROR: Training failed!")
        print(f"   {str(e)}")
        print("\nPlease check the error message above and fix the issue.")
        raise


if __name__ == '__main__':
    main()

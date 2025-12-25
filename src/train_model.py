"""
GRU Model Training Pipeline
Trains a GRU neural network for Fear/Greed Index prediction.
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
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

SEQ_LEN = 7
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
GRAD_CLIP = 1.0

SEQUENCE_FEATURES = [
    'fear_greed', 'btc_return_1d', 'btc_return_7d',
    'gold_return_1d', 'gold_return_7d', 'btc_gold_ratio',
    'ratio_change_7d', 'btc_vol_7d', 'btc_ma7_ratio', 'gold_ma7_ratio'
]

DEVICE = torch.device('cpu')

class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.X, self.y = self._create_sequences(features, targets)

    def _create_sequences(self, features, targets):
        X, y = [], []
        for i in range(len(features) - self.seq_len):
            X.append(features[i:i + self.seq_len])
            y.append(targets[i + self.seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class FearGreedGRU(nn.Module):
    def __init__(self, input_dim=len(SEQUENCE_FEATURES), hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.reshape(-1, features)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, features)
        _, hidden = self.gru(x)
        x = hidden[-1]
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x.squeeze(-1)

def load_train_val():
    try:
        train = pd.read_csv('datasets/train.csv')
        val = pd.read_csv('datasets/val.csv')
        return train, val
    except FileNotFoundError:
        raise FileNotFoundError("Datasets not found. Run prepare_data.py first.")

def prepare_data_for_gru(train_df, val_df, feature_cols=SEQUENCE_FEATURES):
    X_train = train_df[feature_cols].values
    y_train = train_df['fear_greed_next'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['fear_greed_next'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = SequenceDataset(X_train_scaled, y_train, SEQ_LEN)
    val_dataset = SequenceDataset(X_val_scaled, y_val, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, scaler

def calculate_direction_accuracy(y_true, y_pred):
    """Calculate percentage of correct directional predictions"""
    if len(y_true) < 2: return 0.0
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    correct = np.sign(y_true_diff) == np.sign(y_pred_diff)
    return np.mean(correct) * 100

def evaluate_on_loader(model, loader, criterion, device=DEVICE):
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

def plot_training_history(history):
    """Plot Loss, RMSE, MAE and Direction Accuracy over epochs"""
    print("\nGenerating training plots...")
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics over Epochs', fontsize=16)

    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss (MSE)')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['val_rmse'], label='Val RMSE', color='orange')
    axs[0, 1].set_title('Root Mean Squared Error (RMSE)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, history['val_mae'], label='Val MAE', color='green')
    axs[1, 0].set_title('Mean Absolute Error (MAE)')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, history['val_acc'], label='Val Accuracy', color='purple')
    axs[1, 1].set_title('Direction Accuracy (%)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Accuracy %')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()

    Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/training_metrics.png')
    plt.close()
    print("✓ Plots saved to outputs/training_metrics.png")

def train_gru(train_loader, val_loader, device=DEVICE):
    print(f"\nTraining on {device}...")

    model = FearGreedGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_acc': []
    }

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        val_loss, val_preds, val_targets = evaluate_on_loader(model, val_loader, criterion, device)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_acc = calculate_direction_accuracy(val_targets, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improved = "*"
        else:
            patience_counter += 1
            improved = ""

        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.2f}|{val_loss:.2f} "
              f"RMSE={val_rmse:.2f} MAE={val_mae:.2f} Acc={val_acc:.1f}% {improved}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return model, history

def save_model(model, scaler):
    Path('models').mkdir(exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {'input_dim': len(SEQUENCE_FEATURES), 'hidden_dim': HIDDEN_DIM,
                         'num_layers': NUM_LAYERS, 'dropout': DROPOUT, 'seq_len': SEQ_LEN},
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'feature_cols': SEQUENCE_FEATURES
    }
    torch.save(checkpoint, 'models/feargreed_model.pt')
    print("\n✓ Model saved to models/feargreed_model.pt")

def main():
    try:
        train_df, val_df = load_train_val()
        train_loader, val_loader, scaler = prepare_data_for_gru(train_df, val_df)
        model, history = train_gru(train_loader, val_loader)
        save_model(model, scaler)

        plot_training_history(history)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == '__main__':
    main()
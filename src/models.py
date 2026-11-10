"""
Predictive Modeling Suite
=========================
Contains the Random Forest baseline and the PyTorch LSTM sequential architecture.
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RandomForestBaseline:
    """Random Forest regressor trained on rolling temporal summary statistics."""
    
    def __init__(self, n_estimators: int = 150, max_depth: int = 18, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, feature_names: list) -> pd.DataFrame:
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": self.model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        return fi


if TORCH_AVAILABLE:
    class TurbofanLSTM(nn.Module):
        """Deep LSTM architecture for multi-sensor sequential health tracking."""
        def __init__(self, input_dim: int = 15, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            pred_rul = self.fc(last_step)
            return pred_rul.squeeze(-1)


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = "models/lstm_turbofan.pt"
) -> Any:
    """Trains the PyTorch LSTM on sliding window sequences."""
    if not TORCH_AVAILABLE:
        print("[WARN] PyTorch not available. Skipping LSTM training.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]
    model = TurbofanLSTM(input_dim=input_dim).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        if epoch % 10 == 0 or epoch == epochs:
            rmse_epoch = np.sqrt(total_loss / len(X_train))
            print(f"[Epoch {epoch:02d}/{epochs:02d}] Train RMSE: {rmse_epoch:.2f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model


def predict_lstm(model: Any, X_test: np.ndarray) -> np.ndarray:
    """Runs inference using trained LSTM model."""
    if not TORCH_AVAILABLE or model is None:
        return np.zeros(len(X_test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X_test).to(device)
        preds = model(tensor_x).cpu().numpy()
    return preds

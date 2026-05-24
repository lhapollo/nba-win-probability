import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = ["score_diff", "seconds_remaining", "is_overtime", "possession", "diff_x_time"]

class WinProbMLP(nn.Module): 
    def __init__(self, input_dim: int = 5): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x).squeeze(-1)
    
def train(df: pd.DataFrame, models_dir: Path, epochs: int = 20, batch_size: int = 2048): 
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["home_team_won"].values.astype(np.float32)
    groups = df["game_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    #scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    #tensors
    X_Train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = WinProbMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X_Train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader: 
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    #evaluate
    model.eval()
    with torch.no_grad(): 
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model(X_test_t).cpu().numpy()

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = ((probs > 0.5) == y_test).mean()

    logger.info(f"[PyTorch] Accuracy: {acc: .4f} | Log Loss: {ll: .4f} | Brier Score: {brier:.4f}")

    torch.save(model.state_dict(), models_dir / "pytorch_mlp.pt")

    import joblib
    joblib.dump(scaler, models_dir / "pytorch_scaler.pkl")
    logger.info("Saved pytorch_mlp.pt and pytorch_scaler.pkl")

    return model, scaler
import torch
import joblib
import numpy as np
from pathlib import Path
from nba_wp.models.pytorch_mlp import WinProbMLP
from nba_wp.features import build_features_single

FEATURE_COLS = ["score_diff", "seconds_remaining", "is_overtime", "possession", "diff_x_time"]

class WinProbabilityModel:
    def __init__ (self, model_path: Path, scaler_path: Path):
        #load scaler
        self.scaler = joblib.load(scaler_path)

        #load model weights
        self.model = WinProbMLP(input_dim = 5)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, game_state: dict) -> float: 
        """
        Takes game state dict, returns home win probability as a float
        Expected keys: period, clock, scoreHome, scoreAway, location (optional)
        """
        features_df = build_features_single(game_state)
        X = features_df[FEATURE_COLS].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            tensor = torch.tensor(X_scaled, dtype=torch.float32)
            prob = self.model(tensor).item()

        return round(prob, 4)
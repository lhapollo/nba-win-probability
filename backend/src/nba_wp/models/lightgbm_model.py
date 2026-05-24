import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = ["score_diff", "seconds_remaining", "is_overtime", "possession", "diff_x_time"]

def train(df: pd.DataFrame, models_dir: Path) -> lgb.LGBMClassifier: 
    X = df[FEATURE_COLS].values
    y = df["home_team_won"].values
    groups = df["game_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=1)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = (model.predict(X_test) == y_test).mean()

    logger.info(f"[LightGBM] Accuracy: {acc:4f} | Log Loss: {ll: .4f} | Brier Score: {brier:.4f}")

    out_path = models_dir / "lightgbm_model.pkl"
    joblib.dump(model, out_path)
    logger.info(f"Saved LightGBM model to {out_path}")

    return model
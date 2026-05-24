import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FEATURE_COLS = ["score_diff", "seconds_remaining", "is_overtime", "possession", "diff_x_time"]

def main(): 
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # load
    logger.info("Loading training data...")
    df = pd.read_parquet(PROCESSED_DIR / "training.parquet")

    X = df[FEATURE_COLS]
    y = df["home_team_won"].values
    groups = df["game_id"].values

    # train/test split by game
    logger.info("Splitting data by game...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info(f"Train: {len(X_train):,} plays | Test: {len(X_test):,} plays")

    # train
    logger.info("Training logistic regression...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # evaluate
    from sklearn.metrics import log_loss, brier_score_loss
    from sklearn.calibration import calibration_curve

    probs = model.predict_proba(X_test)[:, 1]

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = (model.predict(X_test) == y_test).mean()

    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Log Loss : {ll:.4f}")
    logger.info(f"Brier Score : {brier:.4f}")

    # save
    out_path = MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(model, out_path)
    logger.info(f"Saved model to {out_path}")

if __name__ == "__main__":
    main()
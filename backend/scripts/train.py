import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
from nba_wp.models import lightgbm_model, pytorch_mlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FEATURE_COLS = ["score_diff", "seconds_remaining", "is_overtime", "possession", "diff_x_time"]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data...")
    df = pd.read_parquet(PROCESSED_DIR / "training.parquet")

    logger.info("Training LightGBM...")
    lightgbm_model.train(df, MODELS_DIR)

    logger.info("Training PyTorch MLP...")
    pytorch_mlp.train(df, MODELS_DIR)


if __name__ == "__main__":
    main()
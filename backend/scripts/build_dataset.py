import logging
import pandas as pd
from pathlib import Path
from nba_wp.features import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def get_game_outcome(df: pd.DataFrame) -> bool | None: 
    """Return True if home team won, based on final score."""
    scored = df[df["scoreHome"] != ""]
    if scored.empty: 
        return None
    last = scored.iloc[-1]
    home = pd.to_numeric(last["scoreHome"], errors="coerce")
    away = pd.to_numeric(last["scoreAway"], errors="coerce")
    if pd.isna(home) or pd.isna(away) or home == away:
        return None
    return bool(home > away)

def main(): 
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = list(RAW_DIR.glob("*.parquet"))
    logger.info(f"Processing {len(files)} games...")

    all_frames = []

    for i, path in enumerate(files):
        try: 
            df = pd.read_parquet(path)
            outcome  = get_game_outcome(df)
            if outcome is None: 
                logger.warning(f"Skipping {path.stem}... could not determine outcome")
                continue
            features = build_features(df, home_team_won=outcome)
            features["game_id"] = path.stem
            all_frames.append(features)
        except Exception as e:
            logger.error(f"Error processing {path.stem}: {e}")

        if (i+1) % 100 == 0: 
            logger.info(f"Processed {i+1}/{len(files)} games...")
    
    dataset = pd.concat(all_frames, ignore_index=True)
    out_path = PROCESSED_DIR / "training.parquet"
    dataset.to_parquet(out_path)
    logger.info(f"Saved {len(dataset)} rows to {out_path}")
    logger.info(f"Class balance: {dataset['home_team_won'].mean():.3f} home win rate")

if __name__ == "__main__":
    main()
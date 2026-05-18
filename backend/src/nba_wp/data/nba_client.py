import time
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[3] / "data" / "raw"

def get_season_game_ids(season: str, season_type: str = "Regular Season") -> list: 
    """Get all game IDs for a given season."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00", # NBA league ID
        season_type_nullable=season_type,
    )
    games = finder.get_data_frames()[0]
    return games["GAME_ID"].unique().tolist()

def fetch_game_pbp(game_id: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame: 
    """Fetching play-by-play data for a single game with caching and retries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{game_id}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    for attempt in range(5): 
        try: 
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
            pbp.to_parquet(cache_path)
            logger.info(f"Cached {game_id}")
            return pbp
        except Exception as e: 
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed for {game_id}: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
        
    raise RuntimeError(f"Failed to fetch play-by-play for {game_id} after 5 attempts.")

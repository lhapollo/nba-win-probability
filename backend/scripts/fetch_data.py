import argparse
import time
import logging
from nba_wp.data.nba_client import get_season_game_ids, fetch_game_pbp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None, help="cap number of games (for testing)")
    parser.add_argument(
        "--season_type",
        default="both",
        choices=["Regular Season", "Playoffs", "both"],
        help="which game types to fetch (default: both)"
    )
    args = parser.parse_args()

    logger.info(f"Fetching game IDs for season {args.season}...")
    game_ids = get_season_game_ids(args.season)

    if args.limit: 
        game_ids = game_ids[:args.limit]
    
    logger.info(f"Pulling {len(game_ids)} games...")

    for i, game_id in enumerate(game_ids):
        try: 
            fetch_game_pbp(game_id)
            logger.info(f"[{i+1}/{len(game_ids)}] {game_id}")
        except Exception as e: 
            logger.error(str(e))
        time.sleep(args.delay)

if __name__ == "__main__":
    main()
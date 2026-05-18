import re
import pandas as pd
import numpy as np

#parsing helpers

def parse_clock(clock_str: str) -> float: 
    """Converting PT:SS format to seconds remaining in period"""
    if not clock_str or pd.isna(clock_str):
        return 0.0
    match = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if not match:
        return 0.0
    minutes, seconds = match.groups()
    return float(minutes) * 60 + float(seconds)

def period_to_seconds_remaining(period: int, clock_seconds: float, regulation_periods: int = 4) -> float: 
    """Convert period and clock to total seconds remaining in the game"""
    period_length = 720 # 12 mins
    ot_period_length = 300 # 5 mins

    if period <= regulation_periods: 
        periods_left = regulation_periods - period
        return periods_left * period_length + clock_seconds
    else: 
        return clock_seconds
    
#feature engineering

def build_features(df: pd.DataFrame, home_team_won: bool = None) -> pd.DataFrame: 
    """Transforming raw play-by-play data into features for win probability modeling. Works for both historical (training) and live (inference) data."""
    df = df.copy()

    # Forward fill
    df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0).astype(int)
    df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0).astype(int)

    # Score differential (from pov of home team)
    df["score_diff"] = df["scoreHome"] - df["scoreAway"]

    # Parse clock + compute seconds remaining
    df["clock_seconds"] = df["clock"].apply(parse_clock)
    df["seconds_remaining"] = df.apply(
        lambda r: period_to_seconds_remaining(r["period"], r["clock_seconds"]), axis = 1
    )

    # is it overtime
    df["is_overtime"] = (df["period"] > 4).astype(int)

    # possession indicator
    df["possession"] = df.apply(infer_possession, axis=1)

    df["diff_x_time"] = df["score_diff"] * df["seconds_remaining"]

    if home_team_won is not None: 
        df["home_team_won"] = int(home_team_won)

    feature_cols = [
        "score_diff",
        "seconds_remaining",
        "is_overtime",
        "possession",
        "diff_x_time",
    ]

    return df[feature_cols + (["home_team_won"] if home_team_won is not None else [])]

def infer_possession(row) -> int: 
    """Infers possession from action type and location. Return 1 (home), 0 (away), or -1 (unknown)"""

    action = str(row.get("actionType", "")).lower()
    location = str(row.get("location", "")).lower()

    # checking if shot attempt is on home side/away side
    if location == "home": 
        return 1
    if location == "away":
        return 0
    
    if action in ("turnover", "steal"):
        # if it's a turnover or steal, possession changes
       team = str(row.get("teamTricode", "")).lower()
       if team: 
           return -1
    
    return -1

def build_features_single(game_state: dict) -> pd.DataFrame: 
    """
    Build features from a single game state dict. Used by inference module during live games.
    Expected keys: period, clock, scoreHome, scoreWaay, location (optional)
    """
    df = pd.DataFrame([game_state])
    return build_features(df)

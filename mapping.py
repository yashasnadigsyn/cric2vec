"""
Mapping module for Cricket2Vec.
Centralizes all ID mapping logic for players, teams, and outcomes.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Set

# =============================================================================
# Outcome Definitions
# =============================================================================

# Granular outcome classes for the model
OUTCOME_CLASSES = [
    # Runs (Normal Deliveries)
    '0_run', '1_run', '2_run', '3_run', '4_run', '5_run', '6_run',
    
    # Extras (Primary events)
    'wide', 'noball',
    
    # Wickets (Specific Types)
    'w_caught', 
    'w_bowled', 
    'w_lbw', 
    'w_runout', 
    'w_stumped', 
    'w_caught_and_bowled',
    
    # Catch-all for rare wickets (hit wicket, obstructing field, etc.)
    'w_other' 
]


def get_outcome_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get outcome-to-ID and ID-to-outcome mappings.
    
    Returns:
        outcome_to_id: Dictionary mapping outcome names to integer IDs
        id_to_outcome: Dictionary mapping integer IDs to outcome names
    """
    outcome_to_id = {outcome: idx for idx, outcome in enumerate(OUTCOME_CLASSES)}
    id_to_outcome = {idx: outcome for idx, outcome in enumerate(OUTCOME_CLASSES)}
    return outcome_to_id, id_to_outcome


def get_outcome_id(row, outcome_to_id: Dict[str, int]) -> int:
    """
    Determines the Outcome Label for a single row (ball).
    Priority: Wicket > Extras > Runs
    
    Args:
        row: A pandas DataFrame row representing a single delivery
        outcome_to_id: Dictionary mapping outcome names to IDs
        
    Returns:
        Integer ID for the outcome
    """
    # 1. Check for Wicket
    if isinstance(row['wicket_type'], str):
        w_type = row['wicket_type'].lower()
        
        wicket_map = {
            'caught': 'w_caught',
            'bowled': 'w_bowled',
            'lbw': 'w_lbw',
            'run out': 'w_runout',
            'stumped': 'w_stumped',
            'caught and bowled': 'w_caught_and_bowled',
        }
        
        outcome_key = wicket_map.get(w_type, 'w_other')
        return outcome_to_id[outcome_key]

    # 2. Check for Extras (Wides / No Balls)
    if row['wides'] > 0:
        return outcome_to_id['wide']
    
    if row['noballs'] > 0:
        return outcome_to_id['noball']
        
    # 3. Runs off Bat (0-6)
    runs = int(row['runs_off_bat'])
    label = f"{runs}_run"
    
    if label in outcome_to_id:
        return outcome_to_id[label]
    else:
        # Cap at 6 for rare cases (>6 runs or data errors)
        if runs > 6:
            return outcome_to_id['6_run']
        return outcome_to_id['0_run']


def save_outcome_mapping(filepath: Path) -> None:
    """Save outcome mapping to a JSON file."""
    _, id_to_outcome = get_outcome_mappings()
    with open(filepath, 'w') as f:
        json.dump(id_to_outcome, f, indent=4)


# =============================================================================
# Player Mapping Functions
# =============================================================================

def create_player_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create player name to ID mapping from a DataFrame.
    
    Args:
        df: DataFrame containing player columns (striker, non_striker, bowler, etc.)
        
    Returns:
        Dictionary mapping player names to integer IDs
    """
    # Collect all unique players from relevant columns
    all_players: Set[str] = set()
    player_columns = ['striker', 'non_striker', 'bowler', 'player_dismissed', 'other_player_dismissed']
    
    for col in player_columns:
        if col in df.columns:
            all_players.update(df[col].dropna().unique())
    
    # Create mapping (sorted for consistency across runs)
    player_list = sorted(all_players)
    player_to_id = {player: idx for idx, player in enumerate(player_list)}
    
    return player_to_id


def save_player_mapping(player_to_id: Dict[str, int], filepath: Path) -> None:
    """Save player mapping to a CSV file."""
    mapping_df = pd.DataFrame({
        'player_name': list(player_to_id.keys()),
        'player_id': list(player_to_id.values())
    })
    mapping_df.to_csv(filepath, index=False)
    print(f"Saved player mapping: {filepath}")
    print(f"Total players mapped: {len(player_to_id)}")


def load_player_mapping(filepath: Path) -> Dict[str, int]:
    """Load player mapping from a CSV file."""
    df = pd.read_csv(filepath)
    return dict(zip(df['player_name'], df['player_id']))


# =============================================================================
# Team Mapping Functions
# =============================================================================

def create_team_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create team name to ID mapping from a DataFrame.
    
    Args:
        df: DataFrame containing team columns (batting_team, bowling_team)
        
    Returns:
        Dictionary mapping team names to integer IDs
    """
    all_teams = set(df['batting_team'].unique()) | set(df['bowling_team'].unique())
    team_list = sorted(all_teams)
    team_to_id = {team: idx for idx, team in enumerate(team_list)}
    
    return team_to_id


def save_team_mapping(team_to_id: Dict[str, int], filepath: Path) -> None:
    """Save team mapping to a CSV file."""
    team_df = pd.DataFrame({
        'team_name': list(team_to_id.keys()),
        'team_id': list(team_to_id.values())
    })
    team_df.to_csv(filepath, index=False)
    print(f"Saved team mapping: {filepath}")
    print(f"Total teams mapped: {len(team_to_id)}")


def load_team_mapping(filepath: Path) -> Dict[str, int]:
    """Load team mapping from a CSV file."""
    df = pd.read_csv(filepath)
    return dict(zip(df['team_name'], df['team_id']))


# =============================================================================
# Utility Functions
# =============================================================================

def get_num_players(filepath: Path) -> int:
    """Get the number of players from a mapping file."""
    df = pd.read_csv(filepath)
    return len(df)


def get_num_outcomes() -> int:
    """Get the number of outcome classes."""
    return len(OUTCOME_CLASSES)

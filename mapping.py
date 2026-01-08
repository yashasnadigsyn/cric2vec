"""
Mapping module for Cricket2Vec.
Centralizes all ID mapping logic for players, teams, and outcomes.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Set

# =============================================================================
# Outcome Definitions (Merged for better class balance)
# =============================================================================

# Reduced outcome classes (16 -> 12) to address class imbalance
OUTCOME_CLASSES = [
    # Runs (Normal Deliveries)
    '0_run',      # 0: dot balls
    '1_run',      # 1: singles
    '2_3_run',    # 2: doubles and triples merged (rare 3s)
    '4_run',      # 3: boundaries
    '6_run',      # 4: sixes
    
    # Extras (merged: wides, no-balls, 5 runs)
    'extras',     # 5: all extras combined
    
    # Wickets (keeping distinct for cricket-meaningful context)
    'w_caught',   # 6: caught
    'w_bowled',   # 7: bowled
    'w_lbw',      # 8: leg before wicket
    'w_runout',   # 9: run out
    'w_stumped',  # 10: stumped
    'w_other'     # 11: caught_and_bowled + other rare wickets
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
    
    Uses merged outcome categories for better class balance.
    
    Args:
        row: A pandas DataFrame row representing a single delivery
        outcome_to_id: Dictionary mapping outcome names to IDs
        
    Returns:
        Integer ID for the outcome
    """
    # 1. Check for Wicket
    if isinstance(row['wicket_type'], str):
        w_type = row['wicket_type'].lower()
        
        # Map wicket types - merge rare ones into w_other
        wicket_map = {
            'caught': 'w_caught',
            'bowled': 'w_bowled',
            'lbw': 'w_lbw',
            'run out': 'w_runout',
            'stumped': 'w_stumped',
            # Rare wickets merged into w_other
            'caught and bowled': 'w_other',
            'hit wicket': 'w_other',
            'obstructing the field': 'w_other',
        }
        
        outcome_key = wicket_map.get(w_type, 'w_other')
        return outcome_to_id[outcome_key]

    # 2. Check for Extras (wides, no-balls -> merged to 'extras')
    if row['wides'] > 0 or row['noballs'] > 0:
        return outcome_to_id['extras']
        
    # 3. Runs off Bat
    runs = int(row['runs_off_bat'])
    
    # Map runs to merged categories
    if runs == 0:
        return outcome_to_id['0_run']
    elif runs == 1:
        return outcome_to_id['1_run']
    elif runs in [2, 3]:
        return outcome_to_id['2_3_run']  # Merge 2 and 3 runs
    elif runs == 4:
        return outcome_to_id['4_run']
    elif runs == 5:
        return outcome_to_id['extras']  # 5 runs usually means extras (wide+4, noball+4)
    elif runs >= 6:
        return outcome_to_id['6_run']
    else:
        return outcome_to_id['0_run']  # Fallback


def save_outcome_mapping(filepath: Path) -> None:
    """Save outcome mapping to a JSON file."""
    outcome_to_id, id_to_outcome = get_outcome_mappings()
    with open(filepath, 'w') as f:
        json.dump({
            'outcome_to_id': outcome_to_id,
            'id_to_outcome': {str(k): v for k, v in id_to_outcome.items()}
        }, f, indent=4)


def load_outcome_mapping(filepath: Path) -> dict:
    """Load outcome mapping from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (just id_to_outcome) and new format (with outcome_to_id)
    if 'outcome_to_id' in data:
        return data
    else:
        # Old format: just {id: outcome}
        id_to_outcome = {int(k): v for k, v in data.items()}
        outcome_to_id = {v: int(k) for k, v in data.items()}
        return {
            'outcome_to_id': outcome_to_id,
            'id_to_outcome': id_to_outcome
        }


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
# Venue Mapping Functions
# =============================================================================

def create_venue_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create venue name to ID mapping from a DataFrame.
    
    Args:
        df: DataFrame containing venue column
        
    Returns:
        Dictionary mapping venue names to integer IDs
    """
    venues = sorted(df['venue'].dropna().unique())
    venue_to_id = {venue: idx for idx, venue in enumerate(venues)}
    
    return venue_to_id


def save_venue_mapping(venue_to_id: Dict[str, int], filepath: Path) -> None:
    """Save venue mapping to a CSV file."""
    venue_df = pd.DataFrame({
        'venue_name': list(venue_to_id.keys()),
        'venue_id': list(venue_to_id.values())
    })
    venue_df.to_csv(filepath, index=False)
    print(f"Saved venue mapping: {filepath}")
    print(f"Total venues mapped: {len(venue_to_id)}")


def load_venue_mapping(filepath: Path) -> Dict[str, int]:
    """Load venue mapping from a CSV file."""
    df = pd.read_csv(filepath)
    return dict(zip(df['venue_name'], df['venue_id']))


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

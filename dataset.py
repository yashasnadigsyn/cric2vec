
"""
PyTorch Dataset for Cricket2Vec training.
Handles loading and preprocessing of ball-by-ball cricket data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from mapping import (
    get_outcome_mappings, get_outcome_id,
    load_player_mapping, load_team_mapping, load_venue_mapping
)


class CricketDataset(Dataset):
    def __init__(self, parquet_file: str, player_mapping_file: str, team_mapping_file: str, venue_mapping_file: str) -> None:
        """
        Args:
            parquet_file (str or Path): Path to the ball-by-ball parquet file.
            player_mapping_file (str or Path): Path to the CSV containing player mappings.
            team_mapping_file (str or Path): Path to the CSV containing team mappings.
            venue_mapping_file (str or Path): Path to the CSV containing venue mappings.
        """
        self.data: pd.DataFrame = pd.read_parquet(parquet_file)
        
        # Load mappings
        self.player_to_id: Dict[str, int] = load_player_mapping(player_mapping_file)
        self.team_to_id: Dict[str, int] = load_team_mapping(team_mapping_file)
        self.venue_to_id: Dict[str, int] = load_venue_mapping(venue_mapping_file)
        
        # Get outcome mappings from centralized module
        self.outcome_to_id: Dict[str, int]
        self.id_to_outcome: Dict[int, str]
        self.outcome_to_id, self.id_to_outcome = get_outcome_mappings()
        
        # Preprocess data: filter and map players/teams/venues to IDs
        self.data = self._preprocess_data(self.data)
        
        # Generate target labels using centralized outcome logic
        self.data['outcome_id'] = self.data.apply(
            lambda row: get_outcome_id(row, self.outcome_to_id), axis=1
        )
        
        # Convert to Tensors
        self.striker_ids = torch.LongTensor(self.data['striker_id'].values)
        self.bowler_ids = torch.LongTensor(self.data['bowler_id'].values)
        self.bat_team_ids = torch.LongTensor(self.data['batting_team_id'].values)
        self.bowl_team_ids = torch.LongTensor(self.data['bowling_team_id'].values)
        self.venue_ids = torch.LongTensor(self.data['venue_id'].values)
        self.outcome_ids = torch.LongTensor(self.data['outcome_id'].values)
        
        # Precompute context features (over, innings normalized to [0,1])
        self._precompute_context_features()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and map names to IDs."""
        # Map player names to IDs
        df['striker_id'] = df['striker'].map(self.player_to_id)
        df['bowler_id'] = df['bowler'].map(self.player_to_id)
        
        # Map team names to IDs
        df['batting_team_id'] = df['batting_team'].map(self.team_to_id)
        df['bowling_team_id'] = df['bowling_team'].map(self.team_to_id)
        
        # Map venue names to IDs
        df['venue_id'] = df['venue'].map(self.venue_to_id)
        
        # Drop rows where any required field is missing from mapping
        # Note: We prioritize data quality over quantity here.
        # Use dropna separately to see what fails if needed, but for now drop all at once.
        required_cols = ['striker_id', 'bowler_id', 'batting_team_id', 'bowling_team_id', 'venue_id']
        before_len = len(df)
        df = df.dropna(subset=required_cols)
        after_len = len(df)
        if before_len != after_len:
            print(f"Dropped {before_len - after_len} rows due to missing mappings.")
        
        # Convert to int
        for col in required_cols:
            df[col] = df[col].astype(int)
        
        return df
    
    def _precompute_context_features(self) -> None:
        """
        Precompute context features for efficient training.
        
        Context features:
        - over_norm: Over number normalized to [0, 1] (over/20.0)
        - innings_norm: 0 for 1st innings, 1 for 2nd innings
        """
        # Extract over from ball column (e.g., 15.3 -> over 15)
        # Ball column format: over.ball_in_over (e.g., 15.3 = over 15, ball 3)
        overs = self.data['ball'].apply(lambda x: int(float(x))).values
        innings = self.data['innings'].astype(int).values
        
        # Normalize to [0, 1] range
        over_norm = overs / 20.0  # T20 has max 20 overs
        innings_norm = (innings - 1) / 1.0  # 0 for 1st, 1 for 2nd
        
        # Stack into context tensor [N, 2]
        self.context = torch.FloatTensor(
            list(zip(over_norm, innings_norm))
        )

    def __len__(self) -> int:
        return len(self.outcome_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            striker_id (int tensor)
            bowler_id (int tensor)
            bat_team_id (int tensor)
            bowl_team_id (int tensor)
            venue_id (int tensor)
            context (float tensor): [over_norm, innings_norm]
            outcome_id (int tensor)
        """
        return (self.striker_ids[idx], self.bowler_ids[idx], 
                self.bat_team_ids[idx], self.bowl_team_ids[idx],
                self.venue_ids[idx],
                self.context[idx], self.outcome_ids[idx])

    def get_num_players(self) -> int:
        return len(self.player_to_id)
    
    def get_num_teams(self) -> int:
        return len(self.team_to_id)
    
    def get_num_venues(self) -> int:
        return len(self.venue_to_id)

    def get_num_outcomes(self) -> int:
        return len(self.outcome_to_id)
    
    def get_match_ids(self) -> pd.Series:
        """Return match IDs for temporal splitting."""
        return self.data['match_id'].values


# --- Usage Example ---
if __name__ == "__main__":
    from config import COMBINED_PARQUET, PLAYER_MAPPING, TEAM_MAPPING, VENUE_MAPPING
    
    print("Initializing Dataset...")
    dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING, TEAM_MAPPING, VENUE_MAPPING)
    
    print(f"Dataset Size: {len(dataset)}")
    print(f"Num Players: {dataset.get_num_players()}")
    print(f"Num Teams: {dataset.get_num_teams()}")
    print(f"Num Venues: {dataset.get_num_venues()}")
    print(f"Num Outcomes: {dataset.get_num_outcomes()}")
    print(f"Unique Matches: {len(set(dataset.get_match_ids()))}")
    
    print("\nSample Data (First 5):")
    for i in range(5):
        s, b, bt, bo, v, ctx, o = dataset[i]
        outcome_label = dataset.id_to_outcome[o.item()]
        print(f"Ball {i}: Striker {s} vs Bowler {b} | Teams {bt} vs {bo} | Venue {v} -> Outcome {o} ({outcome_label})")
        print(f"  Context: over_norm={ctx[0]:.2f}, innings_norm={ctx[1]:.0f}")

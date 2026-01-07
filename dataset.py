"""
PyTorch Dataset for Cricket2Vec training.
Handles loading and preprocessing of ball-by-ball cricket data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from mapping import (
    get_outcome_mappings, get_outcome_id,
    load_player_mapping
)


class CricketDataset(Dataset):
    def __init__(self, parquet_file, player_mapping_file):
        """
        Args:
            parquet_file (str or Path): Path to the ball-by-ball parquet file.
            player_mapping_file (str or Path): Path to the CSV containing player mappings.
        """
        self.data = pd.read_parquet(parquet_file)
        
        # Load player mapping from file
        self.player_to_id = load_player_mapping(player_mapping_file)
        
        # Get outcome mappings from centralized module
        self.outcome_to_id, self.id_to_outcome = get_outcome_mappings()
        
        # Preprocess data: filter and map players to IDs
        self.data = self._preprocess_data(self.data)
        
        # Generate target labels using centralized outcome logic
        self.data['outcome_id'] = self.data.apply(
            lambda row: get_outcome_id(row, self.outcome_to_id), axis=1
        )
        
        # Convert to Tensors
        self.striker_ids = torch.LongTensor(self.data['striker_id'].values)
        self.bowler_ids = torch.LongTensor(self.data['bowler_id'].values)
        self.outcome_ids = torch.LongTensor(self.data['outcome_id'].values)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and map player names to IDs."""
        # Map player names to IDs
        df['striker_id'] = df['striker'].map(self.player_to_id)
        df['bowler_id'] = df['bowler'].map(self.player_to_id)
        
        # Drop rows where players might be missing from mapping
        df = df.dropna(subset=['striker_id', 'bowler_id'])
        
        # Convert to int
        df['striker_id'] = df['striker_id'].astype(int)
        df['bowler_id'] = df['bowler_id'].astype(int)
        
        return df

    def __len__(self):
        return len(self.outcome_ids)

    def __getitem__(self, idx):
        """
        Returns:
            striker_id (int tensor)
            bowler_id (int tensor)
            outcome_id (int tensor)
        """
        return self.striker_ids[idx], self.bowler_ids[idx], self.outcome_ids[idx]

    def get_num_players(self):
        return len(self.player_to_id)

    def get_num_outcomes(self):
        return len(self.outcome_to_id)


# --- Usage Example ---
if __name__ == "__main__":
    from config import COMBINED_PARQUET, PLAYER_MAPPING
    
    print("Initializing Dataset...")
    dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING)
    
    print(f"Dataset Size: {len(dataset)}")
    print(f"Num Players: {dataset.get_num_players()}")
    print(f"Num Outcomes: {dataset.get_num_outcomes()}")
    
    print("\nSample Data (First 5):")
    for i in range(5):
        s, b, o = dataset[i]
        outcome_label = dataset.id_to_outcome[o.item()]
        print(f"Ball {i}: Striker {s} vs Bowler {b} -> Outcome {o} ({outcome_label})")

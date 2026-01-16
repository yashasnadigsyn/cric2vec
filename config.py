"""
Configuration module for Cricket2Vec.
Centralizes all paths and configuration constants.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ipl_male_csv2"


COMBINED_CSV = BASE_DIR / "ipl_combined.csv"
COMBINED_PARQUET = BASE_DIR / "ipl_combined.parquet"


PLAYER_MAPPING = BASE_DIR / "player_mapping.csv"
TEAM_MAPPING = BASE_DIR / "team_mapping.csv"
VENUE_MAPPING = BASE_DIR / "venue_mapping.csv"
OUTCOME_MAPPING = BASE_DIR / "outcome_mapping.json"


COLUMNS_TO_KEEP = [
    'match_id',
    'season',        # For temporal analysis
    'start_date',    # For temporal/chronological splitting
    'venue',         # For venue embeddings (pitch/ground effects)
    'innings',
    'ball',
    'batting_team',
    'bowling_team',
    'striker',
    'non_striker',
    'bowler',
    'runs_off_bat',
    'extras',
    'wides',
    'noballs',
    'byes',
    'legbyes',
    'penalty',
    'wicket_type',
    'player_dismissed',
    'other_wicket_type',
    'other_player_dismissed',
    'gender' 
]


WPL_JSON_DIR = BASE_DIR / "wpl_json"

# Model Config
MODEL_CONFIG = {
    'embedding_dim': 9,      # Paper uses 9-dimensional embeddings
    'hidden_dim': 128,       # Single hidden layer
}

# Training Config
TRAINING_CONFIG = {
    'batch_size': 128,       # Paper uses 100, we use 128 for GPU efficiency
    'learning_rate': 0.01,   # Paper uses 0.01
    'momentum': 0.9,         # Nesterov momentum
    'weight_decay': 1e-6,    # Paper uses decay of 10^-6
    'epochs': 100,
    'val_split': 0.1,
    'num_workers': 0,        # Set to 4 if running on a machine with multiple cores
}

# Device
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint directory
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

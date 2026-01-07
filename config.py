"""
Configuration module for Cricket2Vec.
Centralizes all paths and configuration constants.
"""

from pathlib import Path

# =============================================================================
# Base Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ipl_male_csv2"

# =============================================================================
# Data Files
# =============================================================================
COMBINED_CSV = BASE_DIR / "ipl_combined.csv"
COMBINED_PARQUET = BASE_DIR / "ipl_combined.parquet"

# =============================================================================
# Mapping Files
# =============================================================================
PLAYER_MAPPING = BASE_DIR / "player_mapping.csv"
TEAM_MAPPING = BASE_DIR / "team_mapping.csv"
OUTCOME_MAPPING = BASE_DIR / "outcome_mapping.json"

# =============================================================================
# Columns to keep from raw CSV files
# =============================================================================
COLUMNS_TO_KEEP = [
    'match_id',
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
    'other_player_dismissed'
]

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    'embedding_dim': 7,
    'hidden_dim_1': 128,
    'hidden_dim_2': 64,
    'dropout': 0.2,
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING_CONFIG = {
    'batch_size': 1024,
    'learning_rate': 0.001,
    'epochs': 100,
    'val_split': 0.1,
    'num_workers': 0,  # Set to 4 if running on a machine with multiple cores
}

# =============================================================================
# Device Configuration
# =============================================================================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint directory
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

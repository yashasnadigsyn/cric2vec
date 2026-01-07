"""
Script to combine all IPL ball-by-ball CSV files into a single dataset.
This prepares the data for Cricket2Vec training.
"""

import pandas as pd
import os
import glob

from config import DATA_DIR, COMBINED_CSV, COMBINED_PARQUET, COLUMNS_TO_KEEP
from config import PLAYER_MAPPING, TEAM_MAPPING, OUTCOME_MAPPING
from mapping import (
    create_player_mapping, save_player_mapping,
    create_team_mapping, save_team_mapping,
    save_outcome_mapping
)


def combine_csv_files() -> pd.DataFrame:
    """Combine all CSVs (excluding _info files) into a single DataFrame."""
    
    # Get all CSV files that are NOT _info files
    csv_pattern = str(DATA_DIR / "*.csv")
    all_files = glob.glob(csv_pattern)
    
    # Filter out _info files
    match_files = [f for f in all_files if '_info' not in f]
    
    print(f"Found {len(match_files)} match files")
    
    # Read and concatenate all files
    dfs = []
    for i, file in enumerate(sorted(match_files)):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            
            if (i + 1) % 200 == 0:
                print(f"Processed {i + 1}/{len(match_files)} files...")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows combined: {len(combined_df):,}")
    
    return combined_df


def process_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns and save to CSV and Parquet."""
    
    # Select only the columns we need (handling missing columns gracefully)
    available_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    missing_cols = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
    
    df_subset = df[available_cols].copy()
    
    # Save as CSV
    df_subset.to_csv(COMBINED_CSV, index=False)
    csv_size = os.path.getsize(COMBINED_CSV) / (1024 * 1024)
    print(f"\nSaved CSV: {COMBINED_CSV}")
    print(f"CSV file size: {csv_size:.2f} MB")
    
    # Save as Parquet (more efficient for repeated loading)
    df_subset.to_parquet(COMBINED_PARQUET, index=False)
    parquet_size = os.path.getsize(COMBINED_PARQUET) / (1024 * 1024)
    print(f"\nSaved Parquet: {COMBINED_PARQUET}")
    print(f"Parquet file size: {parquet_size:.2f} MB")
    
    return df_subset


def print_statistics(df: pd.DataFrame) -> None:
    """Print useful statistics for Cricket2Vec preparation."""
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal Balls (rows): {len(df):,}")
    print(f"Total Matches: {df['match_id'].nunique():,}")
    
    # Player statistics
    all_players = set()
    for col in ['striker', 'non_striker', 'bowler', 'player_dismissed']:
        if col in df.columns:
            all_players.update(df[col].dropna().unique())
    
    print(f"\nUnique Players: {len(all_players):,}")
    print(f"  - Unique Strikers: {df['striker'].nunique():,}")
    print(f"  - Unique Non-Strikers: {df['non_striker'].nunique():,}")
    print(f"  - Unique Bowlers: {df['bowler'].nunique():,}")
    
    # Team statistics
    all_teams = set(df['batting_team'].unique()) | set(df['bowling_team'].unique())
    print(f"\nUnique Teams: {len(all_teams)}")
    print(f"Teams: {sorted(all_teams)}")
    
    # Runs distribution
    print(f"\nRuns off bat distribution:")
    print(df['runs_off_bat'].value_counts().sort_index())
    
    # Extras
    print(f"\nExtras columns:")
    for col in ['wides', 'noballs', 'byes', 'legbyes', 'penalty']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            non_zero = (df[col] > 0).sum() if non_null > 0 else 0
            print(f"  - {col}: {non_zero:,} occurrences")
    
    # Wicket types
    print(f"\nWicket Types:")
    wicket_counts = df['wicket_type'].value_counts()
    for wtype, count in wicket_counts.items():
        print(f"  - {wtype}: {count:,}")
    
    # Other wicket types (rare events)
    if 'other_wicket_type' in df.columns:
        other_wickets = df['other_wicket_type'].dropna()
        if len(other_wickets) > 0:
            print(f"\nOther Wicket Types (rare):")
            print(other_wickets.value_counts())
    
    # Penalty check
    if 'penalty' in df.columns:
        penalty_events = df['penalty'].notna() & (df['penalty'] > 0)
        print(f"\nPenalty Events: {penalty_events.sum()}")


if __name__ == "__main__":
    print("Starting data combination...")
    print("="*60)
    
    # Step 1: Combine all CSVs
    combined_df = combine_csv_files()
    
    # Step 2: Process and save
    subset_df = process_and_save(combined_df)
    
    # Step 3: Print statistics
    print_statistics(subset_df)
    
    # Step 4: Create and save mappings (using centralized mapping module)
    print("\n" + "="*60)
    print("CREATING MAPPINGS")
    print("="*60)
    
    player_to_id = create_player_mapping(subset_df)
    save_player_mapping(player_to_id, PLAYER_MAPPING)
    
    team_to_id = create_team_mapping(subset_df)
    save_team_mapping(team_to_id, TEAM_MAPPING)
    
    save_outcome_mapping(OUTCOME_MAPPING)
    print(f"Saved outcome mapping: {OUTCOME_MAPPING}")
    
    print("\n" + "="*60)
    print("DONE! Files created:")
    print(f"  - {COMBINED_CSV.name} (for exploration)")
    print(f"  - {COMBINED_PARQUET.name} (for fast loading)")
    print(f"  - {PLAYER_MAPPING.name}")
    print(f"  - {TEAM_MAPPING.name}")
    print(f"  - {OUTCOME_MAPPING.name}")
    print("="*60)

"""
Script to combine all IPL ball-by-ball CSV files into a single dataset.
This prepares the data for Cricket2Vec training.
"""

import pandas as pd
import os
import glob
from pathlib import Path

from config import DATA_DIR, COMBINED_CSV, COMBINED_PARQUET, COLUMNS_TO_KEEP, WPL_JSON_DIR
from config import PLAYER_MAPPING, TEAM_MAPPING, VENUE_MAPPING, OUTCOME_MAPPING
from mapping import (
    create_player_mapping, save_player_mapping,
    create_team_mapping, save_team_mapping,
    create_venue_mapping, save_venue_mapping,
    save_outcome_mapping
)


import json

def process_wpl_json(json_file: str) -> pd.DataFrame:
    """Parse a single WPL JSON file and convert to DataFrame matching IPL schema."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    match_id = Path(json_file).stem
    info = data.get('info', {})
    meta = data.get('meta', {})
    
    # Match metadata
    season = info.get('season')
    dates = info.get('dates', [])
    start_date = dates[0] if dates else None
    venue = info.get('venue')
    gender = info.get('gender', 'female') # Default to female for WPL if missing
    
    # Teams
    teams = info.get('teams', [])
    
    rows = []
    
    for i, inning in enumerate(data.get('innings', [])):
        innings_num = i + 1
        batting_team = inning.get('team')
        
        # Determine bowling team
        bowling_team = next((t for t in teams if t != batting_team), None)
        
        for over in inning.get('overs', []):
            over_num = over.get('over')
            
            for ball_idx, delivery in enumerate(over.get('deliveries', [])):
                # Construct ball number (e.g. 0.1, 0.2 ... 0.6)
                # Note: CSV uses 0.1 for first ball of first over
                ball_num = float(f"{over_num}.{ball_idx + 1}")
                
                batter = delivery.get('batter')
                non_striker = delivery.get('non_striker')
                bowler = delivery.get('bowler')
                
                runs = delivery.get('runs', {})
                runs_off_bat = runs.get('batter', 0)
                extras_total = runs.get('extras', 0)
                
                # Extras breakdown
                extras_data = delivery.get('extras', {})
                wides = extras_data.get('wides')
                noballs = extras_data.get('noballs')
                byes = extras_data.get('byes')
                legbyes = extras_data.get('legbyes')
                penalty = extras_data.get('penalty')
                
                # Wickets
                wickets = delivery.get('wickets', [])
                if wickets:
                    # Take the first wicket if multiple (rare but possible in runouts)
                    # Ideally we might want to create multiple rows or handle differently,
                    # but for player embeddings main wicket is key.
                    wicket = wickets[0]
                    wicket_type = wicket.get('kind')
                    player_dismissed = wicket.get('player_out')
                else:
                    wicket_type = None
                    player_dismissed = None
                
                row = {
                    'match_id': match_id,
                    'season': season,
                    'start_date': start_date,
                    'venue': venue,
                    'innings': innings_num,
                    'ball': ball_num,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'striker': batter,
                    'non_striker': non_striker,
                    'bowler': bowler,
                    'runs_off_bat': runs_off_bat,
                    'extras': extras_total,
                    'wides': wides,
                    'noballs': noballs,
                    'byes': byes,
                    'legbyes': legbyes,
                    'penalty': penalty,
                    'wicket_type': wicket_type,
                    'player_dismissed': player_dismissed,
                    'gender': gender,
                    'other_wicket_type': None,        # Placeholders to match CSV
                    'other_player_dismissed': None
                }
                rows.append(row)
                
    return pd.DataFrame(rows)

def combine_all_data() -> pd.DataFrame:
    """Combine IPL CSVs and WPL JSONs into a single DataFrame."""
    
    all_dfs = []
    
    # --- 1. Process IPL CSV Files ---
    print("\nProcessing IPL CSV files...")
    csv_pattern = str(DATA_DIR / "*.csv")
    csv_files = glob.glob(csv_pattern)
    match_csvs = [f for f in csv_files if '_info' not in f]
    
    print(f"Found {len(match_csvs)} IPL CSV files")
    
    for i, file in enumerate(sorted(match_csvs)):
        try:
            df = pd.read_csv(file)
            df['gender'] = 'male'  # Label IPL data as male
            all_dfs.append(df)
            
            if (i + 1) % 200 == 0:
                print(f"Processed {i + 1}/{len(match_csvs)} CSV files...")
        except Exception as e:
            print(f"Error reading CSV {file}: {e}")

    # --- 2. Process WPL JSON Files ---
    print("\nProcessing WPL JSON files...")
    from pathlib import Path
    json_pattern = str(WPL_JSON_DIR / "*.json")
    json_files = glob.glob(json_pattern)
    
    print(f"Found {len(json_files)} WPL JSON files")
    
    for i, file in enumerate(sorted(json_files)):
        try:
            df = process_wpl_json(file)
            all_dfs.append(df)
             
            if (i + 1) % 10 == 0:
                 print(f"Processed {i + 1}/{len(json_files)} JSON files...")
                 
        except Exception as e:
            print(f"Error reading JSON {file}: {e}")
    
    # --- 3. Combine ---
    if not all_dfs:
        print("No data found!")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows combined: {len(combined_df):,}")
    print(f"Gender distribution:\n{combined_df['gender'].value_counts()}")
    
    return combined_df


def process_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns and save to CSV and Parquet."""
    
    # Select only the columns we need (handling missing columns gracefully)
    available_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    missing_cols = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
    
    df_subset = df[available_cols].copy()
    
    # Ensure string columns stay as strings (avoid parquet type inference issues)
    string_cols = ['season', 'venue', 'start_date']
    for col in string_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].astype(str)
    
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
    
    # Step 1: Combine all data (IPL + WPL)
    combined_df = combine_all_data()
    
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
    
    # Create venue mapping if venue column exists
    if 'venue' in subset_df.columns:
        venue_to_id = create_venue_mapping(subset_df)
        save_venue_mapping(venue_to_id, VENUE_MAPPING)
    
    save_outcome_mapping(OUTCOME_MAPPING)
    print(f"Saved outcome mapping: {OUTCOME_MAPPING}")
    
    print("\n" + "="*60)
    print("DONE! Files created:")
    print(f"  - {COMBINED_CSV.name} (for exploration)")
    print(f"  - {COMBINED_PARQUET.name} (for fast loading)")
    print(f"  - {PLAYER_MAPPING.name}")
    print(f"  - {TEAM_MAPPING.name}")
    print(f"  - {VENUE_MAPPING.name}")
    print(f"  - {OUTCOME_MAPPING.name}")
    print("="*60)

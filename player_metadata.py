"""
Player Metadata Preprocessing Script

Creates a unified player metadata file by mapping player names from
player_mapping.csv to batting/bowling styles from players_data_with_all_info.csv
using people.csv and names.csv as intermediate lookups.

Mapping chain:
1. player_mapping.csv: "RG Sharma" → player_id (545)
2. people.csv: "RG Sharma" → identifier (e.g., "740742ef")
3. names.csv: identifier → alternative names (e.g., "Rohit Sharma")
4. players_data_with_all_info.csv: fullname → batting_style, bowling_style, position, country
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz, process

# File paths
BASE_DIR = Path(__file__).parent
PLAYER_MAPPING = BASE_DIR / "player_mapping.csv"
PEOPLE_CSV = BASE_DIR / "people.csv"
NAMES_CSV = BASE_DIR / "names.csv"
PLAYERS_INFO_CSV = BASE_DIR / "players_data_with_all_info.csv"
OUTPUT_CSV = BASE_DIR / "player_metadata.csv"


def load_people_lookup() -> Dict[str, str]:
    """
    Load people.csv to create name → identifier mapping.
    Returns dict mapping player names (e.g., "RG Sharma") to identifiers.
    """
    df = pd.read_csv(PEOPLE_CSV)
    # Use the 'name' column as key and 'identifier' as value
    return dict(zip(df['name'].str.strip(), df['identifier']))


def load_names_lookup() -> Dict[str, List[str]]:
    """
    Load names.csv to create identifier → list of alternative names mapping.
    """
    df = pd.read_csv(NAMES_CSV)
    lookup: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        identifier = row['identifier']
        name = row['name'].strip() if pd.notna(row['name']) else None
        if name:
            if identifier not in lookup:
                lookup[identifier] = []
            lookup[identifier].append(name)
    return lookup


def load_players_info() -> pd.DataFrame:
    """
    Load players_data_with_all_info.csv with relevant columns.
    """
    df = pd.read_csv(PLAYERS_INFO_CSV)
    # Keep only relevant columns
    cols = ['fullname', 'battingstyle', 'bowlingstyle', 'position', 'country_name']
    df = df[[c for c in cols if c in df.columns]].copy()
    df['fullname'] = df['fullname'].str.strip()
    return df


def find_best_match(
    name: str, 
    candidates: List[str], 
    threshold: int = 80
) -> Optional[str]:
    """
    Find the best fuzzy match for a name from a list of candidates.
    
    Args:
        name: The name to match
        candidates: List of candidate names to match against
        threshold: Minimum similarity score (0-100)
        
    Returns:
        Best matching name or None if no match above threshold
    """
    if not candidates:
        return None
    
    result = process.extractOne(name, candidates, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def map_player_to_metadata(
    player_name: str,
    people_lookup: Dict[str, str],
    names_lookup: Dict[str, List[str]],
    players_info: pd.DataFrame,
    fullname_list: List[str]
) -> Optional[Dict]:
    """
    Map a single player name to their metadata.
    
    Args:
        player_name: Name from player_mapping.csv (e.g., "RG Sharma")
        people_lookup: name → identifier mapping
        names_lookup: identifier → alternative names mapping
        players_info: DataFrame with player metadata
        fullname_list: List of all fullnames for fuzzy matching
        
    Returns:
        Dict with batting_style, bowling_style, position, country or None
    """
    # Step 1: Try direct lookup in people.csv
    identifier = people_lookup.get(player_name)
    
    if not identifier:
        # Try fuzzy match on people names
        match = find_best_match(player_name, list(people_lookup.keys()), threshold=85)
        if match:
            identifier = people_lookup[match]
    
    # Step 2: Get alternative names from names.csv
    alt_names = []
    if identifier:
        alt_names = names_lookup.get(identifier, [])
    
    # Step 3: Try to find in players_info using alternative names
    for alt_name in alt_names:
        match_row = players_info[players_info['fullname'] == alt_name]
        if not match_row.empty:
            row = match_row.iloc[0]
            return {
                'batting_style': row.get('battingstyle', ''),
                'bowling_style': row.get('bowlingstyle', ''),
                'position': row.get('position', ''),
                'country': row.get('country_name', '')
            }
    
    # Step 4: Try fuzzy matching alternative names against fullnames
    for alt_name in alt_names:
        best_match = find_best_match(alt_name, fullname_list, threshold=85)
        if best_match:
            match_row = players_info[players_info['fullname'] == best_match]
            if not match_row.empty:
                row = match_row.iloc[0]
                return {
                    'batting_style': row.get('battingstyle', ''),
                    'bowling_style': row.get('bowlingstyle', ''),
                    'position': row.get('position', ''),
                    'country': row.get('country_name', '')
                }
    
    # Step 5: Last resort - fuzzy match player_name directly against fullnames
    best_match = find_best_match(player_name, fullname_list, threshold=75)
    if best_match:
        match_row = players_info[players_info['fullname'] == best_match]
        if not match_row.empty:
            row = match_row.iloc[0]
            return {
                'batting_style': row.get('battingstyle', ''),
                'bowling_style': row.get('bowlingstyle', ''),
                'position': row.get('position', ''),
                'country': row.get('country_name', '')
            }
    
    return None


def create_player_metadata() -> pd.DataFrame:
    """
    Create the unified player metadata DataFrame.
    
    Returns:
        DataFrame with columns: player_name, player_id, batting_style, 
        bowling_style, position, country
    """
    print("Loading source files...")
    
    # Load all source data
    player_mapping = pd.read_csv(PLAYER_MAPPING)
    people_lookup = load_people_lookup()
    names_lookup = load_names_lookup()
    players_info = load_players_info()
    fullname_list = players_info['fullname'].dropna().tolist()
    
    print(f"Player mapping: {len(player_mapping)} players")
    print(f"People lookup: {len(people_lookup)} entries")
    print(f"Names lookup: {len(names_lookup)} identifiers")
    print(f"Players info: {len(players_info)} records")
    
    # Process each player
    results = []
    matched_count = 0
    
    print("\nMapping players to metadata...")
    for idx, row in player_mapping.iterrows():
        player_name = row['player_name']
        player_id = row['player_id']
        
        metadata = map_player_to_metadata(
            player_name, 
            people_lookup, 
            names_lookup, 
            players_info,
            fullname_list
        )
        
        if metadata:
            matched_count += 1
            results.append({
                'player_name': player_name,
                'player_id': player_id,
                'batting_style': metadata['batting_style'],
                'bowling_style': metadata['bowling_style'],
                'position': metadata['position'],
                'country': metadata['country']
            })
        else:
            # Add with empty metadata
            results.append({
                'player_name': player_name,
                'player_id': player_id,
                'batting_style': '',
                'bowling_style': '',
                'position': '',
                'country': ''
            })
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(player_mapping)} players...")
    
    print(f"\nMatched {matched_count}/{len(player_mapping)} players ({100*matched_count/len(player_mapping):.1f}%)")
    
    return pd.DataFrame(results)


def main():
    """Main entry point for creating player metadata."""
    print("=" * 60)
    print("Player Metadata Preprocessing")
    print("=" * 60)
    
    # Create metadata
    metadata_df = create_player_metadata()
    
    # Save to CSV
    metadata_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    
    # Show sample
    print("\nSample output:")
    print(metadata_df.head(10).to_string())
    
    # Show statistics
    print("\nMetadata Statistics:")
    print(f"  Total players: {len(metadata_df)}")
    print(f"  With batting style: {metadata_df['batting_style'].notna().sum()}")
    print(f"  With bowling style: {metadata_df['bowling_style'].notna().sum()}")
    print(f"  With position: {metadata_df['position'].notna().sum()}")
    
    # Show batting style distribution
    print("\nBatting Style Distribution:")
    print(metadata_df['batting_style'].value_counts().head(10))
    
    # Show bowling style distribution  
    print("\nBowling Style Distribution:")
    print(metadata_df['bowling_style'].value_counts().head(10))


if __name__ == "__main__":
    main()

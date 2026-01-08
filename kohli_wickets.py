"""
Find which bowler took the most wickets of Virat Kohli in IPL.
"""

import pandas as pd

# Load the data
df = pd.read_csv('ipl_combined.csv')

# Find all rows where Kohli was dismissed
# First, let's find the exact name used for Kohli in the dataset
kohli_names = df[df['player_dismissed'].str.contains('V Kohli', case=False, na=False)]['player_dismissed'].unique()
print("Kohli name variations in data:", kohli_names)

# Filter for rows where Kohli was dismissed
kohli_dismissals = df[df['player_dismissed'].str.contains('V Kohli', case=False, na=False)]

print(f"\nTotal times Kohli was dismissed: {len(kohli_dismissals)}")

# Count wickets by each bowler
bowler_wickets = kohli_dismissals['bowler'].value_counts()

print("\n=== Bowlers who dismissed V Kohli ===")
print(bowler_wickets.to_string())

print("\n" + "="*50)
print(f"🏏 ANSWER: {bowler_wickets.index[0]} took the most wickets of V Kohli!")
print(f"   Total wickets: {bowler_wickets.iloc[0]}")
print("="*50)

# Also show the types of dismissals by the top bowler
top_bowler = bowler_wickets.index[0]
top_bowler_dismissals = kohli_dismissals[kohli_dismissals['bowler'] == top_bowler]
print(f"\nDismissal types by {top_bowler}:")
print(top_bowler_dismissals['wicket_type'].value_counts().to_string())

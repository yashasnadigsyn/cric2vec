"""
Baseline Analysis for Cricket2Vec.
Computes class distribution and majority-class baseline accuracy.
"""

from collections import Counter
import numpy as np

from config import COMBINED_PARQUET, PLAYER_MAPPING
from dataset import CricketDataset


def analyze_class_distribution():
    """Analyze class distribution and compute baseline accuracy."""
    print("Loading dataset...")
    dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING)
    
    # Count occurrences of each class
    outcome_ids = dataset.outcome_ids.numpy()
    counts = Counter(outcome_ids)
    total = len(outcome_ids)
    
    print(f"\nTotal samples: {total:,}")
    print(f"Number of outcome classes: {dataset.get_num_outcomes()}")
    
    # Print class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"{'Class':<15} {'Outcome':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 60)
    
    for class_id, count in sorted(counts.items(), key=lambda x: -x[1]):
        outcome_name = dataset.id_to_outcome[class_id]
        percentage = 100 * count / total
        print(f"{class_id:<15} {outcome_name:<15} {count:>10,} {percentage:>11.2f}%")
    
    # Compute majority class baseline
    majority_class_id, majority_count = counts.most_common(1)[0]
    majority_class_name = dataset.id_to_outcome[majority_class_id]
    majority_baseline = 100 * majority_count / total
    
    print("\n" + "=" * 60)
    print("BASELINE METRICS")
    print("=" * 60)
    print(f"Majority class: {majority_class_name} (id={majority_class_id})")
    print(f"Majority class count: {majority_count:,}")
    print(f"Majority class baseline accuracy: {majority_baseline:.2f}%")
    
    # Compute random baseline
    random_baseline = 100 / dataset.get_num_outcomes()
    print(f"Random guess baseline accuracy: {random_baseline:.2f}%")
    
    # Compute weighted random baseline (based on class frequencies)
    class_probs = np.array([counts[i] / total for i in range(dataset.get_num_outcomes())])
    weighted_random = 100 * np.sum(class_probs ** 2)  # P(correct) = sum(p_i^2)
    print(f"Weighted random baseline accuracy: {weighted_random:.2f}%")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Any model should beat the majority baseline of {majority_baseline:.2f}%")
    print(f"If model accuracy is close to {majority_baseline:.2f}%, it's just predicting the majority class.")
    
    # Identify rare classes (wickets)
    print("\n" + "=" * 60)
    print("RARE CLASSES (< 5%)")
    print("=" * 60)
    for class_id, count in counts.items():
        percentage = 100 * count / total
        if percentage < 5:
            outcome_name = dataset.id_to_outcome[class_id]
            print(f"  {outcome_name}: {percentage:.2f}% ({count:,} samples)")


if __name__ == "__main__":
    analyze_class_distribution()

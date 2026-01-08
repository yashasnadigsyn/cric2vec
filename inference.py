"""
Cric2Vec Inference Engine

Provides advanced analysis of cricket player embeddings including:
- Matchup simulations
- Player similarity search  
- Player clustering (K-Means)
- Player algebra (vector arithmetic)
- Style doppelgängers
- Matchup heatmaps
- Dream XI generator
- Interactive quiz
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rapidfuzz import fuzz, process

from config import (
    CHECKPOINT_DIR, PLAYER_MAPPING, OUTCOME_MAPPING, 
    MODEL_CONFIG, DEVICE
)
from model import Cricket2Vec
from mapping import load_player_mapping, load_outcome_mapping


# =============================================================================
# Archetype Definitions (Verified player names from dataset)
# =============================================================================

BATTER_ARCHETYPES: Dict[str, List[str]] = {
    'anchor': ['V Kohli', 'KL Rahul', 'S Dhawan', 'AM Rahane'],
    'aggressor': ['AB de Villiers', 'CH Gayle', 'GJ Maxwell', 'DA Warner'],
    'finisher': ['MS Dhoni', 'KA Pollard', 'HH Pandya', 'AD Russell', 'RA Jadeja'],
    'opener': ['RG Sharma', 'DA Warner', 'S Dhawan', 'CH Gayle'],
    'accumulator': ['V Kohli', 'KL Rahul', 'AM Rahane'],
}

BOWLER_ARCHETYPES: Dict[str, List[str]] = {
    'death_specialist': ['SL Malinga', 'DJ Bravo', 'JJ Bumrah', 'B Kumar'],
    'powerplay_pacer': ['B Kumar', 'TA Boult', 'Mohammed Shami', 'JJ Bumrah'],
    'spinner': ['R Ashwin', 'Harbhajan Singh', 'SP Narine', 'YS Chahal', 'Rashid Khan'],
    'all_rounder': ['AD Russell', 'HH Pandya', 'RA Jadeja', 'DJ Bravo'],
}


class CricInsights:
    """
    Main inference engine for Cric2Vec model.
    
    Provides methods for:
    - Player matchup analysis
    - Similarity search
    - Clustering and visualization
    - Player algebra
    - Style doppelgängers
    - Team building
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, uses latest.
        """
        self.device = DEVICE
        print(f"Loading Inference Engine on {self.device}...")
        
        # 1. Load Mappings
        self.player_to_id: Dict[str, int] = load_player_mapping(PLAYER_MAPPING)
        self.id_to_player: Dict[int, str] = {v: k for k, v in self.player_to_id.items()}
        self._player_names: List[str] = list(self.player_to_id.keys())
        
        outcome_map = load_outcome_mapping(OUTCOME_MAPPING)
        self.outcome_to_id: Dict[str, int] = outcome_map['outcome_to_id']
        self.id_to_outcome: Dict[int, str] = {int(k): v for k, v in outcome_map['id_to_outcome'].items()}
        
        self.num_players: int = len(self.player_to_id)
        self.num_outcomes: int = len(self.outcome_to_id)
        
        # 2. Load Model
        self.model = Cricket2Vec(
            self.num_players, 
            self.num_outcomes, 
            MODEL_CONFIG['embedding_dim']
        ).to(self.device)
        
        if checkpoint_path is None:
            checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found! Train the model first.")
            checkpoint_path = checkpoints[-1]
            
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 3. Cache embeddings for faster operations
        with torch.no_grad():
            all_ids = torch.arange(self.num_players).to(self.device)
            self.bat_embeddings: np.ndarray = self.model.get_batter_embeddings(all_ids).cpu().numpy()
            self.bowl_embeddings: np.ndarray = self.model.get_bowler_embeddings(all_ids).cpu().numpy()
        
        # 4. Player metadata (lazy loaded)
        self._player_metadata: Optional[pd.DataFrame] = None
        
        print(f"Loaded {self.num_players} players, {self.num_outcomes} outcomes")

    # =========================================================================
    # Core Player ID Methods
    # =========================================================================
    
    def get_player_id(self, name: str) -> int:
        """
        Get player ID from name.
        
        Args:
            name: Player name (exact match required)
            
        Returns:
            Player ID
            
        Raises:
            ValueError: If player not found
        """
        if name not in self.player_to_id:
            raise ValueError(f"Player '{name}' not found. Use find_player() for fuzzy search.")
        return self.player_to_id[name]
    
    def find_player(
        self, 
        partial_name: str, 
        top_k: int = 5,
        threshold: int = 60
    ) -> List[Tuple[str, int]]:
        """
        Find players matching a partial or misspelled name using fuzzy matching.
        
        Args:
            partial_name: Partial or approximate player name
            top_k: Number of matches to return
            threshold: Minimum similarity score (0-100)
            
        Returns:
            List of (player_name, similarity_score) tuples
        """
        results = process.extract(
            partial_name, 
            self._player_names, 
            scorer=fuzz.ratio,
            limit=top_k
        )
        return [(name, score) for name, score, _ in results if score >= threshold]
    
    def get_player_id_fuzzy(self, name: str, threshold: int = 80) -> int:
        """
        Get player ID with fuzzy matching fallback.
        
        Args:
            name: Player name (exact or approximate)
            threshold: Minimum similarity score for fuzzy match
            
        Returns:
            Player ID
            
        Raises:
            ValueError: If no match found above threshold
        """
        if name in self.player_to_id:
            return self.player_to_id[name]
        
        matches = self.find_player(name, top_k=1, threshold=threshold)
        if matches:
            matched_name = matches[0][0]
            print(f"Fuzzy matched '{name}' → '{matched_name}'")
            return self.player_to_id[matched_name]
        
        raise ValueError(f"No player found matching '{name}' (threshold={threshold})")

    # =========================================================================
    # Matchup Analysis
    # =========================================================================
    
    def get_matchup_probs(
        self, 
        batter_name: str, 
        bowler_name: str
    ) -> Dict[str, float]:
        """
        Get probability distribution for a specific batter-bowler matchup.
        
        Args:
            batter_name: Name of the batter
            bowler_name: Name of the bowler
            
        Returns:
            Dict mapping outcome names to probabilities
        """
        bat_id = self.get_player_id_fuzzy(batter_name)
        bowl_id = self.get_player_id_fuzzy(bowler_name)
        
        bat_tensor = torch.tensor([bat_id]).to(self.device)
        bowl_tensor = torch.tensor([bowl_id]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(bat_tensor, bowl_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        return {self.id_to_outcome[i]: float(p) for i, p in enumerate(probs)}

    def simulate_over(
        self, 
        batter_name: str, 
        bowler_name: str
    ) -> List[str]:
        """
        Simulate an over (6 balls) between batter and bowler.
        
        Args:
            batter_name: Name of the batter
            bowler_name: Name of the bowler
            
        Returns:
            List of outcome strings for each ball
        """
        probs_dict = self.get_matchup_probs(batter_name, bowler_name)
        outcomes = list(probs_dict.keys())
        probabilities = list(probs_dict.values())
        
        print(f"\n--- Simulation: {batter_name} vs {bowler_name} ---")
        total_runs = 0
        wickets = 0
        
        events = random.choices(outcomes, weights=probabilities, k=6)
        
        for i, event in enumerate(events):
            print(f"Ball {i+1}: {event}")
            
            if event.startswith('w_'):
                wickets += 1
            elif event == '0_run':
                pass
            elif event == '1_run':
                total_runs += 1
            elif event == '2_3_run':
                total_runs += 2
            elif event == '4_run':
                total_runs += 4
            elif event == '6_run':
                total_runs += 6
            elif event == 'extras':
                total_runs += 1
                
        print(f"Over Summary: {total_runs}/{wickets}")
        return events

    # =========================================================================
    # Similarity Search
    # =========================================================================
    
    def find_similar_players(
        self, 
        player_name: str, 
        role: str = 'bat', 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find players with similar embeddings.
        
        Args:
            player_name: Name of the reference player
            role: 'bat' for batter embeddings, 'bowl' for bowler
            top_k: Number of similar players to return
            
        Returns:
            List of (player_name, similarity_score) tuples
        """
        pid = self.get_player_id_fuzzy(player_name)
        
        if role == 'bat':
            embeddings = self.bat_embeddings
            target_vec = self.bat_embeddings[pid].reshape(1, -1)
        else:
            embeddings = self.bowl_embeddings
            target_vec = self.bowl_embeddings[pid].reshape(1, -1)
            
        sims = cosine_similarity(target_vec, embeddings)[0]
        top_indices = sims.argsort()[::-1][1:top_k+1]
        
        print(f"\n--- Players similar to {player_name} ({role}) ---")
        results = []
        for idx in top_indices:
            name = self.id_to_player[idx]
            score = float(sims[idx])
            print(f"{name}: {score:.4f}")
            results.append((name, score))
        return results

    def find_bunnies(
        self, 
        bowler_name: str, 
        top_k: int = 5, 
        min_prob: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find batters most likely to get OUT against this bowler.
        
        Args:
            bowler_name: Name of the bowler
            top_k: Number of batters to return
            min_prob: Minimum wicket probability threshold
            
        Returns:
            List of (batter_name, wicket_probability) tuples
        """
        bowl_id = self.get_player_id_fuzzy(bowler_name)
        
        wicket_indices = [idx for outcome, idx in self.outcome_to_id.items() 
                          if outcome.startswith('w_')]
        
        if not wicket_indices:
            print("No wicket outcomes found in mapping!")
            return []
        
        all_bat_ids = torch.arange(self.num_players).to(self.device)
        bowl_ids_repeated = torch.full((self.num_players,), bowl_id, device=self.device)
        
        with torch.no_grad():
            logits = self.model(all_bat_ids, bowl_ids_repeated)
            probs = F.softmax(logits, dim=1)
            
        wicket_probs = probs[:, wicket_indices].sum(dim=1).cpu().numpy()
        top_indices = wicket_probs.argsort()[::-1][:top_k]
        
        print(f"\n--- Bunnies for {bowler_name} (High Wicket Prob) ---")
        results = []
        for idx in top_indices:
            prob = float(wicket_probs[idx])
            if prob > min_prob:
                name = self.id_to_player[idx]
                print(f"{name}: {prob:.4f}")
                results.append((name, prob))
        return results

    # =========================================================================
    # K-Means Clustering (Section 7.2)
    # =========================================================================
    
    def cluster_players(
        self, 
        role: str = 'bat', 
        n_clusters: int = 6
    ) -> Tuple[Dict[int, List[str]], np.ndarray]:
        """
        K-means clustering on player embeddings.
        
        Args:
            role: 'bat' or 'bowl'
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_dict, centroids) where cluster_dict maps
            cluster_id to list of player names
        """
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(self.id_to_player[idx])
        
        print(f"\n--- K-Means Clustering ({role}, k={n_clusters}) ---")
        for cluster_id, players in clusters.items():
            print(f"Cluster {cluster_id}: {len(players)} players")
            print(f"  Sample: {players[:5]}")
        
        return clusters, kmeans.cluster_centers_
    
    def plot_elbow(
        self, 
        role: str = 'bat', 
        max_k: int = 12, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot elbow curve to find optimal k for clustering.
        
        Args:
            role: 'bat' or 'bowl'
            max_k: Maximum k to test
            save_path: Path to save the plot (optional)
        """
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title(f'Elbow Method for {role.capitalize()} Embeddings', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"plots/elbow_{role}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved elbow plot to {save_path}")
    
    def plot_clustered_tsne(
        self, 
        role: str = 'bat', 
        n_clusters: int = 6, 
        annotate_top: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        t-SNE visualization with cluster colors.
        
        Args:
            role: 'bat' or 'bowl'
            n_clusters: Number of clusters
            annotate_top: Number of players to annotate per cluster
            save_path: Path to save the plot
        """
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # t-SNE
        print(f"Running t-SNE for {role} embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(16, 12))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], 
            c=labels, cmap='tab10', alpha=0.6, s=50
        )
        plt.colorbar(scatter, label='Cluster')
        
        # Annotate top players per cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0][:annotate_top // n_clusters]
            for idx in cluster_indices:
                plt.annotate(
                    self.id_to_player[idx], 
                    (reduced[idx, 0], reduced[idx, 1]),
                    fontsize=7, alpha=0.8
                )
        
        plt.title(f't-SNE Visualization of {role.capitalize()} Embeddings (K={n_clusters})', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"plots/tsne_clustered_{role}_k{n_clusters}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved clustered t-SNE plot to {save_path}")

    # =========================================================================
    # Player Algebra (Section 7.1)
    # =========================================================================
    
    def get_archetype_vector(
        self, 
        archetype: str, 
        role: str = 'bat'
    ) -> np.ndarray:
        """
        Get the average embedding for a player archetype.
        
        Args:
            archetype: One of the defined archetype names (e.g., 'anchor', 'finisher')
            role: 'bat' or 'bowl'
            
        Returns:
            Average embedding vector for the archetype
        """
        archetypes = BATTER_ARCHETYPES if role == 'bat' else BOWLER_ARCHETYPES
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        if archetype not in archetypes:
            available = list(archetypes.keys())
            raise ValueError(f"Unknown archetype '{archetype}'. Available: {available}")
        
        exemplar_names = archetypes[archetype]
        exemplar_ids = []
        
        for name in exemplar_names:
            if name in self.player_to_id:
                exemplar_ids.append(self.player_to_id[name])
            else:
                print(f"Warning: Exemplar '{name}' not found in dataset")
        
        if not exemplar_ids:
            raise ValueError(f"No valid exemplars found for archetype '{archetype}'")
        
        exemplar_embeddings = embeddings[exemplar_ids]
        return exemplar_embeddings.mean(axis=0)
    
    def player_algebra(
        self, 
        positive: List[str], 
        negative: List[str], 
        role: str = 'bat', 
        top_k: int = 5,
        exclude_input: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Compute vector arithmetic: sum(positive) - sum(negative).
        Return nearest players to the resulting vector.
        
        Args:
            positive: List of player names or archetype names to add
            negative: List of player names or archetype names to subtract
            role: 'bat' or 'bowl'
            top_k: Number of results to return
            exclude_input: Whether to exclude input players from results
            
        Returns:
            List of (player_name, similarity_score) tuples
            
        Example:
            >>> engine.player_algebra(['V Kohli'], ['anchor'], role='bat')
            # Returns players similar to "V Kohli minus anchor-ness"
        """
        archetypes = BATTER_ARCHETYPES if role == 'bat' else BOWLER_ARCHETYPES
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        # Build result vector
        result_vec = np.zeros(embeddings.shape[1])
        exclude_ids = set()
        
        # Add positive vectors
        for item in positive:
            if item in archetypes:
                result_vec += self.get_archetype_vector(item, role)
            elif item in self.player_to_id:
                pid = self.player_to_id[item]
                result_vec += embeddings[pid]
                if exclude_input:
                    exclude_ids.add(pid)
            else:
                # Try fuzzy match
                matches = self.find_player(item, top_k=1, threshold=80)
                if matches:
                    name = matches[0][0]
                    pid = self.player_to_id[name]
                    result_vec += embeddings[pid]
                    if exclude_input:
                        exclude_ids.add(pid)
                    print(f"Fuzzy matched '{item}' → '{name}'")
                else:
                    raise ValueError(f"Unknown player or archetype: '{item}'")
        
        # Subtract negative vectors
        for item in negative:
            if item in archetypes:
                result_vec -= self.get_archetype_vector(item, role)
            elif item in self.player_to_id:
                pid = self.player_to_id[item]
                result_vec -= embeddings[pid]
                if exclude_input:
                    exclude_ids.add(pid)
            else:
                matches = self.find_player(item, top_k=1, threshold=80)
                if matches:
                    name = matches[0][0]
                    pid = self.player_to_id[name]
                    result_vec -= embeddings[pid]
                    if exclude_input:
                        exclude_ids.add(pid)
                else:
                    raise ValueError(f"Unknown player or archetype: '{item}'")
        
        # Find nearest neighbors
        result_vec = result_vec.reshape(1, -1)
        sims = cosine_similarity(result_vec, embeddings)[0]
        
        # Mask excluded players
        for pid in exclude_ids:
            sims[pid] = -1
        
        top_indices = sims.argsort()[::-1][:top_k]
        
        print(f"\n--- Player Algebra: {positive} - {negative} ---")
        results = []
        for idx in top_indices:
            name = self.id_to_player[idx]
            score = float(sims[idx])
            print(f"{name}: {score:.4f}")
            results.append((name, score))
        
        return results

    # =========================================================================
    # Player Metadata and Doppelgängers (Section 7.3)
    # =========================================================================
    
    def load_player_metadata(
        self, 
        filepath: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load player metadata with batting/bowling styles.
        
        Args:
            filepath: Path to player_metadata.csv
            
        Returns:
            DataFrame with player metadata
        """
        if self._player_metadata is not None:
            return self._player_metadata
        
        if filepath is None:
            filepath = Path(__file__).parent / "player_metadata.csv"
        
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Player metadata not found at {filepath}. "
                "Run 'python player_metadata.py' first."
            )
        
        self._player_metadata = pd.read_csv(filepath)
        return self._player_metadata
    
    def find_doppelganger(
        self, 
        player_name: str,
        from_style: str,
        to_style: str,
        role: str = 'bat',
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find player's doppelgänger with different handedness/style.
        
        Uses formula: player_vec - avg(from_style_players) + avg(to_style_players)
        
        Args:
            player_name: Name of the reference player
            from_style: Current style (e.g., 'right-hand-bat')
            to_style: Target style (e.g., 'left-hand-bat')
            role: 'bat' or 'bowl'
            top_k: Number of results
            
        Returns:
            List of (player_name, similarity) tuples
        """
        metadata = self.load_player_metadata()
        style_col = 'batting_style' if role == 'bat' else 'bowling_style'
        embeddings = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        # Get player embedding
        pid = self.get_player_id_fuzzy(player_name)
        player_vec = embeddings[pid]
        
        # Get average vectors for styles
        from_players = metadata[metadata[style_col] == from_style]['player_name'].tolist()
        to_players = metadata[metadata[style_col] == to_style]['player_name'].tolist()
        
        from_ids = [self.player_to_id[n] for n in from_players if n in self.player_to_id]
        to_ids = [self.player_to_id[n] for n in to_players if n in self.player_to_id]
        
        if not from_ids or not to_ids:
            raise ValueError(f"No players found for styles '{from_style}' or '{to_style}'")
        
        from_vec = embeddings[from_ids].mean(axis=0)
        to_vec = embeddings[to_ids].mean(axis=0)
        
        # Compute doppelgänger vector
        doppel_vec = player_vec - from_vec + to_vec
        doppel_vec = doppel_vec.reshape(1, -1)
        
        # Find nearest in target style
        sims = cosine_similarity(doppel_vec, embeddings)[0]
        
        # Mask non-target-style players
        target_ids = set(to_ids)
        for i in range(len(sims)):
            if i not in target_ids:
                sims[i] = -1
        
        top_indices = sims.argsort()[::-1][:top_k]
        
        print(f"\n--- Doppelgänger: {player_name} ({from_style} → {to_style}) ---")
        results = []
        for idx in top_indices:
            if sims[idx] > 0:
                name = self.id_to_player[idx]
                score = float(sims[idx])
                print(f"{name}: {score:.4f}")
                results.append((name, score))
        
        return results

    # =========================================================================
    # Matchup Heatmap (Section 7.5)
    # =========================================================================
    
    def matchup_heatmap(
        self,
        batters: List[str],
        bowlers: List[str],
        outcome: str = 'wicket',
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a heatmap of outcome probabilities for batter-bowler pairs.
        
        Args:
            batters: List of batter names
            bowlers: List of bowler names
            outcome: 'wicket', 'boundary' (4+6), or 'dot' (0_run)
            save_path: Path to save the heatmap
            
        Returns:
            2D array of probabilities [batters x bowlers]
        """
        # Resolve outcome indices
        if outcome == 'wicket':
            outcome_indices = [idx for o, idx in self.outcome_to_id.items() if o.startswith('w_')]
        elif outcome == 'boundary':
            outcome_indices = [self.outcome_to_id.get('4_run', -1), self.outcome_to_id.get('6_run', -1)]
            outcome_indices = [i for i in outcome_indices if i >= 0]
        elif outcome == 'dot':
            outcome_indices = [self.outcome_to_id.get('0_run', -1)]
            outcome_indices = [i for i in outcome_indices if i >= 0]
        else:
            raise ValueError(f"Unknown outcome type: {outcome}")
        
        # Get IDs
        bat_ids = [self.get_player_id_fuzzy(b) for b in batters]
        bowl_ids = [self.get_player_id_fuzzy(b) for b in bowlers]
        
        # Compute probabilities
        probs = np.zeros((len(batters), len(bowlers)))
        
        for i, bat_id in enumerate(bat_ids):
            for j, bowl_id in enumerate(bowl_ids):
                bat_tensor = torch.tensor([bat_id]).to(self.device)
                bowl_tensor = torch.tensor([bowl_id]).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(bat_tensor, bowl_tensor)
                    all_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                probs[i, j] = sum(all_probs[idx] for idx in outcome_indices)
        
        # Plot heatmap
        plt.figure(figsize=(max(12, len(bowlers)), max(10, len(batters) * 0.5)))
        sns.heatmap(
            probs, 
            xticklabels=bowlers, 
            yticklabels=batters,
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn_r' if outcome == 'wicket' else 'RdYlGn',
            vmin=0, vmax=1
        )
        plt.title(f'{outcome.capitalize()} Probability Heatmap', fontsize=14)
        plt.xlabel('Bowlers')
        plt.ylabel('Batters')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"plots/heatmap_{outcome}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {save_path}")
        
        return probs

    # =========================================================================
    # Dream XI Generator (Section 7.4)
    # =========================================================================
    
    def build_dream_team(
        self,
        captain_name: str,
        team_size: int = 11,
        min_batters: int = 5,
        min_bowlers: int = 4,
        max_per_cluster: int = 2
    ) -> List[str]:
        """
        Build a diverse team around a captain using farthest-point sampling.
        
        Args:
            captain_name: Name of the team captain
            team_size: Total team size
            min_batters: Minimum number of batters
            min_bowlers: Minimum number of bowlers
            max_per_cluster: Max players from same cluster
            
        Returns:
            List of player names forming the team
        """
        # Get captain
        cap_id = self.get_player_id_fuzzy(captain_name)
        team = [captain_name]
        used_ids = {cap_id}
        
        # Cluster players
        bat_clusters, _ = self.cluster_players('bat', n_clusters=6)
        bowl_clusters, _ = self.cluster_players('bowl', n_clusters=6)
        
        # Build cluster membership lookup
        bat_cluster_of = {}
        for cid, players in bat_clusters.items():
            for p in players:
                bat_cluster_of[p] = cid
        
        bowl_cluster_of = {}
        for cid, players in bowl_clusters.items():
            for p in players:
                bowl_cluster_of[p] = cid
        
        # Track cluster usage
        bat_cluster_count = {i: 0 for i in range(6)}
        bowl_cluster_count = {i: 0 for i in range(6)}
        
        # Add batters using diversity
        print(f"\n--- Building Dream Team around {captain_name} ---")
        batters_added = 1  # Captain counts
        
        while batters_added < min_batters and len(team) < team_size:
            # Find farthest batter from current team
            best_id = None
            best_min_sim = 1.0
            
            for i in range(self.num_players):
                if i in used_ids:
                    continue
                    
                player_name = self.id_to_player[i]
                cluster = bat_cluster_of.get(player_name, 0)
                
                if bat_cluster_count[cluster] >= max_per_cluster:
                    continue
                
                # Compute min similarity to team
                min_sim = min(
                    cosine_similarity(
                        self.bat_embeddings[i].reshape(1, -1),
                        self.bat_embeddings[list(used_ids)]
                    ).min()
                    for _ in [1]  # Dummy loop for expression
                )
                
                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_id = i
            
            if best_id is not None:
                name = self.id_to_player[best_id]
                team.append(name)
                used_ids.add(best_id)
                bat_cluster_count[bat_cluster_of.get(name, 0)] += 1
                batters_added += 1
        
        # Add bowlers
        bowlers_added = 0
        while bowlers_added < min_bowlers and len(team) < team_size:
            best_id = None
            best_min_sim = 1.0
            
            for i in range(self.num_players):
                if i in used_ids:
                    continue
                    
                player_name = self.id_to_player[i]
                cluster = bowl_cluster_of.get(player_name, 0)
                
                if bowl_cluster_count[cluster] >= max_per_cluster:
                    continue
                
                vec = self.bowl_embeddings[i].reshape(1, -1)
                team_bowl_ids = [self.player_to_id[n] for n in team if n in self.player_to_id]
                if team_bowl_ids:
                    sims = cosine_similarity(vec, self.bowl_embeddings[team_bowl_ids])
                    min_sim = sims.min()
                else:
                    min_sim = 0
                
                if min_sim < best_min_sim:
                    best_min_sim = min_sim
                    best_id = i
            
            if best_id is not None:
                name = self.id_to_player[best_id]
                team.append(name)
                used_ids.add(best_id)
                bowl_cluster_count[bowl_cluster_of.get(name, 0)] += 1
                bowlers_added += 1
        
        print(f"Dream Team ({len(team)} players):")
        for i, p in enumerate(team):
            captain_marker = " (C)" if i == 0 else ""
            print(f"  {i+1}. {p}{captain_marker}")
        
        return team

    # =========================================================================
    # Quiz Feature (Section 7.8)
    # =========================================================================
    
    def who_would_you_face_quiz(
        self,
        player_name: str,
        role: str = 'bowl',
        rounds: int = 5
    ) -> None:
        """
        CLI interactive quiz: guess which bowler is 'easier' based on wicket probability.
        
        Args:
            player_name: Name of the batter (or bowler if role='bat')
            role: 'bowl' = compare bowlers, 'bat' = compare batters
            rounds: Number of quiz rounds
        """
        pid = self.get_player_id_fuzzy(player_name)
        
        print(f"\n{'='*60}")
        print(f"WHO WOULD YOU RATHER FACE? Quiz")
        print(f"{'='*60}")
        print(f"Player: {player_name}")
        print(f"Guess which opponent has LOWER wicket probability!")
        print()
        
        score = 0
        
        for round_num in range(1, rounds + 1):
            # Pick two random opponents
            opponents = random.sample(range(self.num_players), 2)
            while pid in opponents:
                opponents = random.sample(range(self.num_players), 2)
            
            opp1_name = self.id_to_player[opponents[0]]
            opp2_name = self.id_to_player[opponents[1]]
            
            # Calculate wicket probs
            wicket_indices = [idx for o, idx in self.outcome_to_id.items() if o.startswith('w_')]
            
            probs = []
            for opp_id in opponents:
                if role == 'bowl':
                    bat_t = torch.tensor([pid]).to(self.device)
                    bowl_t = torch.tensor([opp_id]).to(self.device)
                else:
                    bat_t = torch.tensor([opp_id]).to(self.device)
                    bowl_t = torch.tensor([pid]).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(bat_t, bowl_t)
                    all_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                wicket_prob = sum(all_probs[idx] for idx in wicket_indices)
                probs.append(wicket_prob)
            
            print(f"Round {round_num}/{rounds}:")
            print(f"  A) {opp1_name}")
            print(f"  B) {opp2_name}")
            
            answer = input("Who is EASIER to face? (A/B): ").strip().upper()
            
            correct_idx = 0 if probs[0] < probs[1] else 1
            correct_letter = 'A' if correct_idx == 0 else 'B'
            user_idx = 0 if answer == 'A' else 1
            
            if user_idx == correct_idx:
                print(f"  ✓ Correct! {['A','B'][correct_idx]} has {probs[correct_idx]:.1%} wicket prob")
                score += 1
            else:
                print(f"  ✗ Wrong! {correct_letter} was easier ({probs[correct_idx]:.1%} vs {probs[1-correct_idx]:.1%})")
            print()
        
        print(f"{'='*60}")
        print(f"Final Score: {score}/{rounds}")
        if score == rounds:
            print("🏆 Perfect score!")
        elif score >= rounds * 0.7:
            print("⭐ Great job!")
        else:
            print("Keep practicing!")

    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_embeddings(
        self, 
        role: str = 'bat', 
        num_points: int = 300,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize embeddings using t-SNE.
        
        Args:
            role: 'bat' or 'bowl'
            num_points: Number of random players to plot
            save_path: Path to save the plot
        """
        print(f"\nGenerating t-SNE for {role} embeddings...")
        
        emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
        
        indices = np.random.choice(
            self.num_players, 
            min(num_points, self.num_players), 
            replace=False
        )
        subset_emb = emb[indices]
        names = [self.id_to_player[i] for i in indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(subset_emb)
        
        plt.figure(figsize=(15, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        
        for i, name in enumerate(names):
            plt.annotate(name, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)
            
        plt.title(f"t-SNE Visualization of {role.capitalize()} Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = f"plots/{role}_embeddings_tsne.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    try:
        engine = CricInsights()
        
        print("\n" + "="*60)
        print("CRIC2VEC INFERENCE DEMO")
        print("="*60)
        
        # 1. Fuzzy Player Search
        print("\n--- 1. Fuzzy Player Search ---")
        matches = engine.find_player("Kohli")
        print(f"Searching 'Kohli': {matches}")
        
        # 2. Matchup Simulation
        print("\n--- 2. Matchup Simulation ---")
        try:
            engine.simulate_over("V Kohli", "Sandeep Sharma")
        except ValueError as e:
            print(e)

        # 3. Similarity Search
        print("\n--- 3. Similarity Search ---")
        try:
            engine.find_similar_players("V Kohli", role='bat')
        except ValueError as e:
            print(e)
            
        # 4. Bunnies
        print("\n--- 4. Bunny Finder ---")
        try:
            engine.find_bunnies("SP Narine", top_k=5)
        except ValueError as e:
            print(e)
        
        # 5. Player Algebra
        print("\n--- 5. Player Algebra ---")
        try:
            # "Kohli minus anchor-ness" = more aggressive version of Kohli
            engine.player_algebra(['V Kohli'], ['anchor'], role='bat', top_k=5)
        except ValueError as e:
            print(e)
        
        # 6. Clustering
        print("\n--- 6. K-Means Clustering ---")
        engine.cluster_players(role='bat', n_clusters=5)
        engine.plot_elbow(role='bat', max_k=10)
        
        # 7. Visualization
        print("\n--- 7. Visualization ---")
        engine.plot_embeddings(role='bat')
        engine.plot_clustered_tsne(role='bat', n_clusters=5)
        
        print("\n" + "="*60)
        print("Demo complete! Check 'plots/' directory for visualizations.")
        print("="*60)

        engine.who_would_you_face_quiz(player_name="V Kohli", role='bat', rounds=5)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

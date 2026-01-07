
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import random
import glob

from config import (
    CHECKPOINT_DIR, PLAYER_MAPPING, OUTCOME_MAPPING, 
    MODEL_CONFIG, DEVICE
)
from model import Cricket2Vec
from mapping import load_player_mapping, load_outcome_mapping

class CricInsights:
    def __init__(self, checkpoint_path=None):
        self.device = DEVICE
        print(f"Loading Inference Engine on {self.device}...")
        
        # 1. Load Mappings
        self.player_to_id = load_player_mapping(PLAYER_MAPPING)
        self.id_to_player = {v: k for k, v in self.player_to_id.items()}
        
        outcome_map = load_outcome_mapping(OUTCOME_MAPPING)
        self.outcome_to_id = outcome_map['outcome_to_id']
        self.id_to_outcome = {int(k): v for k, v in outcome_map['id_to_outcome'].items()}
        
        self.num_players = len(self.player_to_id)
        self.num_outcomes = len(self.outcome_to_id)
        
        # 2. Load Model
        self.model = Cricket2Vec(
            self.num_players, 
            self.num_outcomes, 
            MODEL_CONFIG['embedding_dim']
        ).to(self.device)
        
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found! Train the model first.")
            checkpoint_path = checkpoints[-1]
            
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Cache embeddings for faster similarity search
        self.bat_embeddings = self.model.bat_embedding.weight.detach().cpu().numpy()
        self.bowl_embeddings = self.model.bowl_embedding.weight.detach().cpu().numpy()

    def get_player_id(self, name):
        if name not in self.player_to_id:
            # Fuzzy match or error
            raise ValueError(f"Player '{name}' not found in mapping.")
        return self.player_to_id[name]

    def get_matchup_probs(self, batter_name, bowler_name):
        """Get probability distribution for a specific matchup."""
        bat_id = self.get_player_id(batter_name)
        bowl_id = self.get_player_id(bowler_name)
        
        bat_tensor = torch.tensor([bat_id]).to(self.device)
        bowl_tensor = torch.tensor([bowl_id]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(bat_tensor, bowl_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        return {self.id_to_outcome[i]: float(p) for i, p in enumerate(probs)}

    def simulate_over(self, batter_name, bowler_name):
        """Simulate an over (6 balls) between batter and bowler."""
        probs_dict = self.get_matchup_probs(batter_name, bowler_name)
        outcomes = list(probs_dict.keys())
        probabilities = list(probs_dict.values())
        
        print(f"\n--- Simulation: {batter_name} vs {bowler_name} ---")
        total_runs = 0
        wickets = 0
        
        events = random.choices(outcomes, weights=probabilities, k=6)
        
        for i, event in enumerate(events):
            print(f"Ball {i+1}: {event}")
            
            # Simple parsing for display stats
            if event == 'W':
                wickets += 1
            elif event.isdigit():
                total_runs += int(event)
                
        print(f"Over Summary: {total_runs}/{wickets}")
        return events

    def find_similar_players(self, player_name, role='bat', top_k=5):
        """Find players with similar embeddings."""
        pid = self.get_player_id(player_name)
        
        if role == 'bat':
            embeddings = self.bat_embeddings
            target_vec = self.bat_embeddings[pid].reshape(1, -1)
        else:
            embeddings = self.bowl_embeddings
            target_vec = self.bowl_embeddings[pid].reshape(1, -1)
            
        sims = cosine_similarity(target_vec, embeddings)[0]
        
        # Get indices of top k similar (excluding self)
        # We use [::-1] to reverse sort, then skip index 0 (self)
        top_indices = sims.argsort()[::-1][1:top_k+1]
        
        print(f"\n--- Players similar to {player_name} ({role}) ---")
        results = []
        for idx in top_indices:
            name = self.id_to_player[idx]
            score = sims[idx]
            print(f"{name}: {score:.4f}")
            results.append((name, score))
        return results

    def find_bunnies(self, bowler_name, top_k=5, min_prob=0.0):
        """Find batters most likely to get OUT against this bowler."""
        bowl_id = self.get_player_id(bowler_name)
        
        # We need the ID for 'W' (Wicket) outcome
        # Adjust this key if your outcome mapping uses different string for wicket
        wicket_key = 'W'
        if wicket_key not in self.outcome_to_id:
             # Try to find a logical fallback if 'W' isn't exact key
             # For now, let's assume 'W' exists based on typical convention
             return
             
        wicket_idx = self.outcome_to_id[wicket_key]
        
        # Compute logits for ALL batters against this bowler
        # Shape: [num_players, outcome_dim]
        # We can simulate batch inference
        all_bat_ids = torch.arange(self.num_players).to(self.device)
        bowl_ids_repeated = torch.tensor([bowl_id] * self.num_players).to(self.device)
        
        with torch.no_grad():
            logits = self.model(all_bat_ids, bowl_ids_repeated)
            probs = F.softmax(logits, dim=1) # [num_players, num_outcomes]
            
        wicket_probs = probs[:, wicket_idx].cpu().numpy()
        
        # Top K batters with highest wicket probability
        top_indices = wicket_probs.argsort()[::-1][:top_k]
        
        print(f"\n--- Bunnies for {bowler_name} (High Wicket Prob) ---")
        for idx in top_indices:
            prob = wicket_probs[idx]
            if prob > min_prob:
                print(f"{self.id_to_player[idx]}: {prob:.4f}")

    def plot_embeddings(self, role='bat', num_points=300):
        """Visualize embeddings using t-SNE."""
        print(f"\nGeneratring t-SNE for {role} embeddings...")
        
        if role == 'bat':
            emb = self.bat_embeddings
        else:
            emb = self.bowl_embeddings
            
        # Select random subset to avoid clutter, or top N players if we had stats
        # For now, random subset
        indices = np.random.choice(self.num_players, min(num_points, self.num_players), replace=False)
        subset_emb = emb[indices]
        names = [self.id_to_player[i] for i in indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(subset_emb)
        
        plt.figure(figsize=(15, 10))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        
        # Annotate points
        for i, name in enumerate(names):
            plt.annotate(name, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)
            
        plt.title(f"t-SNE Visualization of {role.capitalize()} Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        
        filename = f"{role}_embeddings_tsne.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")


if __name__ == "__main__":
    try:
        engine = CricInsights()
        
        # --- Examples ---
        
        # 1. Matchup
        print("\n--- 1. Matchup Sim ---")
        try:
            # Using some popular names, might need adjustment if not in dataset
            engine.simulate_over("V Kohli", "JJ Bumrah")
        except ValueError as e:
            print(e)  # print error but continue

        # 2. Similarity
        print("\n--- 2. Similarity Search ---")
        try:
            engine.find_similar_players("V Kohli", role='bat')
            engine.find_similar_players("JJ Bumrah", role='bowl')
        except ValueError as e:
            print(e)
            
        # 3. Bunnies
        print("\n--- 3. Bunny Finder ---")
        try:
            engine.find_bunnies("SP Narine", top_k=5)
        except ValueError as e:
            print(e)
            
        # 4. Visualization
        # engine.plot_embeddings(role='bat')
        
    except Exception as e:
        print(f"Error initializing: {e}")

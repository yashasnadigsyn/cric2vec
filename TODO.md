# cric2vec TODO

> **Last Updated**: January 8, 2026  
> **Current State**: Model achieves ~39% accuracy on 12 outcome classes

Work through this file **top to bottom**. Each section builds on previous ones.

---

## 1. Immediate Bug Fixes (Do First)

### 1.1 Fix Optimizer/Comment Mismatch

**File**: `main.py` (lines 252-283)

**Problem**: Code uses Adam optimizer, but comments and print statements say "SGD with Nesterov". Config defines `momentum: 0.9` that's never used.

**Why it matters**: Confusing for anyone reading the code. Adam and SGD have very different convergence behaviors.

**What to do**:
- [x] Switch to SGD with Nesterov (as per batter_pitcher2vec paper):
  ```python
  optimizer = optim.SGD(
      model.parameters(), 
      lr=TRAINING_CONFIG['learning_rate'],  # 0.01
      momentum=TRAINING_CONFIG['momentum'],  # 0.9
      nesterov=True,
      weight_decay=TRAINING_CONFIG['weight_decay']
  )
  ```

---

### 1.2 Fix Embedding Retrieval in Inference

**File**: `inference.py` (lines 56-57)

**Problem**: 
```python
# WRONG: Uses raw embeddings (pre-sigmoid)
self.bat_embeddings = self.model.bat_embedding.weight.detach().cpu().numpy()
```

But the model applies sigmoid after embedding lookup. This means t-SNE plots and similarity searches use **different embeddings** than what the model actually uses.

**Why it matters**: Similarity analysis becomes misleading. Two players may look similar in raw space but different after sigmoid transformation.

**What to do**:
- [x] Replace with:
  ```python
  with torch.no_grad():
      all_ids = torch.arange(self.num_players).to(self.device)
      self.bat_embeddings = self.model.get_batter_embeddings(all_ids).cpu().numpy()
      self.bowl_embeddings = self.model.get_bowler_embeddings(all_ids).cpu().numpy()
  ```

---

### 1.3 REDO Checkpoint logic

**What to do**:
- [x] Update saving logic in `main.py` to save less frequently (e.g., every 20 or 25 epochs)

---

## 2. Evaluation Improvements (Before Changing Model)

### 2.1 Add Per-Class Metrics

**File**: `main.py` (add after validation loop)

**Problem**: Only tracking overall accuracy (~39%). This hides whether model just predicts common classes (`0_run`, `1_run`).

**Why it matters**: Without class-level metrics, you can't tell if the model learned anything meaningful about rare events (wickets).

**What to do**:
- [x] Add imports:
  ```python
  from sklearn.metrics import classification_report, confusion_matrix
  ```
- [x] After validation loop, add:
  ```python
  from sklearn.metrics import f1_score
  # Collect all predictions and labels during validation
  all_preds = []
  all_labels = []
  # ... in validation loop, append to these lists ...
  
  # After loop:
  f1 = f1_score(all_labels, all_preds, average='weighted')
  print(f"Weighted F1: {f1:.4f}")
  
  # Every N epochs, print full classification report
  if (epoch + 1) % 20 == 0:
      print(classification_report(all_labels, all_preds, 
            target_names=list(full_dataset.id_to_outcome.values())))
  ```

---

### 2.2 Add Confusion Matrix Visualization

**File**: `main.py` or new `evaluation.py`

**Problem**: Can't see which outcomes the model confuses (e.g., is it confusing `w_caught` with `w_bowled`?).

**Why it matters**: Helps identify if merging more outcome classes would help, or if certain predictions are hopeless.

**What to do**:
- [x] Add at end of training:
  ```python
  import seaborn as sns
  
  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(12, 10))
  sns.heatmap(cm, annot=True, fmt='d', 
              xticklabels=list(full_dataset.id_to_outcome.values()),
              yticklabels=list(full_dataset.id_to_outcome.values()))
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.savefig(PLOTS_DIR / 'confusion_matrix.png')
  ```

---

### 2.3 Check Class Distribution Baseline

**File**: New file `baseline.py` or add to `main.py`

**Problem**: 39% accuracy sounds low, but what's the baseline? If just predicting `0_run` every time gives 35%, model only adds 4%.

**Why it matters**: Need to know if model is learning meaningful signal or just class frequencies.

**What to do**:
- [x] Create `baseline.py`:
  ```python
  from collections import Counter
  import numpy as np
  from config import COMBINED_PARQUET, PLAYER_MAPPING
  from dataset import CricketDataset
  
  dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING)
  
  # Majority class baseline
  counts = Counter(dataset.outcome_ids.numpy())
  total = len(dataset)
  majority_class = counts.most_common(1)[0]
  print(f"Class distribution:")
  for cls, count in counts.most_common():
      print(f"  {dataset.id_to_outcome[cls]}: {count} ({100*count/total:.1f}%)")
  
  print(f"\nMajority class baseline: {100*majority_class[1]/total:.1f}%")
  ```

---

### 2.4 Consider Focal Loss for Class Imbalance

**File**: `main.py`

**Problem**: Current approach uses sqrt-inverse weighted CrossEntropyLoss clamped to [0.5, 2.0]. This may not be aggressive enough for very rare classes (wickets like `w_stumped`, `w_lbw`).

**Why it matters**: Focal loss down-weights easy examples (common classes) and focuses training on hard examples (rare classes). Better than just class weighting.

**What to do**:
- [x] Implement focal loss:
  ```python
  class FocalLoss(nn.Module):
      def __init__(self, alpha=None, gamma=2.0):
          super().__init__()
          self.alpha = alpha  # class weights
          self.gamma = gamma  # focusing parameter
      
      def forward(self, inputs, targets):
          ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
          pt = torch.exp(-ce_loss)  # probability of correct class
          focal_loss = ((1 - pt) ** self.gamma) * ce_loss
          return focal_loss.mean()
  
  # Replace criterion:
  criterion = FocalLoss(alpha=class_weights, gamma=2.0)
  ```
- [ ] Experiment with `gamma` values: 0.5, 1.0, 2.0, 3.0
- [ ] Compare per-class F1 scores with and without focal loss

---

## 3. Data Pipeline Fixes

### 3.1 Fix Temporal Data Leak

**File**: `main.py` (lines 219-221)

**Problem**:
```python
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
```
Random split means balls from 2024 matches can appear in training, while balls from 2018 matches appear in validation. This is data leakage.

**Why it matters**: Model performance on validation set is artificially inflated. Won't generalize to truly future matches.

**What to do**:
- [ ] Modify `dataset.py` to include `match_id` or `season` in output
- [ ] Split by match or season instead of random:
  ```python
  # In main.py
  unique_matches = dataset.data['match_id'].unique()
  np.random.shuffle(unique_matches)
  val_matches = set(unique_matches[:int(len(unique_matches) * 0.1)])
  
  train_mask = ~dataset.data['match_id'].isin(val_matches)
  val_mask = dataset.data['match_id'].isin(val_matches)
  
  # Create subset datasets or use boolean indexing
  ```

---

### 3.2 Extract Venue Column (Currently Missing)

**Files**: `config.py`, then re-run `combine_data.py`

**Problem**: Raw match CSVs have `venue` column, but it's not in `COLUMNS_TO_KEEP`, so it wasn't extracted to `ipl_combined.csv`.

**Why it matters**: Venue significantly affects outcomes (e.g., Chinnaswamy = high-scoring, Chepauk = spin-friendly).

**What to do**:
- [ ] Add `'venue'` to `COLUMNS_TO_KEEP` in `config.py`:
  ```python
  COLUMNS_TO_KEEP = [
      'match_id',
      'season',        # Also add season!
      'start_date',
      'venue',         # <-- Add this
      'innings',
      # ... rest same
  ]
  ```
- [ ] Re-run `uv run python combine_data.py` to regenerate the combined data
- [ ] Add venue mapping similar to player mapping in `mapping.py`

---

### 3.3 Add Match Context Features to Dataset

**File**: `dataset.py`

**Problem**: Current dataset only returns `(striker_id, bowler_id, outcome_id)`. Missing crucial context.

**Why it matters**: This is the **root cause** of low accuracy. Cricket outcomes heavily depend on:
- Powerplay (overs 1-6) vs death overs (16-20)
- 1st vs 2nd innings
- Score pressure, wickets lost

**What to do** (data extraction only, model change is separate):
- [ ] Modify `CricketDataset.__getitem__` to also return context:
  ```python
  def __getitem__(self, idx):
      row = self.data.iloc[idx]
      
      # Extract context features
      over = int(row['ball'])  # Ball column has format like 1.3 = over 1, ball 3
      innings = int(row['innings'])
      
      # Normalize to [0, 1] range
      over_norm = over / 20.0
      innings_norm = (innings - 1) / 1.0  # 0 for 1st, 1 for 2nd
      
      context = torch.FloatTensor([over_norm, innings_norm])
      
      return (self.striker_ids[idx], self.bowler_ids[idx], 
              context, self.outcome_ids[idx])
  ```

---

## 4. Model Architecture Improvements

### 4.1 Create Context-Aware Model V2
- [x] Create new model file:
  ```python
  # model_v2.py
  import torch
  import torch.nn as nn
  
  class Cricket2VecV2(nn.Module):
      def __init__(self, num_players, num_outcomes, embedding_dim=16, context_dim=2):
          super().__init__()
          
          self.bat_embedding = nn.Embedding(num_players, embedding_dim)
          self.bowl_embedding = nn.Embedding(num_players, embedding_dim)
          
          # Context encoder
          self.context_encoder = nn.Sequential(
              nn.Linear(context_dim, 32),
              nn.ReLU(),
              nn.Linear(32, 16)
          )
          
          combined_dim = embedding_dim * 2 + 16
          
          self.classifier = nn.Sequential(
              nn.Linear(combined_dim, 128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, num_outcomes)
          )
          
          self._init_weights()
      
      def _init_weights(self):
          nn.init.normal_(self.bat_embedding.weight, std=0.1)
          nn.init.normal_(self.bowl_embedding.weight, std=0.1)
      
      def forward(self, striker_ids, bowler_ids, context):
          bat_vec = self.bat_embedding(striker_ids)
          bowl_vec = self.bowl_embedding(bowler_ids)
          ctx_vec = self.context_encoder(context)
          
          combined = torch.cat([bat_vec, bowl_vec, ctx_vec], dim=1)
          return self.classifier(combined)
  ```

---

### 4.2 Update Training Script for V2

**File**: New `train_v2.py` or modify `main.py`

**What to do**:
- [x] Update training loop to pass context:
  ```python
  for striker_ids, bowler_ids, context, outcome_ids in pbar:
      striker_ids = striker_ids.to(DEVICE)
      bowler_ids = bowler_ids.to(DEVICE)
      context = context.to(DEVICE)
      outcome_ids = outcome_ids.to(DEVICE)
      
      optimizer.zero_grad()
      outputs = model(striker_ids, bowler_ids, context)
      # ... rest same
  ```

---

### 4.3 Add Team Embeddings

**File**: `model_v2.py` and `dataset.py`

**Problem**: Current model ignores which teams are playing. Team strategies differ significantly (e.g., MI's death bowling vs CSK's spin-heavy approach).

**Why it matters**: Team context captures tactical patterns not visible in individual player embeddings. Data already has `batting_team` and `bowling_team` columns.

**What to do**:
- [x] Add team mapping in `mapping.py` (already exists, just need to use it):
  ```python
  from mapping import load_team_mapping
  team_to_id = load_team_mapping(TEAM_MAPPING)
  ```
- [x] Modify dataset to return team IDs:
  ```python
  # In dataset.py
  self.bat_team_ids = torch.LongTensor(self.data['batting_team'].map(team_to_id).values)
  self.bowl_team_ids = torch.LongTensor(self.data['bowling_team'].map(team_to_id).values)
  ```
- [x] Add team embeddings to model:
  ```python
  class Cricket2VecV2(nn.Module):
      def __init__(self, num_players, num_teams, num_outcomes, embedding_dim=16):
          # ... existing code ...
          self.bat_team_embedding = nn.Embedding(num_teams, 8)
          self.bowl_team_embedding = nn.Embedding(num_teams, 8)
          
          # Update combined_dim to include team embeddings
          combined_dim = embedding_dim * 2 + 16 + 8 * 2  # players + context + teams
  ```

---

### 4.4 Add Venue Embeddings (After 3.2)

**File**: `model_v2.py`

**Problem**: Different venues have very different characteristics (pitch, boundary size, altitude).

**Why it matters**: Wankhede (small boundaries) vs Eden Gardens (large) produces different boundary frequencies.

**What to do** (requires completing 3.2 first):
- [x] Create venue mapping:
  ```python
  # In mapping.py
  def create_venue_mapping(df: pd.DataFrame) -> Dict[str, int]:
      venues = sorted(df['venue'].dropna().unique())
      return {venue: idx for idx, venue in enumerate(venues)}
  ```
- [x] Add venue embedding to model:
  ```python
  self.venue_embedding = nn.Embedding(num_venues, 8)
  ```
- [x] Update combined_dim and forward pass

## 6. Code Quality (Lower Priority)

### 6.1 Add Type Hints Consistently

**Files**: All `.py` files

**Problem**: Some functions have type hints, some don't.

**Example fixes**:
- [ ] `inference.py`: `def get_matchup_probs(self, batter_name: str, bowler_name: str) -> Dict[str, float]:`
- [ ] `dataset.py`: `def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:`

### 6.2 Fix Hardcoded Player Names in Examples

**File**: `inference.py` (lines 216, 223, 231)

**Problem**: Example code uses hardcoded names like "V Kohli" which may not match exactly.

**What to do**:
- [ ] Add fuzzy matching or partial name search:
  ```python
  def find_player(self, partial_name: str) -> str:
      matches = [p for p in self.player_to_id if partial_name.lower() in p.lower()]
      if len(matches) == 1:
          return matches[0]
      elif len(matches) > 1:
          print(f"Multiple matches: {matches[:5]}")
      raise ValueError(f"No unique match for '{partial_name}'")
  ```

---

## 7. WOW Factor: Inference & Visualization Enhancements

> The goal is **visualization and exploration**, not prediction or betting.  
> These features will make people say "wow" when they see your embeddings in action.

---

### 7.1 Player Algebra (THE Signature Feature) ⭐

**File**: `inference.py`

**Inspiration**: Word2Vec's famous "King - Man + Woman = Queen" analogy. The batter_pitcher2vec paper does this too (opposite-handed doppelgängers).

**Cricket Examples**:
- `V Kohli - Anchor + Aggressor = ?` → Find aggressive version of Kohli
- `MS Dhoni - Finisher + Powerplay = ?` → Who plays like Dhoni but in powerplay?
- `JJ Bumrah - Pace + Spin = ?` → Spinner equivalent of Bumrah's effectiveness

**Why it's WOW**: People can explore "what if" scenarios with their favorite players.

**What to do**:
- [ ] Implement player algebra function:
  ```python
  def player_algebra(self, positive: List[str], negative: List[str], role='bat', top_k=5):
      """
      Compute: sum(positive embeddings) - sum(negative embeddings)
      Return nearest players to the resulting vector.
      
      Example: player_algebra(["V Kohli", "AB de Villiers"], ["Anchor playstyle"], role='bat')
      """
      emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
      
      result_vec = np.zeros(emb.shape[1])
      for name in positive:
          result_vec += emb[self.get_player_id(name)]
      for name in negative:
          result_vec -= emb[self.get_player_id(name)]
      
      # Find nearest neighbors to result_vec
      sims = cosine_similarity(result_vec.reshape(1, -1), emb)[0]
      top_indices = sims.argsort()[::-1][:top_k]
      
      return [(self.id_to_player[i], sims[i]) for i in top_indices]
  ```
- [ ] Create helper to compute "average archetype" vectors (e.g., average of all openers, all finishers)

---

### 7.2 K-Means Clustering for Batters & Bowlers

**File**: `inference.py`

**Problem**: Instead of using predefined roles, let embeddings discover natural clusters.

**Why it's WOW**: The model might discover roles we don't think about - "death over specialist", "spin basher", "pace killer".

**What to do**:
- [ ] Add clustering function for both roles:
  ```python
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  
  def cluster_players(self, role='bat', n_clusters=6):
      """K-means clustering on player embeddings."""
      emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
      
      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
      labels = kmeans.fit_predict(emb)
      
      # Group players by cluster
      clusters = {i: [] for i in range(n_clusters)}
      for pid, label in enumerate(labels):
          clusters[label].append(self.id_to_player[pid])
      
      # Print cluster summary
      print(f"\n--- {role.upper()} Clusters (k={n_clusters}) ---")
      for cid, players in clusters.items():
          print(f"\nCluster {cid} ({len(players)} players):")
          print(f"  Sample: {players[:10]}")
      
      return clusters, labels
  ```

- [ ] Add elbow plot to find optimal k:
  ```python
  def plot_elbow(self, role='bat', max_k=12):
      """Plot elbow curve to find optimal number of clusters."""
      emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
      
      inertias = []
      silhouettes = []
      K = range(2, max_k + 1)
      
      for k in K:
          kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
          labels = kmeans.fit_predict(emb)
          inertias.append(kmeans.inertia_)
          silhouettes.append(silhouette_score(emb, labels))
      
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
      
      ax1.plot(K, inertias, 'bo-')
      ax1.set_xlabel('Number of Clusters (k)')
      ax1.set_ylabel('Inertia')
      ax1.set_title(f'Elbow Plot - {role.capitalize()}')
      
      ax2.plot(K, silhouettes, 'ro-')
      ax2.set_xlabel('Number of Clusters (k)')
      ax2.set_ylabel('Silhouette Score')
      ax2.set_title(f'Silhouette Score - {role.capitalize()}')
      
      plt.tight_layout()
      plt.savefig(f'{role}_clustering_elbow.png')
      print(f"Saved: {role}_clustering_elbow.png")
  ```

- [ ] Visualize clusters with colored t-SNE:
  ```python
  def plot_clustered_tsne(self, role='bat', n_clusters=6, annotate_top=20):
      """t-SNE visualization with cluster colors."""
      emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
      
      # Cluster
      kmeans = KMeans(n_clusters=n_clusters, random_state=42)
      labels = kmeans.fit_predict(emb)
      
      # t-SNE
      tsne = TSNE(n_components=2, random_state=42, perplexity=30)
      reduced = tsne.fit_transform(emb)
      
      # Plot
      plt.figure(figsize=(14, 10))
      scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6, s=50)
      
      # Annotate some players per cluster
      for cluster_id in range(n_clusters):
          mask = labels == cluster_id
          cluster_indices = np.where(mask)[0][:annotate_top // n_clusters]
          for idx in cluster_indices:
              plt.annotate(self.id_to_player[idx], 
                          (reduced[idx, 0], reduced[idx, 1]),
                          fontsize=7, alpha=0.8)
      
      plt.colorbar(scatter, label='Cluster')
      plt.title(f'{role.capitalize()} Embeddings - K-Means Clustering (k={n_clusters})')
      plt.xlabel('t-SNE Dim 1')
      plt.ylabel('t-SNE Dim 2')
      plt.savefig(f'{role}_clustered_tsne.png', dpi=150)
      print(f"Saved: {role}_clustered_tsne.png")
  ```

- [ ] Run clustering for both batters and bowlers:
  ```python
  # In main block of inference.py:
  engine.plot_elbow(role='bat')
  engine.plot_elbow(role='bowl')
  engine.plot_clustered_tsne(role='bat', n_clusters=5)
  engine.plot_clustered_tsne(role='bowl', n_clusters=4)
  ```

- [ ] Name the clusters based on who's in them (manual analysis)

---

### 7.3 Opposite-Handed / Style Doppelgängers

**File**: `inference.py`

**Inspiration**: Directly from batter_pitcher2vec paper - "Bryce Harper might be Mike Trout's left-handed doppelgänger"

**Cricket Examples**:
- Right-handed Kohli's left-handed equivalent
- Pace bowler equivalent of a spinner's effectiveness

**What to do**:
- [ ] Compute average embeddings for groups:
  ```python
  def find_doppelganger(self, player_name, remove_group='right-handed', add_group='left-handed'):
      """
      Find: Player - avg(remove_group) + avg(add_group)
      
      Requires external data about player handedness (could use cricinfo API or manual CSV)
      """
      # Implementation similar to player_algebra but with group averages
  ```
- [ ] Create a CSV with player metadata (batting hand, bowling style, role)

---

### 7.4 Dream XI Generator

**File**: `inference.py`

**Problem**: Given a captain/key player, find the most complementary team.

**Why it's WOW**: Fantasy cricket players would love this!

**What to do**:
- [ ] Implement dream team diversity:
  ```python
  def build_dream_team(self, captain_name, team_size=11, role='bat'):
      """
      Build a team that maximizes embedding diversity around the captain.
      Uses farthest-point sampling to get varied player types.
      """
      emb = self.bat_embeddings
      captain_id = self.get_player_id(captain_name)
      
      selected = [captain_id]
      remaining = set(range(self.num_players)) - {captain_id}
      
      while len(selected) < team_size:
          # Find player farthest from all selected (diversity)
          best_id = None
          best_min_dist = -1
          for pid in remaining:
              min_dist = min(np.linalg.norm(emb[pid] - emb[s]) for s in selected)
              if min_dist > best_min_dist:
                  best_min_dist = min_dist
                  best_id = pid
          selected.append(best_id)
          remaining.remove(best_id)
      
      return [self.id_to_player[pid] for pid in selected]
  ```

---

### 7.5 Historic Rivalries & Matchup Heatmap

**File**: `inference.py`

**Problem**: Visualize which batters dominate which bowlers (and vice versa).

**Why it's WOW**: Cricket fans love debating "Kohli vs Starc" type matchups.

**What to do**:
- [ ] Create matchup heatmap:
  ```python
  def matchup_heatmap(self, batters: List[str], bowlers: List[str], outcome='wicket'):
      """
      Create a heatmap of wicket probabilities for batter-bowler pairs.
      """
      heatmap = np.zeros((len(batters), len(bowlers)))
      
      for i, bat in enumerate(batters):
          for j, bowl in enumerate(bowlers):
              probs = self.get_matchup_probs(bat, bowl)
              # Sum all wicket probabilities
              wicket_prob = sum(v for k, v in probs.items() if k.startswith('w_'))
              heatmap[i, j] = wicket_prob
      
      # Plot with seaborn
      import seaborn as sns
      plt.figure(figsize=(12, 8))
      sns.heatmap(heatmap, xticklabels=bowlers, yticklabels=batters, 
                  annot=True, fmt='.2f', cmap='RdYlGn_r')
      plt.title('Wicket Probability Heatmap')
      plt.savefig('matchup_heatmap.png')
  ```

---

### 7.6 Interactive 3D t-SNE Visualization (Web)

**File**: New `visualize_web.py` or Jupyter notebook

**Problem**: Static t-SNE images are boring.

**Why it's WOW**: Interactive 3D visualization with player names on hover.

**What to do**:
- [ ] Create Plotly 3D scatter:
  ```python
  import plotly.express as px
  from sklearn.manifold import TSNE
  
  def create_3d_visualization(self, role='bat'):
      emb = self.bat_embeddings if role == 'bat' else self.bowl_embeddings
      
      tsne = TSNE(n_components=3, random_state=42)
      reduced = tsne.fit_transform(emb)
      
      # Create DataFrame for Plotly
      df = pd.DataFrame({
          'x': reduced[:, 0],
          'y': reduced[:, 1],
          'z': reduced[:, 2],
          'player': [self.id_to_player[i] for i in range(self.num_players)]
      })
      
      fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='player')
      fig.write_html('player_embeddings_3d.html')  # Interactive HTML file!
  ```
- [ ] Add player photos as hover images (optional, requires image data)
- [ ] Color by cluster from 7.2

---

### 7.7 Player Evolution Over Seasons

**File**: Requires training separate models per season

**Problem**: How has a player's embedding changed over their career?

**Why it's WOW**: "Kohli 2016 vs Kohli 2023 - how has his playstyle evolved?"

**What to do**:
- [ ] Train separate models on 2016-2018, 2019-2021, 2022-2024 data
- [ ] Extract same player's embedding from each model
- [ ] Visualize trajectory in embedding space
- [ ] **Note**: This requires significant work but is very impressive

---

### 7.8 "Who Would You Rather Face?" Quiz

**File**: `inference.py` or new `quiz.py`

**Problem**: Make it interactive and fun!

**Why it's WOW**: Gamification makes people engage more.

**What to do**:
- [ ] Create quiz function:
  ```python
  def who_would_you_face(self, player_name, role='bowl'):
      """
      Given a player, show 2 random opponents.
      User guesses which one is 'easier' based on wicket probability.
      Model reveals the answer.
      """
      # Select 2 random opponents
      # Compute wicket probabilities
      # Display to user, wait for input
      # Reveal answer with probabilities
  ```

---

## 8. Additional Data Extraction (Optional)

The info CSV files have useful data not currently extracted:

| Field | In Combined? | Could Help? |
|-------|--------------|-------------|
| `toss_decision` | ❌ | Yes - affects innings strategy |
| `city` | ❌ | Maybe - grouping venues |
| `player_of_match` | ❌ | Fun for visualization |

**What to do**:
- [ ] If needed, modify `COLUMNS_TO_KEEP` or parse `_info.csv` files separately
- [ ] Most critical is `toss_decision` (bat/field) as it affects game context

---

## Progress Tracking

| Section | Status | Notes |
|---------|--------|-------|
| 1. Bug Fixes | ⬜ | |
| 2. Evaluation | ⬜ | |
| 3. Data Pipeline | ⬜ | |
| 4. Model V2 | ⬜ | |
| 5. Documentation | ⬜ | |
| 6. Code Quality | ⬜ | |
| 7. WOW Factor | ⬜ | The fun stuff! |
| 8. Optional Data | ⬜ | Nice to have |

Legend: ⬜ Not started | 🟡 In progress | ✅ Done


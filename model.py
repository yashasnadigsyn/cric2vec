import torch
import torch.nn as nn

class Cricket2Vec(nn.Module):
    """
    Cricket2Vec model inspired by batter_pitcher2vec paper.
    
    Architecture matches the paper:
    1. Embedding lookup for batter and bowler
    2. Sigmoid activation after embeddings (key difference from typical ReLU)
    3. Concatenation of embeddings
    4. Single hidden layer with ReLU
    5. Output layer (softmax applied by CrossEntropyLoss)
    """
    def __init__(self, num_players, num_outcomes, embedding_dim=9):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 1. Embedding Layers (separate for batting and bowling skills)
        self.bat_embedding = nn.Embedding(num_players, embedding_dim)
        self.bowl_embedding = nn.Embedding(num_players, embedding_dim)
        
        # 2. Sigmoid activation (as per batter_pitcher2vec paper)
        # Paper: w_b = σ(W_b · h_b) where σ is logistic/sigmoid
        self.sigmoid = nn.Sigmoid()
        
        # 3. Hidden layer (concatenated embeddings -> hidden)
        # Paper uses direct softmax, but we add one hidden layer for expressiveness
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.relu = nn.ReLU()
        
        # 4. Output Layer
        self.output_layer = nn.Linear(128, num_outcomes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small values for stable training."""
        nn.init.normal_(self.bat_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.bowl_embedding.weight, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, striker_ids, bowler_ids):
        # Step 1: Lookup Embeddings
        bat_vecs = self.bat_embedding(striker_ids)   # [batch, emb_dim]
        bowl_vecs = self.bowl_embedding(bowler_ids)  # [batch, emb_dim]
        
        # Step 2: Apply Sigmoid (as per paper - key architectural choice)
        bat_vecs = self.sigmoid(bat_vecs)
        bowl_vecs = self.sigmoid(bowl_vecs)
        
        # Step 3: Concatenate
        combined = torch.cat([bat_vecs, bowl_vecs], dim=1)  # [batch, emb_dim * 2]
        
        # Step 4: Hidden layer
        x = self.fc1(combined)
        x = self.relu(x)
        
        # Step 5: Output logits
        logits = self.output_layer(x)  # [batch, num_outcomes]
        
        return logits
    
    def get_batter_embeddings(self, player_ids):
        """Get batting embeddings for visualization/analysis."""
        with torch.no_grad():
            return self.sigmoid(self.bat_embedding(player_ids))
    
    def get_bowler_embeddings(self, player_ids):
        """Get bowling embeddings for visualization/analysis."""
        with torch.no_grad():
            return self.sigmoid(self.bowl_embedding(player_ids))
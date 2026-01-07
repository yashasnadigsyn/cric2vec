import torch
import torch.nn as nn

class Cricket2Vec(nn.Module):
    def __init__(self, num_players, num_outcomes, embedding_dim=7):
        super().__init__()
        
        # 1. Embedding Layers
        # Concept: Input ID -> Dense Vector
        # We need two separate sets of skills.
        self.bat_embedding = nn.Embedding(num_players, embedding_dim)
        self.bowl_embedding = nn.Embedding(num_players, embedding_dim)
        
        # 2. Merge Layer (Concatenation)
        # Input to next layer will be size: embedding_dim * 2

        # 3. Dense / Linear Layers (The "Interaction" Logic)
        # We want to learn non-linear interactions (e.g., Left Hand Bat vs Off Spin)
        self.fc1 = nn.Linear(embedding_dim * 2, 128) # Hidden layer 1
        self.relu = nn.ReLU()                        # Activation
        self.dropout = nn.Dropout(0.2)               # Dropout for regularization
        self.fc2 = nn.Linear(128, 64)                # Hidden layer 2 (optional)
        
        # 4. Output Layer
        # Map to the probability of each outcome
        self.output_layer = nn.Linear(64, num_outcomes)
        
        # Note: No Softmax here if using nn.CrossEntropyLoss during training 
        # (CrossEntropyLoss includes LogSoftmax)
    def forward(self, striker_ids, bowler_ids):
        # Step 1: Lookup Embeddings
        # striker_ids shape: [batch_size]
        bat_vecs = self.bat_embedding(striker_ids)  # Shape: [batch, emb_dim]
        bowl_vecs = self.bowl_embedding(bowler_ids) # Shape: [batch, emb_dim]
        
        # Step 2: Combine
        # Concatenate the two vectors side-by-side
        combined = torch.cat([bat_vecs, bowl_vecs], dim=1) # Shape: [batch, emb_dim * 2]
        
        # Step 3: Pass through hidden layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x) # Added dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x) # Added dropout
        
        # Step 4: Output Logits
        logits = self.output_layer(x) # Shape: [batch, num_outcomes]
        
        return logits
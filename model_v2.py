
import torch
import torch.nn as nn

class Cricket2VecV2(nn.Module):
    def __init__(self, num_players, num_teams, num_venues, num_outcomes, 
                 embedding_dim=16, team_embedding_dim=8, venue_embedding_dim=8, 
                 context_dim=2):
        """
        Context-Aware Cricket2Vec Model (V2).
        
        Args:
            num_players (int): Number of unique players
            num_teams (int): Number of unique teams
            num_venues (int): Number of unique venues
            num_outcomes (int): Number of outcome classes
            embedding_dim (int): Dimension of player embeddings
            team_embedding_dim (int): Dimension of team embeddings
            venue_embedding_dim (int): Dimension of venue embeddings
            context_dim (int): Dimension of context vector (default 2: over_norm, innings_norm)
        """
        super().__init__()
        
        # 1. Embeddings
        # We use separate embeddings for batting and bowling roles as per V1
        self.bat_embedding = nn.Embedding(num_players, embedding_dim)
        self.bowl_embedding = nn.Embedding(num_players, embedding_dim)
        
        # Team embeddings (separate for batting vs bowling team?? 
        # Usually a team is just a team, but their role matters. 
        # Let's use SHARED team embeddings, but position tells the role)
        self.team_embedding = nn.Embedding(num_teams, team_embedding_dim)
        
        # Venue embedding
        self.venue_embedding = nn.Embedding(num_venues, venue_embedding_dim)
        
        # 2. Context Encoder
        # Processes continuous context features (over, innings)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()  # Added ReLU to match typical MLP
        )
        
        # 3. Classifier
        # Input: [Bat(E) + Bowl(E) + BatTeam(Te) + BowlTeam(Te) + Venue(Ve) + Context(16)]
        combined_dim = (embedding_dim * 2) + (team_embedding_dim * 2) + venue_embedding_dim + 16
        
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
        """Initialize weights for better convergence."""
        nn.init.xavier_uniform_(self.bat_embedding.weight)
        nn.init.xavier_uniform_(self.bowl_embedding.weight)
        nn.init.xavier_uniform_(self.team_embedding.weight)
        nn.init.xavier_uniform_(self.venue_embedding.weight)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, striker_ids, bowler_ids, bat_team_ids, bowl_team_ids, venue_ids, context):
        """
        Forward pass.
        
        Args:
            striker_ids: (batch_size)
            bowler_ids: (batch_size)
            bat_team_ids: (batch_size)
            bowl_team_ids: (batch_size)
            venue_ids: (batch_size)
            context: (batch_size, context_dim)
        """
        # Get embeddings
        bat_vec = self.bat_embedding(striker_ids)        # [B, E]
        bowl_vec = self.bowl_embedding(bowler_ids)       # [B, E]
        
        bat_team_vec = self.team_embedding(bat_team_ids) # [B, Te]
        bowl_team_vec = self.team_embedding(bowl_team_ids) # [B, Te]
        
        venue_vec = self.venue_embedding(venue_ids)      # [B, Ve]
        
        # Process context
        ctx_vec = self.context_encoder(context)          # [B, 16]
        
        # Concatenate all features
        combined = torch.cat([
            bat_vec, 
            bowl_vec, 
            bat_team_vec, 
            bowl_team_vec, 
            venue_vec, 
            ctx_vec
        ], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        return logits

    # Helper methods for inference / inspection
    def get_batter_embeddings(self, player_ids):
        return self.bat_embedding(player_ids)

    def get_bowler_embeddings(self, player_ids):
        return self.bowl_embedding(player_ids)

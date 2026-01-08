import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from config import (
    COMBINED_PARQUET, PLAYER_MAPPING, TEAM_MAPPING, VENUE_MAPPING,
    MODEL_CONFIG, TRAINING_CONFIG, DEVICE, CHECKPOINT_DIR
)
from dataset import CricketDataset
from model_v2 import Cricket2VecV2

def save_checkpoint(model, optimizer, epoch, loss, filename):
    filepath = CHECKPOINT_DIR / filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def train():
    print(f"Using device: {DEVICE}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Dataset
    print("Loading dataset...")
    full_dataset = CricketDataset(
        COMBINED_PARQUET, 
        PLAYER_MAPPING, 
        TEAM_MAPPING, 
        VENUE_MAPPING
    )
    
    # 2. Split Train/Val
    val_size = int(len(full_dataset) * TRAINING_CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # 3. Initialize Model V2
    model = Cricket2VecV2(
        num_players=full_dataset.get_num_players(),
        num_teams=full_dataset.get_num_teams(),
        num_venues=full_dataset.get_num_venues(),
        num_outcomes=full_dataset.get_num_outcomes(),
        embedding_dim=MODEL_CONFIG['embedding_dim'],    # 9
        team_embedding_dim=8,
        venue_embedding_dim=8,
        context_dim=2
    ).to(DEVICE)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Optimizer (SGD with Nesterov as per paper/TODO)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'], 
        momentum=TRAINING_CONFIG['momentum'],
        nesterov=True,
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 5. Loss Function
    # TODO: Add Focal Loss support if needed, for now standard CrossEntropy
    # But wait, TODO 2.4 says "Consider Focal Loss". Let's stick to CrossEntropy for initial V2 test
    # to isolate variables, unless the TODO explicitly demanded it for V2.
    # The TODO 2.4 was "Consider Focal Loss", not mandatory for V2, but good to have.
    # Let's use CrossEntropy for stability first.
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        
        for striker, bowler, bat_team, bowl_team, venue, context, outcome in pbar:
            # Move to device
            striker = striker.to(DEVICE)
            bowler = bowler.to(DEVICE)
            bat_team = bat_team.to(DEVICE)
            bowl_team = bowl_team.to(DEVICE)
            venue = venue.to(DEVICE)
            context = context.to(DEVICE)
            outcome = outcome.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            logits = model(striker, bowler, bat_team, bowl_team, venue, context)
            
            loss = criterion(logits, outcome)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for striker, bowler, bat_team, bowl_team, venue, context, outcome in val_loader:
                striker = striker.to(DEVICE)
                bowler = bowler.to(DEVICE)
                bat_team = bat_team.to(DEVICE)
                bowl_team = bowl_team.to(DEVICE)
                venue = venue.to(DEVICE)
                context = context.to(DEVICE)
                outcome = outcome.to(DEVICE)
                
                logits = model(striker, bowler, bat_team, bowl_team, venue, context)
                loss = criterion(logits, outcome)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += outcome.size(0)
                correct += (predicted == outcome).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Acc = {accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, "best_model_v2.pt")
            
        # Periodic save (less frequent as per TODO 1.3)
        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, epoch, avg_val_loss, f"checkpoint_v2_ep{epoch+1}.pt")

if __name__ == "__main__":
    train()

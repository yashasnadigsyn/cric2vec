"""
Main training script for Cricket2Vec model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm

from config import (
    COMBINED_PARQUET, PLAYER_MAPPING, CHECKPOINT_DIR, DEVICE,
    TRAINING_CONFIG, MODEL_CONFIG
)
from dataset import CricketDataset
from model import Cricket2Vec


def train_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    print("Loading Dataset...")
    
    full_dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING)
    
    # Split into Train/Val
    val_size = int(len(full_dataset) * TRAINING_CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")
    
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
    
    # 2. Initialize Model
    num_players = full_dataset.get_num_players()
    num_outcomes = full_dataset.get_num_outcomes()
    
    model = Cricket2Vec(
        num_players, 
        num_outcomes, 
        MODEL_CONFIG['embedding_dim']
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    # 3. Training Loop
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(TRAINING_CONFIG['epochs']):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        for striker_ids, bowler_ids, outcome_ids in pbar:
            striker_ids = striker_ids.to(DEVICE)
            bowler_ids = bowler_ids.to(DEVICE)
            outcome_ids = outcome_ids.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(striker_ids, bowler_ids)
            loss = criterion(outputs, outcome_ids)
            
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
            for striker_ids, bowler_ids, outcome_ids in val_loader:
                striker_ids = striker_ids.to(DEVICE)
                bowler_ids = bowler_ids.to(DEVICE)
                outcome_ids = outcome_ids.to(DEVICE)
                
                outputs = model(striker_ids, bowler_ids)
                loss = criterion(outputs, outcome_ids)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += outcome_ids.size(0)
                correct += (predicted == outcome_ids).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Done [{elapsed:.1f}s]. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%")
        
        # Save Checkpoint
        save_path = CHECKPOINT_DIR / f"cric2vec_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'player_mapping': full_dataset.player_to_id,
            'outcome_mapping': full_dataset.id_to_outcome
        }, save_path)

    print("\nTraining Complete!")


if __name__ == "__main__":
    train_model()

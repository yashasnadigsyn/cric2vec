"""
Main training script for Cricket2Vec model.
Aligned with batter_pitcher2vec paper hyperparameters.
Includes training visualization (loss curves & loss landscape).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from config import (
    COMBINED_PARQUET, PLAYER_MAPPING, CHECKPOINT_DIR, DEVICE,
    TRAINING_CONFIG, MODEL_CONFIG, BASE_DIR
)
from dataset import CricketDataset
from model import Cricket2Vec


# Directory for plots
PLOTS_DIR = BASE_DIR / "plots"


def compute_class_weights(dataset):
    """
    Compute class weights using sqrt-inverse frequency (gentler than inverse).
    This balances rare classes without destabilizing common ones.
    """
    class_counts = torch.bincount(dataset.outcome_ids)
    total_samples = len(dataset.outcome_ids)
    num_classes = len(class_counts)
    
    # Use sqrt for gentler weighting (less aggressive than inverse)
    weights = torch.sqrt(total_samples / (num_classes * class_counts.float()))
    
    # Normalize to mean of 1
    weights = weights / weights.mean()
    
    # Clamp to prevent extreme weights
    weights = torch.clamp(weights, min=0.5, max=2.0)
    
    print(f"Class weights (sqrt-based, clamped): {weights.tolist()}")
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples and focuses training on hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (optional). If provided, should be a tensor of shape [num_classes]
        gamma: Focusing parameter. Higher values focus more on hard examples.
               gamma=0 equivalent to cross-entropy. Typical values: 0.5, 1.0, 2.0, 3.0
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def plot_training_curves(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2 = axes[1]
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_loss_landscape(model, criterion, val_loader, param_history, save_path, 
                        resolution=30, range_scale=1.0):
    """
    Plot 2D loss landscape with optimization trajectory.
    Projects the parameter space onto 2 random directions and visualizes loss.
    
    Inspired by "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    """
    print("\nGenerating loss landscape visualization...")
    
    model.eval()
    
    # Get current model parameters as a flattened vector
    def get_params_vector(model):
        return torch.cat([p.data.flatten() for p in model.parameters()])
    
    def set_params_vector(model, params_vec):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(params_vec[offset:offset+numel].view_as(p))
            offset += numel
    
    # Save original parameters
    original_params = get_params_vector(model).clone()
    
    # Create two random orthogonal directions for projection
    torch.manual_seed(42)
    direction1 = torch.randn_like(original_params)
    direction1 = direction1 / direction1.norm()
    
    direction2 = torch.randn_like(original_params)
    # Make direction2 orthogonal to direction1 (Gram-Schmidt)
    direction2 = direction2 - (direction2 @ direction1) * direction1
    direction2 = direction2 / direction2.norm()
    
    # Scale directions by parameter norm for meaningful exploration
    param_norm = original_params.norm()
    direction1 = direction1 * param_norm * 0.5
    direction2 = direction2 * param_norm * 0.5
    
    # Create grid
    alphas = np.linspace(-range_scale, range_scale, resolution)
    betas = np.linspace(-range_scale, range_scale, resolution)
    loss_surface = np.zeros((resolution, resolution))
    
    # Compute loss at each grid point (use subset of data for speed)
    val_subset = []
    max_batches = 10
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        val_subset.append(batch)
    
    for i, alpha in enumerate(tqdm(alphas, desc="Computing loss landscape")):
        for j, beta in enumerate(betas):
            # Perturb parameters
            new_params = original_params + alpha * direction1 + beta * direction2
            set_params_vector(model, new_params)
            
            # Compute loss on subset
            total_loss = 0.0
            with torch.no_grad():
                for striker_ids, bowler_ids, context, outcome_ids in val_subset:
                    striker_ids = striker_ids.to(DEVICE)
                    bowler_ids = bowler_ids.to(DEVICE)
                    outcome_ids = outcome_ids.to(DEVICE)
                    outputs = model(striker_ids, bowler_ids)
                    total_loss += criterion(outputs, outcome_ids).item()
            
            loss_surface[j, i] = total_loss / len(val_subset)
    
    # Restore original parameters
    set_params_vector(model, original_params)
    
    # Project parameter history onto the 2D plane
    if param_history:
        trajectory_alpha = []
        trajectory_beta = []
        for params in param_history:
            diff = params - original_params
            trajectory_alpha.append((diff @ direction1 / (direction1 @ direction1)).item())
            trajectory_beta.append((diff @ direction2 / (direction2 @ direction2)).item())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a nice colormap (similar to the reference image)
    X, Y = np.meshgrid(alphas, betas)
    
    # Contour plot
    contour = ax.contourf(X, Y, loss_surface, levels=30, cmap='copper_r')
    ax.contour(X, Y, loss_surface, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    
    # Plot optimization trajectory
    if param_history and len(trajectory_alpha) > 1:
        ax.plot(trajectory_alpha, trajectory_beta, 'c-', linewidth=1.5, alpha=0.8)
        ax.scatter(trajectory_alpha[0], trajectory_beta[0], c='cyan', s=100, 
                   marker='o', edgecolors='white', linewidths=2, zorder=5, label='Start')
        ax.scatter(trajectory_alpha[-1], trajectory_beta[-1], c='cyan', s=100, 
                   marker='s', edgecolors='white', linewidths=2, zorder=5, label='End')
    
    ax.set_xlabel(r'$\phi_0$ (Direction 1)', fontsize=12)
    ax.set_ylabel(r'$\phi_1$ (Direction 2)', fontsize=12)
    ax.set_title(r'Loss Landscape $L[\phi]$', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Loss', fontsize=11)
    
    # Add text annotation
    ax.text(0.05, 0.05, 'SGD with Nesterov\nMomentum', transform=ax.transAxes,
            fontsize=10, color='cyan', verticalalignment='bottom')
    
    if param_history:
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss landscape saved to: {save_path}")


def train_model():
    print(f"Using device: {DEVICE}")
    print(f"Config: batch_size={TRAINING_CONFIG['batch_size']}, lr={TRAINING_CONFIG['learning_rate']}, "
          f"embedding_dim={MODEL_CONFIG['embedding_dim']}")
    
    # Create directories
    PLOTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Prepare Data
    print("\nLoading Dataset...")
    full_dataset = CricketDataset(COMBINED_PARQUET, PLAYER_MAPPING)
    
    # Match-based split to prevent temporal data leak
    # (balls from same match should not be split across train/val)
    match_ids = full_dataset.get_match_ids()
    unique_matches = list(set(match_ids))
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(unique_matches)
    
    val_match_count = int(len(unique_matches) * TRAINING_CONFIG['val_split'])
    val_matches = set(unique_matches[:val_match_count])
    
    # Create indices for train/val based on match assignment
    train_indices = [i for i, mid in enumerate(match_ids) if mid not in val_matches]
    val_indices = [i for i, mid in enumerate(match_ids) if mid in val_matches]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train Size: {len(train_dataset)} ({len(unique_matches) - val_match_count} matches)")
    print(f"Val Size: {len(val_dataset)} ({val_match_count} matches)")
    print(f"Num Players: {full_dataset.get_num_players()}, Num Outcomes: {full_dataset.get_num_outcomes()}")
    
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
    
    class_weights = compute_class_weights(full_dataset).to(DEVICE)
    
    # Choose loss function based on config
    if TRAINING_CONFIG.get('use_focal_loss', False):
        gamma = TRAINING_CONFIG.get('focal_gamma', 2.0)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        print(f"Using Focal Loss with gamma={gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using CrossEntropyLoss with class weights")
    
    # SGD with Nesterov momentum (as per batter_pitcher2vec paper)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],  # 0.01
        momentum=TRAINING_CONFIG['momentum'],  # 0.9
        nesterov=True,
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    # History tracking for visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Track parameter trajectory for loss landscape
    param_history = []
    
    # Helper to get params
    def get_params_vector():
        return torch.cat([p.data.cpu().flatten() for p in model.parameters()])
    
    # Save initial parameters
    param_history.append(get_params_vector())
    
    # 3. Training Loop
    best_val_acc = 0.0
    print(f"\nStarting Training (SGD with Nesterov, lr={TRAINING_CONFIG['learning_rate']})...")
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        for striker_ids, bowler_ids, context, outcome_ids in pbar:
            striker_ids = striker_ids.to(DEVICE)
            bowler_ids = bowler_ids.to(DEVICE)
            # context = context.to(DEVICE)  # Unused by v1 model, but ready for v2
            outcome_ids = outcome_ids.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(striker_ids, bowler_ids)
            loss = criterion(outputs, outcome_ids)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for striker_ids, bowler_ids, context, outcome_ids in val_loader:
                striker_ids = striker_ids.to(DEVICE)
                bowler_ids = bowler_ids.to(DEVICE)
                # context = context.to(DEVICE)  # Unused by v1 model
                outcome_ids = outcome_ids.to(DEVICE)
                
                outputs = model(striker_ids, bowler_ids)
                loss = criterion(outputs, outcome_ids)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += outcome_ids.size(0)
                correct += (predicted == outcome_ids).sum().item()
                
                # Collect predictions for per-class metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(outcome_ids.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Compute weighted F1 score
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        scheduler.step(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save param snapshot every 5 epochs for trajectory
        if (epoch + 1) % 5 == 0:
            param_history.append(get_params_vector())
        
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print classification report every 20 epochs
        if (epoch + 1) % 20 == 0:
            target_names = [full_dataset.id_to_outcome[i] for i in range(num_outcomes)]
            print("\n" + "=" * 60)
            print(f"Classification Report (Epoch {epoch + 1})")
            print("=" * 60)
            print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        print(f"Epoch {epoch+1} [{elapsed:.1f}s] Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, F1: {weighted_f1:.4f}, "
              f"Acc: {accuracy:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            save_path = CHECKPOINT_DIR / "cric2vec_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_val_loss,
                'player_mapping': full_dataset.player_to_id,
                'outcome_mapping': full_dataset.id_to_outcome,
                'history': history
            }, save_path)
            print(f"  ↳ New best model saved! Acc: {accuracy:.2f}%")
        
        # Save periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            save_path = CHECKPOINT_DIR / f"cric2vec_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'loss': avg_val_loss,
                'history': history
            }, save_path)

    print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.2f}%")
    
    # Save final parameter snapshot
    param_history.append(get_params_vector())
    
    # Generate final confusion matrix
    print("\nGenerating final confusion matrix...")
    target_names = [full_dataset.id_to_outcome[i] for i in range(num_outcomes)]
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix - Final Epoch', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {PLOTS_DIR / 'confusion_matrix.png'}")
    
    # Plot training curves
    plot_training_curves(history, PLOTS_DIR / "training_curves.png")
    
    # Plot loss landscape
    plot_loss_landscape(
        model, criterion, val_loader, param_history,
        PLOTS_DIR / "loss_landscape.png",
        resolution=25,  # Lower for faster computation
        range_scale=0.5
    )
    
    # Save history as numpy
    np.savez(
        PLOTS_DIR / "training_history.npz",
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
        val_acc=history['val_acc'],
        lr=history['lr']
    )
    print(f"Training history saved to: {PLOTS_DIR / 'training_history.npz'}")


if __name__ == "__main__":
    train_model()

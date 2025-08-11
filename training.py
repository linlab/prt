"""
Training functions for different model types
Includes general training, diffusion training, and VAE training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def plot_training_curves(train_losses, val_losses, learning_rates, model_name):
    """Plot and save training curves"""
    
    # Extract dataset name from model_name if present
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_dir = f'outputs/{dataset_name}/figures/training_curves'
    else:
        save_dir = 'outputs/figures/training_curves'
        pure_model_name = model_name
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{pure_model_name} - Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Learning rate
    axes[1].plot(learning_rates, label='Learning Rate', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title(f'{pure_model_name} - Learning Rate Schedule')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_model_comprehensive(model, train_loader, val_loader, model_name, num_epochs=20, device='cuda'):
    """Comprehensive training with early stopping and logging"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    mse_loss = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    
    print(f"Training {model_name}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in pbar:
            input_images = batch['input_image'].to(device)
            condition_ids = batch['condition_id'].to(device)
            severities = batch['severity'].to(device)
            target_images = batch['target_image'].to(device)
            
            optimizer.zero_grad()
            
            predicted_images = model(input_images, condition_ids, severities)
            
            # Loss calculation
            reconstruction_loss = mse_loss(predicted_images, target_images)
            
            # Regularization for normal condition
            normal_mask = (condition_ids == 0)
            if normal_mask.any():
                normal_outputs = predicted_images[normal_mask]
                normal_inputs = input_images[normal_mask]
                identity_loss = mse_loss(normal_outputs, normal_inputs) * 0.1
                total_loss = reconstruction_loss + identity_loss
            else:
                total_loss = reconstruction_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_images = batch['input_image'].to(device)
                condition_ids = batch['condition_id'].to(device)
                severities = batch['severity'].to(device)
                target_images = batch['target_image'].to(device)
                
                predicted_images = model(input_images, condition_ids, severities)
                loss = mse_loss(predicted_images, target_images)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, LR = {current_lr:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Extract dataset name for saving
            if '_' in model_name:
                dataset_name = model_name.split('_')[0]
                pure_model_name = '_'.join(model_name.split('_')[1:])
                save_path = f'outputs/{dataset_name}/models/{pure_model_name}_best.pth'
            else:
                save_path = f'outputs/models/{model_name}_best.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Save final model
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_path = f'outputs/{dataset_name}/models/{pure_model_name}_final.pth'
    else:
        save_path = f'outputs/models/{model_name}_final.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
    }, save_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, learning_rates, model_name)
    
    return train_losses, val_losses

def train_diffusion_model(model, train_loader, val_loader, model_name, num_epochs=30, device='cuda'):
    """Specialized training for diffusion model"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    
    train_losses = []
    val_losses = []
    
    print(f"Training Diffusion Model {model_name}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in pbar:
            input_images = batch['input_image'].to(device)
            condition_ids = batch['condition_id'].to(device)
            severities = batch['severity'].to(device)
            target_images = batch['target_image'].to(device)
            
            batch_size = input_images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(0, model.timesteps, (batch_size,), device=device)
            
            # Sample noise
            noise = torch.randn_like(target_images)
            
            # Add noise to target images
            noisy_targets = model.add_noise(target_images, noise, timesteps)
            
            # Predict noise
            predicted_noise = model(noisy_targets, condition_ids, severities, timesteps)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_images = batch['input_image'].to(device)
                condition_ids = batch['condition_id'].to(device)
                severities = batch['severity'].to(device)
                target_images = batch['target_image'].to(device)
                
                batch_size = input_images.shape[0]
                timesteps = torch.randint(0, model.timesteps, (batch_size,), device=device)
                noise = torch.randn_like(target_images)
                noisy_targets = model.add_noise(target_images, noise, timesteps)
                
                predicted_noise = model(noisy_targets, condition_ids, severities, timesteps)
                loss = F.mse_loss(predicted_noise, noise)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        scheduler.step()
        
        # Save best model
        if len(val_losses) == 1 or avg_val_loss < min(val_losses[:-1]):
            if '_' in model_name:
                dataset_name = model_name.split('_')[0]
                pure_model_name = '_'.join(model_name.split('_')[1:])
                save_path = f'outputs/{dataset_name}/models/{pure_model_name}_best.pth'
            else:
                save_path = f'outputs/models/{model_name}_best.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Save final model
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_path = f'outputs/{dataset_name}/models/{pure_model_name}_final.pth'
    else:
        save_path = f'outputs/models/{model_name}_final.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, save_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, [optimizer.param_groups[0]['lr']] * len(train_losses), model_name)
    
    return train_losses, val_losses

def train_vae_model(model, train_loader, val_loader, model_name, num_epochs=25, device='cuda'):
    """Specialized training for VAE model"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    early_stopping = EarlyStopping(patience=12, min_delta=1e-4)
    
    train_losses = []
    val_losses = []
    
    print(f"Training VAE Model {model_name}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in pbar:
            input_images = batch['input_image'].to(device)
            condition_ids = batch['condition_id'].to(device)
            severities = batch['severity'].to(device)
            target_images = batch['target_image'].to(device)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(input_images, condition_ids, severities)
            
            # VAE loss = reconstruction loss + KL divergence
            recon_loss = F.mse_loss(recon_batch, target_images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = recon_loss + 0.1 * kl_loss  # Beta-VAE with beta=0.1
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_images = batch['input_image'].to(device)
                condition_ids = batch['condition_id'].to(device)
                severities = batch['severity'].to(device)
                target_images = batch['target_image'].to(device)
                
                recon_batch, mu, logvar = model(input_images, condition_ids, severities)
                
                recon_loss = F.mse_loss(recon_batch, target_images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + 0.1 * kl_loss
                
                val_loss += total_loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        scheduler.step()
        
        # Save best model
        if len(val_losses) == 1 or avg_val_loss < min(val_losses[:-1]):
            if '_' in model_name:
                dataset_name = model_name.split('_')[0]
                pure_model_name = '_'.join(model_name.split('_')[1:])
                save_path = f'outputs/{dataset_name}/models/{pure_model_name}_best.pth'
            else:
                save_path = f'outputs/models/{model_name}_best.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Save final model and plot curves
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_path = f'outputs/{dataset_name}/models/{pure_model_name}_final.pth'
    else:
        save_path = f'outputs/models/{model_name}_final.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, save_path)
    
    plot_training_curves(train_losses, val_losses, [optimizer.param_groups[0]['lr']] * len(train_losses), model_name)
    
    return train_losses, val_losses
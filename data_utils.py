"""
Dataset utilities for Perceptual Reality Transformer
Handles dataset loading, splitting, and synthetic data generation
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from perturbations import PERTURBATION_FUNCTIONS

class PerceptualRealityDataset(Dataset):
    """Dataset that generates synthetic training data"""
    
    def __init__(self, base_dataset, num_samples_per_condition=500):
        self.base_dataset = base_dataset
        self.num_conditions = len(PERTURBATION_FUNCTIONS)
        self.samples_per_condition = num_samples_per_condition
        
        # Pre-generate all combinations
        self.samples = []
        for i in range(len(base_dataset)):
            for condition_id in range(self.num_conditions):
                for severity in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    self.samples.append((i, condition_id, severity))
        
        # Limit total samples if needed
        if len(self.samples) > num_samples_per_condition * self.num_conditions:
            self.samples = random.sample(self.samples, 
                                       num_samples_per_condition * self.num_conditions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        base_idx, condition_id, severity = self.samples[idx]
        
        # Get base image
        if isinstance(self.base_dataset[base_idx], tuple):
            base_image, _ = self.base_dataset[base_idx]
        else:
            base_image = self.base_dataset[base_idx]
        
        # Apply perturbation to create target
        target_image = PERTURBATION_FUNCTIONS[condition_id](
            base_image.unsqueeze(0), severity
        ).squeeze(0)
        
        return {
            'input_image': base_image,
            'condition_id': torch.tensor(condition_id, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.float32),
            'target_image': target_image
        }

def setup_dataset(dataset_choice='imagenet', image_size=224, num_samples_per_condition=300):
    """Setup dataset with publication-quality considerations"""
    
    print(f"Setting up {dataset_choice} dataset...")
    
    if dataset_choice == 'imagenet':
        # ImageNet validation set - higher quality, more diverse
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            base_dataset = torchvision.datasets.ImageFolder(
                root='./imagenet/val',
                transform=transform
            )
            print(f"✓ Loaded ImageNet validation set with {len(base_dataset)} images")
            print(f"✓ Classes: {len(base_dataset.classes)} categories")
            
            # Use subset for faster training but better diversity
            indices = list(range(0, len(base_dataset), len(base_dataset) // 5000))  # ~5K images
            subset_dataset = torch.utils.data.Subset(base_dataset, indices)
            
            print(f"✓ Using subset of {len(subset_dataset)} images for training")
            return subset_dataset, 'ImageNet_val'
            
        except Exception as e:
            print(f"✗ Failed to load ImageNet: {e}")
            print("Falling back to CIFAR-10...")
            return setup_dataset('cifar10', image_size, num_samples_per_condition)
    
    elif dataset_choice == 'cifar10':
        # CIFAR-10 - smaller but reliable
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        base_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        print(f"✓ Loaded CIFAR-10 with {len(base_dataset)} images")
        return base_dataset, 'CIFAR10'
    
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")

def create_publication_splits(full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create publication-quality train/val/test splits"""
    
    total_size = len(full_dataset)
    
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Ensure we have enough samples for robust evaluation
    min_test_size = 1000
    if test_size < min_test_size:
        print(f"Warning: Test set size ({test_size}) is small. Consider using more data.")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    print(f"Dataset splits:")
    print(f"  Training:   {len(train_dataset):,} samples ({train_ratio:.1%})")
    print(f"  Validation: {len(val_dataset):,} samples ({val_ratio:.1%})")
    print(f"  Testing:    {len(test_dataset):,} samples ({test_ratio:.1%})")
    
    return train_dataset, val_dataset, test_dataset
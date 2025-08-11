#!/usr/bin/env python3
"""
Perceptual Reality Transformer - Main Execution Script
"""

import torch
import numpy as np
from pathlib import Path
import json
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

# Import our modules
from models import *
from data_utils import setup_dataset, create_publication_splits, PerceptualRealityDataset
from training import train_model_comprehensive, train_diffusion_model, train_vae_model
from evaluation import compute_comprehensive_metrics, create_condition_comparison, create_severity_comparison
from utils import setup_directories_for_dataset, print_dataset_results_table

# Model definitions for easy selection
ALL_MODELS = {
    'simple': SimpleCNN,
    'residual': ResidualPerceptual,
    'hybrid': HybridModel,
    'vit': ViTPerceptual,
    'recurrent': RecurrentPerceptual,
    'diffusion': DiffusionPerceptual,
    'vae': GenerativePerceptual,
}

# Model groups for quick selection
MODEL_GROUPS = {
    'fast': ['simple', 'residual', 'hybrid'],
    'advanced': ['vit', 'recurrent'],
    'generative': ['diffusion', 'vae'],
    'all': list(ALL_MODELS.keys())
}

def parse_arguments():
    """Parse command line arguments for model and dataset selection"""
    parser = argparse.ArgumentParser(description='Perceptual Reality Transformer Training')
    
    parser.add_argument('--models', nargs='+', 
                       choices=list(ALL_MODELS.keys()) + list(MODEL_GROUPS.keys()),
                       default=['fast'], 
                       help='Models to train/evaluate. Use model names or groups (fast, advanced, generative, all)')
    
    parser.add_argument('--datasets', nargs='+', 
                       choices=['imagenet', 'cifar10', 'both'],
                       default=['cifar10'],
                       help='Datasets to use')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for training')
    
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (auto-adjusted for dataset)')
    
    parser.add_argument('--samples-per-condition', type=int, default=200,
                       help='Samples per condition (reduced for M3 speed)')
    
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, only evaluate existing models')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain even if model exists')
    
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], 
                       default='auto',
                       help='Device to use')
    
    parser.add_argument('--parallel-id', type=int, default=None,
                       help='Process ID for parallel execution (0, 1, 2...)')
    
    return parser.parse_args()

def resolve_models(model_args):
    """Resolve model arguments to actual model list"""
    models_to_run = []
    
    for arg in model_args:
        if arg in MODEL_GROUPS:
            models_to_run.extend(MODEL_GROUPS[arg])
        elif arg in ALL_MODELS:
            models_to_run.append(arg)
        else:
            print(f"Warning: Unknown model/group '{arg}', skipping")
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(models_to_run))

def resolve_datasets(dataset_args):
    """Resolve dataset arguments"""
    if 'both' in dataset_args:
        return ['imagenet', 'cifar10']
    else:
        return dataset_args

def setup_device(device_arg):
    """Setup optimal device for M3 MacBook"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'  # M3 optimization
            print("✓ Using MPS (Metal Performance Shaders) for M3 acceleration")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("✓ Using CUDA")
        else:
            device = 'cpu'
            print("⚠ Using CPU (slower)")
    else:
        device = device_arg
        print(f"Using specified device: {device}")
    
    # M3 optimizations
    if device == 'mps':
        # Set environment variables for better MPS performance
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce memory usage
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable fallback for unsupported ops
        print("✓ MPS optimizations applied")
    
    return device

def run_single_model_evaluation(model_key, model_class, dataset_config, args, device):
    """Run evaluation for a single model on a single dataset"""
    
    dataset_name = dataset_config['name']
    model_name = f"{dataset_name}_{model_key}"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {model_key.upper()} ON {dataset_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Dataset setup
        base_dataset, actual_dataset_name = setup_dataset(
            dataset_choice=dataset_config['choice'],
            image_size=dataset_config['image_size'],
            num_samples_per_condition=args.samples_per_condition
        )
        
        # Create perceptual reality dataset
        full_dataset = PerceptualRealityDataset(
            base_dataset, 
            num_samples_per_condition=args.samples_per_condition
        )
        
        # Create splits
        train_dataset, val_dataset, test_dataset = create_publication_splits(
            full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Create data loaders with M3 optimizations
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=dataset_config['batch_size'], 
            shuffle=True, 
            num_workers=2,  # Reduced for M3
            pin_memory=True if device != 'cpu' else False,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=dataset_config['batch_size'], 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if device != 'cpu' else False,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=dataset_config['batch_size'], 
            shuffle=False, 
            num_workers=1  # Minimal for evaluation
        )
        
        # Initialize model
        model = model_class().to(device)
        model_path = f'outputs/{dataset_name}/models/{model_key}_best.pth'
        
        # Check for existing model
        model_exists = Path(model_path).exists() and not args.force_retrain
        
        if model_exists and not args.skip_training:
            print(f"✓ Loading existing model from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ {model_key} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {model_key}: {e}, retraining...")
                model_exists = False
        
        # Train if needed
        if not model_exists and not args.skip_training:
            print(f"Training {model_key} on {dataset_name}...")
            
            try:
                # Choose training function
                if model_key == 'diffusion':
                    train_losses, val_losses = train_diffusion_model(
                        model, train_loader, val_loader, 
                        model_name, num_epochs=args.epochs, device=device
                    )
                elif model_key == 'vae':
                    train_losses, val_losses = train_vae_model(
                        model, train_loader, val_loader, 
                        model_name, num_epochs=args.epochs, device=device
                    )
                else:
                    train_losses, val_losses = train_model_comprehensive(
                        model, train_loader, val_loader, 
                        model_name, num_epochs=args.epochs, device=device
                    )
                
                print(f"✓ {model_key} training completed")
                
            except Exception as e:
                print(f"✗ {model_key} training failed: {e}")
                return None
        
        elif args.skip_training and not model_exists:
            print(f"⚠ Skipping {model_key} - no existing model found and training disabled")
            return None
        
        # Evaluate model
        try:
            print(f"Evaluating {model_key} on {dataset_name}...")
            
            # Test images for visualization
            test_images = [base_dataset[i][0] for i in [0, len(base_dataset)//4, len(base_dataset)//2]]
            
            metrics = compute_comprehensive_metrics(
                model, test_loader, test_images, 
                model_name, device
            )
            
            # Save results
            with open(f'outputs/{dataset_name}/results/{model_key}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"✓ {model_key} evaluation completed")
            
            # Create visualizations
            try:
                create_condition_comparison(model, test_images, model_name, device)
                create_severity_comparison(model, test_images[0], model_name, device)
                print(f"✓ {model_key} visualizations created")
            except Exception as e:
                print(f"⚠ {model_key} visualization failed: {e}")
            
            return {model_key: metrics}
            
        except Exception as e:
            print(f"✗ {model_key} evaluation failed: {e}")
            return None
    
    except Exception as e:
        print(f"✗ {model_key} processing failed: {e}")
        return None

def run_dataset_evaluation(dataset_config, model_keys, args, device):
    """Run evaluation for all selected models on one dataset"""
    
    dataset_name = dataset_config['name']
    setup_directories_for_dataset(dataset_name)
    
    print(f"\n{'#'*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"Models to process: {', '.join(model_keys)}")
    print(f"{'#'*80}")
    
    dataset_results = {}
    
    # Process each model
    for model_key in model_keys:
        if model_key in ALL_MODELS:
            model_class = ALL_MODELS[model_key]
            result = run_single_model_evaluation(
                model_key, model_class, dataset_config, args, device
            )
            if result:
                dataset_results.update(result)
        else:
            print(f"⚠ Unknown model: {model_key}")
    
    # Save comprehensive results
    if dataset_results:
        dataset_summary = {
            'dataset_info': {
                'dataset_name': dataset_name,
                'models_tested': list(dataset_results.keys()),
                'total_models': len(dataset_results),
                'samples_per_condition': args.samples_per_condition,
                'epochs': args.epochs
            },
            'model_results': dataset_results
        }
        
        with open(f'outputs/{dataset_name}/results/comprehensive_results.json', 'w') as f:
            json.dump(dataset_summary, f, indent=2)
        
        # Print results table
        print_dataset_results_table(dataset_name, dataset_results)
    
    return dataset_summary if dataset_results else None

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup
    device = setup_device(args.device)
    model_keys = resolve_models(args.models)
    dataset_choices = resolve_datasets(args.datasets)
    
    # Handle parallel execution
    if args.parallel_id is not None:
        # Split models across processes
        models_per_process = len(model_keys) // max(1, len(model_keys))
        start_idx = args.parallel_id * models_per_process
        end_idx = start_idx + models_per_process if args.parallel_id < 2 else len(model_keys)
        model_keys = model_keys[start_idx:end_idx]
        print(f"🔄 Parallel process {args.parallel_id}: handling models {model_keys}")
    
    print(f"\n🚀 PERCEPTUAL REALITY TRANSFORMER")
    print(f"📱 Optimized for M3 MacBook Pro")
    print(f"🎯 Models: {', '.join(model_keys)}")
    print(f"📊 Datasets: {', '.join(dataset_choices)}")
    print(f"⚙️  Device: {device}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Samples per condition: {args.samples_per_condition}")
    
    # Dataset configurations
    dataset_configs = []
    
    if 'imagenet' in dataset_choices:
        dataset_configs.append({
            'name': 'ImageNet',
            'choice': 'imagenet',
            'image_size': 224,
            'batch_size': 4 if device == 'mps' else args.batch_size,  # Smaller for M3
            'description': 'High-resolution natural images'
        })
    
    if 'cifar10' in dataset_choices:
        dataset_configs.append({
            'name': 'CIFAR10',
            'choice': 'cifar10', 
            'image_size': 224,
            'batch_size': 8 if device == 'mps' else args.batch_size,  # M3 optimized
            'description': 'Controlled, consistent images'
        })
    
    # Run evaluation for each dataset
    all_results = {}
    
    for dataset_config in dataset_configs:
        try:
            result = run_dataset_evaluation(dataset_config, model_keys, args, device)
            if result:
                all_results[dataset_config['name']] = result
        except Exception as e:
            print(f"✗ Dataset {dataset_config['name']} failed: {e}")
    
    # Cross-dataset analysis (only if we have multiple datasets and results)
    if len(all_results) >= 2:
        try:
            from cross_analysis import create_cross_dataset_analysis
            create_cross_dataset_analysis({k: {'results': v['model_results'], 'config': None} for k, v in all_results.items()})
            print("✓ Cross-dataset analysis completed")
        except Exception as e:
            print(f"⚠ Cross-dataset analysis failed: {e}")
    
    # Final summary
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"✅ Datasets processed: {len(all_results)}")
    total_models = sum(len(result['model_results']) for result in all_results.values())
    print(f"✅ Total model evaluations: {total_models}")
    print(f"📁 Results saved in 'outputs/' directory")
    
    if args.parallel_id is not None:
        print(f"🔄 Parallel process {args.parallel_id} completed")

if __name__ == "__main__":
    main()
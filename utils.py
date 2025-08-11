"""
Utility functions for directory setup, printing results, etc.
"""

from pathlib import Path
import numpy as np

def setup_directories_for_dataset(dataset_name):
    """Create dataset-specific directory structure"""
    dirs = [
        f'outputs/{dataset_name}',
        f'outputs/{dataset_name}/models',
        f'outputs/{dataset_name}/figures',
        f'outputs/{dataset_name}/figures/conditions',
        f'outputs/{dataset_name}/figures/severity',
        f'outputs/{dataset_name}/figures/training_curves',
        f'outputs/{dataset_name}/results',
        f'outputs/{dataset_name}/logs',
        'outputs/cross_dataset_analysis'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def print_dataset_results_table(dataset_name, results):
    """Print formatted results table for a dataset"""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} RESULTS TABLE")
    print(f"{'='*80}")
    
    if not results:
        print("No results available for this dataset")
        return
    
    print(f"{'Model':<20} {'Recon MSE':<12} {'Diversity':<10} {'Severity':<10} {'Literature':<12} {'Perceptual':<12}")
    print(f"{'-'*80}")
    
    # Sort by reconstruction quality
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1].get('reconstruction_mse', float('inf')))
    
    for model_name, metrics in sorted_results:
        recon = f"{metrics.get('reconstruction_mse', 0):.4f}"
        diversity = f"{metrics.get('condition_diversity', 0):.4f}"
        severity = f"{metrics.get('severity_scaling', 0):.4f}"
        literature = f"{metrics.get('literature_consistency', 0):.4f}"
        perceptual = f"{metrics.get('perceptual_distance', 0):.4f}" if metrics.get('perceptual_distance') else "N/A"
        
        print(f"{model_name:<20} {recon:<12} {diversity:<10} {severity:<10} {literature:<12} {perceptual:<12}")

def setup_directories():
    """Create basic directory structure"""
    dirs = [
        'outputs',
        'outputs/models',
        'outputs/figures',
        'outputs/figures/conditions',
        'outputs/figures/severity',
        'outputs/figures/training_curves',
        'outputs/results',
        'outputs/logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")
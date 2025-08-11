"""
Evaluation metrics and visualization functions
Comprehensive model evaluation and figure generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path

def compute_comprehensive_metrics(model, test_loader, test_images, model_name, device='cuda'):
    """Compute comprehensive evaluation metrics"""
    
    model.eval()
    metrics = {}
    
    print(f"Evaluating {model_name}...")
    
    # 1. Reconstruction quality
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing reconstruction quality"):
            input_images = batch['input_image'].to(device)
            condition_ids = batch['condition_id'].to(device)
            severities = batch['severity'].to(device)
            target_images = batch['target_image'].to(device)
            
            # Handle VAE models that return tuples
            model_output = model(input_images, condition_ids, severities)
            if isinstance(model_output, tuple):
                predictions = model_output[0]  # Take reconstruction from VAE
            else:
                predictions = model_output
            
            mse = F.mse_loss(predictions, target_images, reduction='sum')
            
            total_mse += mse.item()
            total_samples += predictions.size(0)
    
    metrics['reconstruction_mse'] = total_mse / total_samples
    
    # 2. Condition diversity
    diversities = []
    with torch.no_grad():
        for img in test_images[:50]:
            img = img.unsqueeze(0).to(device)
            outputs = {}
            
            for condition_id in range(8):
                condition_tensor = torch.tensor([condition_id], dtype=torch.long).to(device)
                severity_tensor = torch.tensor([0.7], dtype=torch.float32).to(device)
                
                model_output = model(img, condition_tensor, severity_tensor)
                if isinstance(model_output, tuple):
                    outputs[condition_id] = model_output[0]
                else:
                    outputs[condition_id] = model_output
            
            total_diff = 0
            pairs = 0
            for i in range(8):
                for j in range(i+1, 8):
                    diff = F.mse_loss(outputs[i], outputs[j]).item()
                    total_diff += diff
                    pairs += 1
            
            diversities.append(total_diff / pairs)
    
    metrics['condition_diversity'] = np.mean(diversities)
    
    # 3. Severity scaling
    scaling_scores = []
    with torch.no_grad():
        for img in test_images[:30]:
            img = img.unsqueeze(0).to(device)
            
            for condition_id in range(1, 8):
                severities = [0.2, 0.4, 0.6, 0.8, 1.0]
                changes = []
                
                condition_tensor = torch.tensor([condition_id], dtype=torch.long).to(device)
                
                for sev in severities:
                    severity_tensor = torch.tensor([sev], dtype=torch.float32).to(device)
                    model_output = model(img, condition_tensor, severity_tensor)
                    if isinstance(model_output, tuple):
                        output = model_output[0]
                    else:
                        output = model_output
                    
                    change = F.mse_loss(output, img).item()
                    changes.append(change)
                
                correlation = np.corrcoef(severities, changes)[0, 1]
                if not np.isnan(correlation):
                    scaling_scores.append(correlation)
    
    metrics['severity_scaling'] = np.mean(scaling_scores)
    
    # 4. Literature consistency (simplified)
    consistency_scores = []
    with torch.no_grad():
        for img in test_images[:30]:
            img = img.unsqueeze(0).to(device)
            
            # Test specific patterns
            for condition_id in [1, 5, 6]:  # simultanagnosia, depression, anxiety
                condition_tensor = torch.tensor([condition_id], dtype=torch.long).to(device)
                severity_tensor = torch.tensor([0.8], dtype=torch.float32).to(device)
                
                model_output = model(img, condition_tensor, severity_tensor)
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
                
                if condition_id == 1:  # simultanagnosia - should be fragmented
                    fragmentation = measure_fragmentation(output)
                    score = min(1.0, fragmentation * 10)  # Scale appropriately
                elif condition_id == 5:  # depression - should be darker
                    brightness_ratio = torch.mean(output) / torch.mean(img)
                    score = max(0.0, 1.0 - brightness_ratio.item())
                elif condition_id == 6:  # anxiety - should have tunnel effect
                    center_vs_edge = measure_center_bias(output)
                    score = min(1.0, center_vs_edge)
                else:
                    score = 0.5
                
                consistency_scores.append(score)
    
    metrics['literature_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5
    
    # 5. Perceptual quality (using LPIPS if available)
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        
        perceptual_distances = []
        with torch.no_grad():
            for img in test_images[:20]:
                img = img.unsqueeze(0).to(device)
                
                for condition_id in range(1, 8):
                    condition_tensor = torch.tensor([condition_id], dtype=torch.long).to(device)
                    severity_tensor = torch.tensor([0.7], dtype=torch.float32).to(device)
                    
                    model_output = model(img, condition_tensor, severity_tensor)
                    if isinstance(model_output, tuple):
                        output = model_output[0]
                    else:
                        output = model_output
                    
                    dist = lpips_model(img, output).item()
                    perceptual_distances.append(dist)
        
        metrics['perceptual_distance'] = np.mean(perceptual_distances)
    except ImportError:
        print("LPIPS not available, skipping perceptual distance calculation")
        metrics['perceptual_distance'] = None
    
    return metrics

def measure_fragmentation(image):
    """Measure image fragmentation (for simultanagnosia evaluation)"""
    # Convert to numpy
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Count connected components
    num_labels, _ = cv2.connectedComponents(edges)
    
    # Normalize by image size
    return num_labels / (image.shape[-1] * image.shape[-2])

def measure_center_bias(image):
    """Measure center bias (for anxiety evaluation)"""
    h, w = image.shape[-2:]
    center_h, center_w = h // 2, w // 2  # Fixed: was center_x
    
    # Create center and edge regions
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2).to(image.device)  # Fixed: was center_x
    
    center_mask = distance < min(h, w) // 4
    edge_mask = distance > min(h, w) // 3
    
    center_intensity = torch.mean(image[:, :, center_mask])
    edge_intensity = torch.mean(image[:, :, edge_mask])
    
    # Higher ratio means more center bias
    if edge_intensity > 0:
        return (center_intensity / edge_intensity).item()
    else:
        return 1.0

def create_condition_comparison(model, test_images, model_name, device='cuda'):
    """Create condition comparison visualization"""
    
    model.eval()
    
    # Extract dataset name for saving
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_dir = f'outputs/{dataset_name}/figures/conditions'
    else:
        save_dir = 'outputs/figures/conditions'
        pure_model_name = model_name
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for img_idx, test_image in enumerate(test_images[:3]):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        condition_names = model.condition_names
        
        with torch.no_grad():
            for i, (ax, condition_name) in enumerate(zip(axes.flat, condition_names)):
                condition_tensor = torch.tensor([i], dtype=torch.long).to(device)
                severity_tensor = torch.tensor([0.8], dtype=torch.float32).to(device)
                test_input = test_image.unsqueeze(0).to(device)
                
                model_output = model(test_input, condition_tensor, severity_tensor)
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
                
                # Convert to displayable format
                img_display = output.squeeze().cpu().permute(1, 2, 0)
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img_display = img_display * std + mean
                img_display = torch.clamp(img_display, 0, 1)
                
                ax.imshow(img_display)
                ax.set_title(condition_name.replace('_', ' ').title(), fontsize=10)
                ax.axis('off')
        
        plt.suptitle(f'{pure_model_name} - All Conditions (Image {img_idx+1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{model_name}_conditions_img{img_idx+1}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def create_severity_comparison(model, test_image, model_name, device='cuda'):
    """Create severity comparison visualization"""
    
    model.eval()
    
    # Extract dataset name for saving
    if '_' in model_name:
        dataset_name = model_name.split('_')[0]
        pure_model_name = '_'.join(model_name.split('_')[1:])
        save_dir = f'outputs/{dataset_name}/figures/severity'
    else:
        save_dir = 'outputs/figures/severity'
        pure_model_name = model_name
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Test multiple conditions
    conditions_to_test = [1, 2, 5, 6]  # simultanagnosia, prosopagnosia, depression, anxiety
    condition_names = ['Simultanagnosia', 'Prosopagnosia', 'Depression', 'Anxiety']
    
    fig, axes = plt.subplots(len(conditions_to_test), 5, figsize=(20, 4*len(conditions_to_test)))
    severities = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    with torch.no_grad():
        for cond_idx, condition_id in enumerate(conditions_to_test):
            for sev_idx, severity in enumerate(severities):
                condition_tensor = torch.tensor([condition_id], dtype=torch.long).to(device)
                severity_tensor = torch.tensor([severity], dtype=torch.float32).to(device)
                test_input = test_image.unsqueeze(0).to(device)
                
                model_output = model(test_input, condition_tensor, severity_tensor)
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
                
                # Convert to displayable format
                img_display = output.squeeze().cpu().permute(1, 2, 0)
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img_display = img_display * std + mean
                img_display = torch.clamp(img_display, 0, 1)
                
                axes[cond_idx, sev_idx].imshow(img_display)
                if cond_idx == 0:
                    axes[cond_idx, sev_idx].set_title(f'Severity: {severity:.1f}', fontsize=10)
                if sev_idx == 0:
                    axes[cond_idx, sev_idx].set_ylabel(condition_names[cond_idx], fontsize=12)
                axes[cond_idx, sev_idx].axis('off')
    
    plt.suptitle(f'{pure_model_name} - Severity Progression', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_severity_progression.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
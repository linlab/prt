"""
Perturbation functions for simulating neurological conditions
Each function takes an image and severity, returns perturbed image
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

def denormalize_image(image):
    """Convert from ImageNet normalized to [0,1] range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
    return image * std + mean

def renormalize_image(image):
    """Convert from [0,1] back to ImageNet normalized"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
    return (image - mean) / std

def fragment_image_strong(image, severity):
    """Strong simultanagnosia simulation - show only fragments"""
    img_denorm = denormalize_image(image)
    batch_size = img_denorm.shape[0]
    h, w = img_denorm.shape[-2:]
    
    fragmented = torch.zeros_like(img_denorm)
    num_fragments = max(2, int((1 - severity) * 8))
    
    for b in range(batch_size):
        for _ in range(num_fragments):
            frag_size = int(30 + (1-severity) * 50)
            y = random.randint(0, max(1, h - frag_size))
            x = random.randint(0, max(1, w - frag_size))
            fragmented[b, :, y:y+frag_size, x:x+frag_size] = \
                img_denorm[b, :, y:y+frag_size, x:x+frag_size]
    
    return renormalize_image(torch.clamp(fragmented, 0, 1))

def prosopagnosia_strong(image, severity):
    """Strong prosopagnosia - scramble face regions"""
    img_denorm = denormalize_image(image)
    h, w = img_denorm.shape[-2:]
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    
    face_mask = ((y - center_y)**2 / (h//3)**2 + (x - center_x)**2 / (w//4)**2) < 1
    face_mask = face_mask.float().unsqueeze(0).unsqueeze(0).to(image.device)
    
    scrambled = img_denorm.clone()
    if severity > 0.3:
        face_noise = torch.randn_like(img_denorm) * severity * 0.3
        scrambled = img_denorm + face_mask * face_noise
        
        if severity > 0.5:
            kernel_size = int(severity * 20) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            blur = transforms.GaussianBlur(kernel_size, sigma=severity * 5)
            blurred_version = blur(img_denorm)
            scrambled = face_mask * blurred_version + (1 - face_mask) * img_denorm
    
    return renormalize_image(torch.clamp(scrambled, 0, 1))

def adhd_attention_strong(image, severity):
    """Strong ADHD simulation - add many distractors"""
    img_denorm = denormalize_image(image)
    distracted = img_denorm.clone()
    
    num_distractors = int(severity * 100)
    for _ in range(num_distractors):
        y = random.randint(0, img_denorm.shape[-2] - 1)
        x = random.randint(0, img_denorm.shape[-1] - 1)
        distracted[:, :, y, x] = torch.rand(3).to(image.device)
    
    noise = torch.randn_like(img_denorm) * severity * 0.2
    distracted += noise
    distracted = distracted * (1 + severity * 0.5)
    
    return renormalize_image(torch.clamp(distracted, 0, 1))

def visual_agnosia_strong(image, severity):
    """Strong visual agnosia - objects become unrecognizable"""
    img_denorm = denormalize_image(image)
    distorted = img_denorm.clone()
    
    noise = torch.randn_like(img_denorm) * severity * 0.4
    
    if severity > 0.4:
        h, w = img_denorm.shape[-2:]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(image.device)
        
        distortion = torch.randn_like(grid) * severity * 0.1
        warped_grid = grid + distortion
        
        for b in range(img_denorm.shape[0]):
            distorted[b] = F.grid_sample(img_denorm[b:b+1], warped_grid, align_corners=False)[0]
    
    distorted += noise
    return renormalize_image(torch.clamp(distorted, 0, 1))

def depression_strong(image, severity):
    """Strong depression simulation - very dark, desaturated"""
    img_denorm = denormalize_image(image)
    
    darkened = img_denorm * (1 - severity * 0.7)
    gray = torch.mean(darkened, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    desaturated = (1 - severity * 0.8) * darkened + severity * 0.8 * gray
    
    blue_tint = torch.zeros_like(desaturated)
    blue_tint[:, 2] = severity * 0.2
    desaturated += blue_tint
    
    return renormalize_image(torch.clamp(desaturated, 0, 1))

def anxiety_tunnel_strong(image, severity):
    """Strong anxiety simulation - severe tunnel vision"""
    img_denorm = denormalize_image(image)
    h, w = img_denorm.shape[-2:]
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    distance = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    max_radius = min(h, w) // 2
    tunnel_radius = max_radius * (1 - severity * 0.8)
    
    tunnel_mask = torch.exp(-(distance - tunnel_radius)**2 / (tunnel_radius * 0.3)**2)
    tunnel_mask = torch.clamp(tunnel_mask, 0, 1)
    tunnel_mask = tunnel_mask.unsqueeze(0).unsqueeze(0).to(image.device)
    
    tunneled = img_denorm * tunnel_mask
    
    edge_mask = 1 - tunnel_mask
    red_tint = torch.zeros_like(img_denorm)
    red_tint[:, 0] = edge_mask.squeeze() * severity * 0.3
    tunneled += red_tint
    
    return renormalize_image(torch.clamp(tunneled, 0, 1))

def alzheimer_strong(image, severity):
    """Strong Alzheimer simulation - severe memory degradation"""
    img_denorm = denormalize_image(image)
    degraded = img_denorm.clone()
    
    kernel_size = int(severity * 25) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if kernel_size > 1:
        blur = transforms.GaussianBlur(kernel_size, sigma=severity * 8)
        degraded = blur(degraded)
    
    memory_noise = torch.randn_like(img_denorm) * severity * 0.6
    degraded += memory_noise
    
    faded = degraded * (1 - severity * 0.3) + severity * 0.3 * 0.5
    
    return renormalize_image(torch.clamp(faded, 0, 1))

# Perturbation function mapping
PERTURBATION_FUNCTIONS = {
    0: lambda x, s: x,  # normal
    1: fragment_image_strong,     # simultanagnosia
    2: prosopagnosia_strong,      # prosopagnosia
    3: adhd_attention_strong,     # adhd_attention
    4: visual_agnosia_strong,     # visual_agnosia
    5: depression_strong,         # depression_mood
    6: anxiety_tunnel_strong,     # anxiety_tunnel
    7: alzheimer_strong          # alzheimer_memory
}
"""
Model architectures for Perceptual Reality Transformer
Includes CNN, residual, ViT, recurrent, diffusion, and VAE models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from perturbations import PERTURBATION_FUNCTIONS

class EncoderDecoderCNN(nn.Module):
    """Encoder-decoder CNN"""
    def __init__(self, num_conditions=8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),    # 224->112
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),        # 112->56
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 56->28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 28->14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # 14->7
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        
        self.condition_embed = nn.Embedding(num_conditions, 256)
        self.severity_embed = nn.Linear(1, 256)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + 256, 256, 4, stride=2, padding=1), # 7->14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),        # 14->28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),         # 28->56
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),          # 56->112
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),           # 112->224
            nn.Tanh()
        )
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, image, condition_id, severity):
        features = self.encoder(image)
        
        cond_emb = self.condition_embed(condition_id)
        sev_emb = self.severity_embed(severity.unsqueeze(-1))
        combined_cond = cond_emb * sev_emb
        
        h, w = features.shape[2], features.shape[3]
        spatial_cond = combined_cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        combined_features = torch.cat([features, spatial_cond], dim=1)
        output = self.decoder(combined_features)
        
        if output.shape[-1] != image.shape[-1]:
            output = F.interpolate(output, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return output

class ResidualPerceptual(nn.Module):
    """Residual model that adds perturbations to original image"""
    def __init__(self, num_conditions=8):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        
        self.condition_embed = nn.Embedding(num_conditions, 64)
        self.severity_embed = nn.Linear(1, 64)
        
        self.perturbation_net = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh()
        )
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
    
    def forward(self, image, condition_id, severity):
        img_features = self.feature_net(image)
        
        cond_emb = self.condition_embed(condition_id)
        sev_emb = self.severity_embed(severity.unsqueeze(-1))
        combined_cond = cond_emb * sev_emb
        
        h, w = img_features.shape[2], img_features.shape[3]
        spatial_cond = combined_cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        combined = torch.cat([img_features, spatial_cond], dim=1)
        perturbation = self.perturbation_net(combined)
        
        severity_scale = severity.view(-1, 1, 1, 1)
        scaled_perturbation = perturbation * severity_scale
        
        output = image + scaled_perturbation * 0.5
        
        normal_mask = (condition_id == 0).float().view(-1, 1, 1, 1)
        output = normal_mask * image + (1 - normal_mask) * output
        
        return torch.clamp(output, -3, 3)

class ViTPerceptual(nn.Module):
    """Vision Transformer based model"""
    def __init__(self, num_conditions=8):
        super().__init__()
        
        # Use pretrained ViT
        self.vit_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, features_only=True)
        
        self.condition_embed = nn.Embedding(num_conditions, 768)
        self.severity_embed = nn.Linear(1, 768)
        
        # Decoder for ViT features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768 + 768, 512, 4, stride=2, padding=1),  # 14->28
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),        # 28->56
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),        # 56->112
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),         # 112->224
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh()
        )
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
    
    def forward(self, image, condition_id, severity):
        features = self.vit_encoder(image)
        vit_features = features[-1]  # Last layer features
        
        cond_emb = self.condition_embed(condition_id)
        sev_emb = self.severity_embed(severity.unsqueeze(-1))
        combined_cond = cond_emb * sev_emb
        
        h, w = vit_features.shape[2], vit_features.shape[3]
        spatial_cond = combined_cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        combined_features = torch.cat([vit_features, spatial_cond], dim=1)
        output = self.decoder(combined_features)
        
        if output.shape[-1] != image.shape[-1]:
            output = F.interpolate(output, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return output

class RecurrentPerceptual(nn.Module):
    """Recurrent model for progressive perturbation"""
    def __init__(self, num_conditions=8):
        super().__init__()
        
        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),    # 224->112
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),        # 112->56
        )
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(64 * 56 * 56, 1024, batch_first=True)
        
        self.condition_embed = nn.Embedding(num_conditions, 256)
        self.severity_embed = nn.Linear(1, 256)
        
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 256, 64 * 56 * 56),
            nn.ReLU()
        )
        
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 56->112
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 112->224
            nn.Tanh()
        )
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
    
    def forward(self, image, condition_id, severity):
        batch_size = image.size(0)
        
        # Extract CNN features
        cnn_feats = self.cnn_features(image)  # [B, 64, 56, 56]
        
        # Flatten for LSTM
        flattened = cnn_feats.view(batch_size, 1, -1)  # [B, 1, 64*56*56]
        
        # LSTM processing
        lstm_out, _ = self.lstm(flattened)  # [B, 1, 1024]
        lstm_out = lstm_out.squeeze(1)  # [B, 1024]
        
        # Condition embedding
        cond_emb = self.condition_embed(condition_id)
        sev_emb = self.severity_embed(severity.unsqueeze(-1))
        combined_cond = cond_emb * sev_emb
        
        # Combine LSTM output with condition
        combined = torch.cat([lstm_out, combined_cond], dim=1)
        decoded = self.decoder(combined)
        
        # Reshape and upsample
        decoded = decoded.view(batch_size, 64, 56, 56)
        output = self.upsampler(decoded)
        
        if output.shape[-1] != image.shape[-1]:
            output = F.interpolate(output, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return output

class DiffusionPerceptual(nn.Module):
    """Diffusion-based model for perceptual reality simulation"""
    def __init__(self, num_conditions=8, timesteps=100):
        super().__init__()
        
        self.timesteps = timesteps
        
        # Simplified U-Net for space efficiency
        self.down1 = nn.Sequential(
            nn.Conv2d(3 + num_conditions + 1, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256), nn.SiLU()
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.SiLU()
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU()
        )
        self.up1 = nn.Conv2d(64 + 64, 3, 3, padding=1)
        
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        self.condition_embed = nn.Embedding(num_conditions, 64)
        self.severity_embed = nn.Linear(1, 64)
        
        # Noise schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
    
    def positional_encoding(self, timesteps, dim=128):
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def forward(self, x, condition_id, severity, timesteps=None):
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # Create condition maps
        batch_size, _, h, w = x.shape
        cond_map = torch.zeros(batch_size, len(self.condition_names), h, w, device=x.device)
        for i, cid in enumerate(condition_id):
            cond_map[i, cid.item()] = 1.0
        
        sev_map = severity.view(-1, 1, 1, 1).expand(-1, 1, h, w)
        x_cond = torch.cat([x, cond_map, sev_map], dim=1)
        
        # U-Net forward pass
        d1 = self.down1(x_cond)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        m = self.middle(d3)
        
        u3 = self.up3(torch.cat([m, d3], dim=1))
        u2 = self.up2(torch.cat([u3, d2], dim=1))
        output = self.up1(torch.cat([u2, d1], dim=1))
        
        return output
    
    def add_noise(self, x, noise, timesteps):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise

class GenerativePerceptual(nn.Module):
    """VAE-based generative model"""
    def __init__(self, num_conditions=8, latent_dim=512):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 224->112
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 112->56
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 56->28
            nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 28->14
            nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1), # 14->7
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
        
        # Condition embedding
        self.condition_embed = nn.Embedding(num_conditions, 128)
        self.severity_embed = nn.Linear(1, 128)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim + 256, 512 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1), # 7->14
            nn.ReLU(), nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 14->28
            nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 28->56
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 56->112
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),    # 112->224
            nn.Tanh()
        )
        
        self.condition_names = [
            'normal', 'simultanagnosia', 'prosopagnosia', 'adhd_attention',
            'visual_agnosia', 'depression_mood', 'anxiety_tunnel', 'alzheimer_memory'
        ]
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition_id, severity):
        cond_emb = self.condition_embed(condition_id)
        sev_emb = self.severity_embed(severity.unsqueeze(-1))
        combined_cond = torch.cat([cond_emb, sev_emb], dim=1)
        
        z_cond = torch.cat([z, combined_cond], dim=1)
        h = self.fc_decode(z_cond)
        h = h.view(h.size(0), 512, 7, 7)
        return self.decoder(h)
    
    def forward(self, x, condition_id, severity):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition_id, severity)
        return recon, mu, logvar
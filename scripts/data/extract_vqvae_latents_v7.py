"""提取 VQ-VAE v7 latent codes"""
import sys, json, numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from models.vq_vae_v2 import VQVAEv2

ckpt_path = project_root / 'checkpoints/vqvae_v7/vqvae_v7_best.pth'
data_dir = project_root / 'datasets/classified/pixel_32_quantized'
output_dir = project_root / 'datasets/vqvae_latent_data_v7'
output_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda'
ckpt = torch.load(ckpt_path, map_location=device)
config = ckpt.get('config', {})
latent_size = config.get('latent_size', 16)

model = VQVAEv2(
    in_channels=config.get('in_channels', 4),
    hidden_channels=config.get('hidden_channels', 256),
    embedding_dim=config.get('embedding_dim', 64),
    num_embeddings=config.get('num_embeddings', 256),
    latent_size=latent_size,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
image_files = sorted([f for f in data_dir.iterdir() if f.suffix == '.png'])
print(f'Found {len(image_files)} images, latent_size={latent_size}')

all_indices, all_latents, all_names = [], [], []
with torch.no_grad():
    for i, img_path in enumerate(image_files):
        img = Image.open(img_path).convert('RGBA')
        img_tensor = transform(img).unsqueeze(0).to(device)
        z_q, indices = model.encode(img_tensor)
        indices_reshaped = indices.view(1, latent_size, latent_size)
        all_indices.append(indices_reshaped.cpu().numpy())
        all_latents.append(z_q.cpu().numpy())
        all_names.append(img_path.name)
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(image_files)}')

all_indices = np.concatenate(all_indices, axis=0)
all_latents = np.concatenate(all_latents, axis=0)
np.save(output_dir / 'indices.npy', all_indices)
np.save(output_dir / 'latents.npy', all_latents)
with open(output_dir / 'names.json', 'w') as f:
    json.dump(all_names, f)
with open(output_dir / 'config.json', 'w') as f:
    json.dump({'vqvae_checkpoint': str(ckpt_path), 'num_samples': len(all_indices), 'indices_shape': list(all_indices.shape[1:]), 'latent_shape': list(all_latents.shape[1:]), 'latent_size': latent_size, 'source': str(data_dir)}, f, indent=2)
print(f'Done: indices={all_indices.shape}, latents={all_latents.shape}')

"""测试VQ-VAE v5模型"""

import sys
sys.path.insert(0, '.')

from models.vq_vae_v2 import VQVAEv2
import torch

model = VQVAEv2(in_channels=4, hidden_channels=256, embedding_dim=64, num_embeddings=256)
x = torch.randn(4, 4, 32, 32)
x_recon, vq_loss, indices = model(x)

print(f'VQ-VAE v5:')
print(f'  输入: {x.shape}')
print(f'  输出: {x_recon.shape}')
print(f'  索引: {indices.shape}')
print(f'  Latent: 16×16 = {16*16} tokens')
print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')

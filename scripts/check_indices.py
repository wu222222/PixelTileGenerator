"""检查indices数据"""

import numpy as np

indices = np.load('datasets/vqvae_latent_data/indices.npy', allow_pickle=True)
print(f'Shape: {indices.shape}')
print(f'Type: {indices.dtype}')
print(f'First: {indices[0].shape}')
print(f'Second: {indices[1].shape}')
print(f'First few values: {indices[0][:10]}')

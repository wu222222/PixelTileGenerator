"""
ZQ Latent GAN — 在 VQ-VAE 的 z_q 空间上训练 GAN

Pipeline:
  noise → G(z) → z_q_normalized → denormalize → nearest codebook → decoder → tile

改进 (v2):
- Spectral Norm 判别器 (稳定 Lipschitz)
- Hinge Loss (替代 WGAN-GP)
- z_q 标准化 (训练前计算 mean/std)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ZQGenerator(nn.Module):
    """z_q 生成器

    输入: noise [B, noise_dim]
    输出: z_q_normalized [B, embed_dim, latent_size, latent_size]
    """

    def __init__(self, noise_dim=128, embed_dim=128, latent_size=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_size = latent_size

        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            # 无 Tanh: 让 G 输出无界值，D 学习实际范围
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv(x)


class ZQDiscriminator(nn.Module):
    """z_q 判别器 (Spectral Norm)

    输入: z_q [B, embed_dim, latent_size, latent_size]
    输出: score [B, 1]
    """

    def __init__(self, embed_dim=128, latent_size=16):
        super().__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(embed_dim, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(256 * 4 * 4, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, z_q):
        x = self.conv(z_q)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ZQLatentGAN(nn.Module):
    """ZQ Latent GAN"""

    def __init__(self, noise_dim=128, embed_dim=128, latent_size=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.latent_size = latent_size

        self.generator = ZQGenerator(noise_dim, embed_dim, latent_size)
        self.discriminator = ZQDiscriminator(embed_dim, latent_size)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, z_q):
        return self.discriminator(z_q)


def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*40}")
    print(f"ZQ Latent GAN v2 (SN + Hinge)")
    print(f"{'='*40}")

    model = ZQLatentGAN(noise_dim=128, embed_dim=128, latent_size=16).to(device)

    batch_size = 4
    noise = torch.randn(batch_size, 128).to(device)
    z_q_fake = model.generate(noise)
    print(f"  噪声: {noise.shape}")
    print(f"  生成 z_q: {z_q_fake.shape}")
    print(f"  生成范围: [{z_q_fake.min():.3f}, {z_q_fake.max():.3f}]")

    score = model.discriminate(z_q_fake)
    print(f"  判别器: {score.shape}")

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"  G 参数量: {g_params:,}")
    print(f"  D 参数量: {d_params:,}")


if __name__ == "__main__":
    test_model()

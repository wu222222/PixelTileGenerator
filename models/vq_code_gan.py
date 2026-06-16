"""
VQ Code GAN 模型

功能:
- 学习离散 code indices 的分布
- 生成新的 indices
- 用 codebook lookup 得到 z_q
- 用 Decoder 生成图片

架构 (v2 - 卷积生成器):
- Generator: noise → FC → reshape → ConvTranspose → logits [B, num_embeddings, 16, 16]
- Discriminator: one-hot → Conv → FC → score
"""

import torch
import torch.nn as nn


class CodeGenerator(nn.Module):
    """Code 索引生成器 (卷积架构)"""

    def __init__(self, num_embeddings=1024, latent_size=16):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        self.fc = nn.Sequential(
            nn.Linear(num_embeddings, 512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 logits
            nn.Conv2d(128, num_embeddings, kernel_size=3, padding=1),
        )

    def forward(self, z):
        """
        Args:
            z: [batch, num_embeddings] 噪声
        Returns:
            logits: [batch, num_embeddings, latent_size, latent_size]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv(x)

    def generate(self, z):
        """生成 indices"""
        logits = self.forward(z)
        indices = torch.argmax(logits, dim=1)  # [batch, latent_size, latent_size]
        return indices


class CodeDiscriminator(nn.Module):
    """Code 索引判别器"""

    def __init__(self, num_embeddings=1024, latent_size=16, input_noise=0.3):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size
        self.input_noise = input_noise

        self.conv = nn.Sequential(
            # 16x16 -> 8x8
            nn.Conv2d(num_embeddings, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([256, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, indices):
        """
        Args:
            indices: [batch, latent_size, latent_size] 离散索引
        Returns:
            validity: [batch, 1]
        """
        one_hot = torch.zeros(indices.size(0), self.num_embeddings, indices.size(1), indices.size(2)).to(indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)

        # 给 one-hot 加噪声，打破离散性，防止 D 秒分
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(one_hot) * self.input_noise
            one_hot = one_hot + noise

        x = self.conv(one_hot)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class VQCodeGAN(nn.Module):
    """VQ Code GAN: 生成离散 code indices"""

    def __init__(self, num_embeddings=1024, latent_size=16):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        self.generator = CodeGenerator(num_embeddings, latent_size)
        self.discriminator = CodeDiscriminator(num_embeddings, latent_size)

    def generate(self, z):
        """生成 indices"""
        return self.generator.generate(z)

    def discriminate(self, indices):
        """判别"""
        return self.discriminator(indices)


def test_model():
    """测试模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for num_emb in [256, 1024]:
        print(f"\n{'='*40}")
        print(f"VQ Code GAN (num_embeddings={num_emb})")
        print(f"{'='*40}")

        model = VQCodeGAN(num_embeddings=num_emb, latent_size=16).to(device)

        batch_size = 4
        z = torch.randn(batch_size, num_emb).to(device)
        fake_indices = torch.randint(0, num_emb, (batch_size, 16, 16)).to(device)

        generated_indices = model.generate(z)
        print(f"  输入噪声: {z.shape}")
        print(f"  生成索引: {generated_indices.shape}")
        print(f"  索引范围: [{generated_indices.min().item()}, {generated_indices.max().item()}]")

        validity = model.discriminate(fake_indices)
        print(f"  判别器输出: {validity.shape}")

        g_params = sum(p.numel() for p in model.generator.parameters())
        d_params = sum(p.numel() for p in model.discriminator.parameters())
        print(f"  生成器参数量: {g_params:,}")
        print(f"  判别器参数量: {d_params:,}")


if __name__ == "__main__":
    test_model()

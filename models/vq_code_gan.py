"""
VQ Code GAN 模型

功能:
- 学习离散code indices的分布
- 生成新的indices
- 用codebook lookup得到z_q
- 用Decoder生成图片
"""

import torch
import torch.nn as nn


class CodeGenerator(nn.Module):
    """Code索引生成器"""

    def __init__(self, num_embeddings=256, latent_size=16):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size
        self.total_tokens = latent_size * latent_size

        # 输入: one-hot编码的indices [batch, num_embeddings, latent_size, latent_size]
        # 或者: noise [batch, num_embeddings]

        self.fc = nn.Sequential(
            nn.Linear(num_embeddings, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.total_tokens * num_embeddings),
        )

    def forward(self, z):
        """
        Args:
            z: [batch, num_embeddings] 噪声
        Returns:
            logits: [batch, num_embeddings, latent_size, latent_size]
        """
        batch_size = z.size(0)
        logits_flat = self.fc(z)
        return logits_flat.view(batch_size, self.num_embeddings, self.latent_size, self.latent_size)

    def generate(self, z):
        """生成indices"""
        logits = self.forward(z)
        indices = torch.argmax(logits, dim=1)  # [batch, latent_size, latent_size]
        return indices


class CodeDiscriminator(nn.Module):
    """Code索引判别器"""

    def __init__(self, num_embeddings=256, latent_size=16):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        # 输入: one-hot编码的indices [batch, num_embeddings, latent_size, latent_size]
        self.conv = nn.Sequential(
            # 16×16 → 8×8
            nn.Conv2d(num_embeddings, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 8×8 → 4×4
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
        # 转换为one-hot编码
        one_hot = torch.zeros(indices.size(0), self.num_embeddings, indices.size(1), indices.size(2)).to(indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)

        # 提取特征
        x = self.conv(one_hot)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class VQCodeGAN(nn.Module):
    """VQ Code GAN: 生成离散code indices"""

    def __init__(self, num_embeddings=256, latent_size=16):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        self.generator = CodeGenerator(num_embeddings, latent_size)
        self.discriminator = CodeDiscriminator(num_embeddings, latent_size)

    def generate(self, z):
        """生成indices"""
        return self.generator.generate(z)

    def discriminate(self, indices):
        """判别"""
        return self.discriminator(indices)


def test_model():
    """测试模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = VQCodeGAN(num_embeddings=256, latent_size=16).to(device)

    # 测试输入
    batch_size = 4
    z = torch.randn(batch_size, 256).to(device)
    fake_indices = torch.randint(0, 256, (batch_size, 16, 16)).to(device)

    # 测试生成器
    generated_indices = model.generate(z)
    print(f"VQ Code GAN:")
    print(f"  输入噪声: {z.shape}")
    print(f"  生成索引: {generated_indices.shape}")
    print(f"  索引范围: [{generated_indices.min().item()}, {generated_indices.max().item()}]")

    # 测试判别器
    validity = model.discriminate(fake_indices)
    print(f"  判别器输出: {validity.shape}")

    # 参数量
    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\n  生成器参数量: {g_params:,}")
    print(f"  判别器参数量: {d_params:,}")


if __name__ == "__main__":
    test_model()

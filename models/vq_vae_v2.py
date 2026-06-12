"""
VQ-VAE v2 模型

改进:
- 8×8 latent (64 tokens)
- GroupNorm (替代BatchNorm)
- Residual Blocks
- 更大的隐藏通道
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    """向量量化层"""

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """
        Args:
            z: [batch, channels, height, width] 编码器输出
        Returns:
            z_q: 量化后的向量
            loss: 量化损失
            indices: 码本索引
        """
        # 转换为 [batch, height, width, channels]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # 计算距离
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 找到最近的码本向量
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # 计算损失
        q_latent_loss = F.mse_loss(z_q.detach(), z)
        e_latent_loss = F.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计器（梯度传播）
        z_q = z + (z_q - z).detach()

        # 转换回 [batch, channels, height, width]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, encoding_indices

    def get_codebook_usage(self, indices):
        """统计码本使用率"""
        unique_indices = torch.unique(indices)
        usage = len(unique_indices) / self.num_embeddings
        return usage, unique_indices


class Encoder(nn.Module):
    """编码器 (32×32 → latent_size×latent_size)"""

    def __init__(self, in_channels=4, hidden_channels=256, embedding_dim=64, latent_size=16):
        super().__init__()
        self.latent_size = latent_size

        self.conv = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # 16×16 → latent_size×latent_size (仅当 latent_size != 16 时生效)
        self.pool = nn.AdaptiveAvgPool2d((latent_size, latent_size)) if latent_size != 16 else nn.Identity()

        # 残差块
        self.residual = nn.Sequential(
            ResidualBlock(hidden_channels // 2),
            ResidualBlock(hidden_channels // 2),
        )

        # 输出层
        self.output = nn.Conv2d(hidden_channels // 2, embedding_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.residual(x)
        x = self.output(x)
        return x


class Decoder(nn.Module):
    """解码器 (latent_size×latent_size → 32×32)"""

    def __init__(self, embedding_dim=64, hidden_channels=256, out_channels=4, latent_size=16):
        super().__init__()
        self.latent_size = latent_size

        # 输入层
        self.input = nn.Conv2d(embedding_dim, hidden_channels // 2, kernel_size=1)

        # 残差块
        self.residual = nn.Sequential(
            ResidualBlock(hidden_channels // 2),
            ResidualBlock(hidden_channels // 2),
        )

        self.conv = nn.Sequential(
            # 16×16 → 32×32
            nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.residual(x)
        # latent_size×latent_size → 16×16 (仅当 latent_size != 16 时)
        if self.latent_size != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class VQVAEv2(nn.Module):
    """VQ-VAE v2 模型"""

    def __init__(self, in_channels=4, hidden_channels=256, embedding_dim=64,
                 num_embeddings=512, commitment_cost=0.25, latent_size=16):
        super().__init__()
        self.latent_size = latent_size

        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim, latent_size)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels, latent_size)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        # 编码
        z = self.encoder(x)

        # 量化
        z_q, vq_loss, indices = self.vq(z)

        # 解码
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, indices

    def encode(self, x):
        """编码并量化"""
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices

    def decode(self, z_q):
        """解码"""
        return self.decoder(z_q)

    def get_indices(self, x):
        """获取码本索引"""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    def get_codebook_usage(self, x):
        """统计码本使用率"""
        indices = self.get_indices(x)
        return self.vq.get_codebook_usage(indices)


def test_model():
    """测试模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for latent_size in [16, 20]:
        print(f"\n{'='*40}")
        print(f"测试 latent_size={latent_size}")
        print(f"{'='*40}")

        model = VQVAEv2(
            in_channels=4,
            hidden_channels=256,
            embedding_dim=64,
            num_embeddings=512,
            latent_size=latent_size,
        ).to(device)

        x = torch.randn(4, 4, 32, 32).to(device)
        x_recon, vq_loss, indices = model(x)

        print(f"  输入: {x.shape}")
        print(f"  输出: {x_recon.shape}")
        print(f"  索引: {indices.shape}")
        print(f"  VQ损失: {vq_loss.item():.4f}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        z_q, indices = model.encode(x)
        x_recon2 = model.decode(z_q)
        print(f"  量化后: {z_q.shape}")
        print(f"  重建: {x_recon2.shape}")

        usage, unique = model.get_codebook_usage(x)
        print(f"  码本使用率: {usage:.2%} ({len(unique)}/512)")


if __name__ == "__main__":
    test_model()

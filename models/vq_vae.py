"""
VQ-VAE 模型

功能:
- 学习离散码本
- 天然输出像素风格图片
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, in_channels=4, hidden_channels=128, embedding_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 8×8 → 4×4
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 4×4 → embedding_dim
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, embedding_dim=64, hidden_channels=128, out_channels=4):
        super().__init__()

        self.conv = nn.Sequential(
            # embedding_dim → hidden_channels
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class VQVAE(nn.Module):
    """VQ-VAE模型"""

    def __init__(self, in_channels=4, hidden_channels=128, embedding_dim=64,
                 num_embeddings=512, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)
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

    def reconstruct_from_indices(self, indices, height=4, width=4):
        """从索引重建"""
        z_q = self.vq.embedding(indices)
        z_q = z_q.view(-1, self.vq.embedding_dim, height, width)
        return self.decoder(z_q)


def test_model():
    """测试模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = VQVAE(
        in_channels=4,
        hidden_channels=128,
        embedding_dim=64,
        num_embeddings=512,
    ).to(device)

    # 测试输入
    x = torch.randn(4, 4, 32, 32).to(device)

    # 前向传播
    x_recon, vq_loss, indices = model(x)

    print(f"VQ-VAE:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {x_recon.shape}")
    print(f"  索引: {indices.shape}")
    print(f"  VQ损失: {vq_loss.item():.4f}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试编码和解码
    z_q, indices = model.encode(x)
    x_recon2 = model.decode(z_q)
    print(f"\n编码-解码:")
    print(f"  量化后: {z_q.shape}")
    print(f"  重建: {x_recon2.shape}")


if __name__ == "__main__":
    test_model()

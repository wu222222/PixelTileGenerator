"""
CNN AutoEncoder 模型

功能:
- 学习32×32像素瓦片的latent space
- 编码器: 32×32 → latent
- 解码器: latent → 32×32
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """编码器: 32×32 → latent_dim"""

    def __init__(self, latent_dim=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4×4 → 2×2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """解码器: latent_dim → 32×32"""

    def __init__(self, latent_dim=64):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)

        self.conv_layers = nn.Sequential(
            # 2×2 → 4×4
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 2, 2)
        x = self.conv_layers(x)
        return x


class AutoEncoder(nn.Module):
    """AutoEncoder: 编码器 + 解码器"""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, x):
        """只编码"""
        return self.encoder(x)

    def decode(self, z):
        """只解码"""
        return self.decoder(z)

    def get_latent(self, x):
        """获取latent vector"""
        return self.encoder(x)


def test_model():
    """测试模型"""
    # 创建模型
    model = AutoEncoder(latent_dim=128)

    # 测试输入
    x = torch.randn(4, 4, 32, 32)  # batch=4, channels=4 (RGBA), 32×32

    # 前向传播
    reconstruction = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {reconstruction.shape}")
    print(f"Latent维度: {model.latent_dim}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_model()

"""
ResNet AutoEncoder 模型

功能:
- 使用残差连接保留更多细节
- 学习32×32像素瓦片的latent space
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """编码器: 32×32 → latent_dim"""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 下采样 + 残差块
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )  # 32 → 16

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )  # 16 → 8

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResBlock(512),
        )  # 8 → 4

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """解码器: latent_dim → 32×32"""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # 上采样 + 残差块
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )  # 4 → 8

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )  # 8 → 16

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )  # 16 → 32

        self.final = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        return x


class ResNetAutoEncoder(nn.Module):
    """ResNet AutoEncoder: 编码器 + 解码器"""

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
    model = ResNetAutoEncoder(latent_dim=128)

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

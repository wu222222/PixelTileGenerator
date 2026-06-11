"""
Structure Generator 模型

功能:
- 生成Structure Map（索引图）
- 输出32×32×32的logits
- 通过argmax得到0~31的索引
"""

import torch
import torch.nn as nn


class StructureGenerator(nn.Module):
    """Structure Generator: 生成索引图"""

    def __init__(self, latent_dim=128, num_classes=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 2 * 2),
            nn.ReLU(inplace=True),
        )

        # 上采样层
        self.conv = nn.Sequential(
            # 2×2 → 4×4
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 输出logits，不加softmax
        )

    def forward(self, z):
        """
        Args:
            z: 随机噪声 [batch_size, latent_dim]
        Returns:
            logits: [batch_size, num_classes, 32, 32]
        """
        x = self.fc(z)
        x = x.view(-1, 512, 2, 2)
        x = self.conv(x)
        return x

    def generate(self, z):
        """生成索引图"""
        logits = self.forward(z)
        indices = torch.argmax(logits, dim=1)  # [batch_size, 32, 32]
        return indices

    def generate_with_palette(self, z, palette):
        """生成带调色板的RGB图像"""
        indices = self.generate(z)
        return self.indices_to_rgb(indices, palette)

    def indices_to_rgb(self, indices, palette):
        """将索引图转换为RGB图像"""
        batch_size = indices.size(0)
        h, w = indices.size(1), indices.size(2)

        # 创建RGB图像
        rgb = torch.zeros(batch_size, 3, h, w).to(indices.device)

        for b in range(batch_size):
            for y in range(h):
                for x in range(w):
                    idx = indices[b, y, x].item()
                    if idx < len(palette):
                        rgb[b, 0, y, x] = palette[idx][0] / 255.0
                        rgb[b, 1, y, x] = palette[idx][1] / 255.0
                        rgb[b, 2, y, x] = palette[idx][2] / 255.0

        return rgb


class StructureDiscriminator(nn.Module):
    """Structure Discriminator: 判断索引图是否真实"""

    def __init__(self, num_classes=32):
        super().__init__()

        self.num_classes = num_classes

        # 输入是one-hot编码的索引图 [batch, 32, 32, 32]
        self.conv = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(num_classes, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 8×8 → 4×4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([256, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 4×4 → 2×2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([512, 2, 2]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, indices):
        """
        Args:
            indices: [batch_size, 32, 32] 索引图
        Returns:
            validity: [batch_size, 1]
        """
        # 转换为one-hot编码
        one_hot = torch.zeros(indices.size(0), self.num_classes, indices.size(1), indices.size(2)).to(indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)

        # 提取特征
        x = self.conv(one_hot)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_models():
    """测试模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    generator = StructureGenerator(latent_dim=128, num_classes=32).to(device)
    discriminator = StructureDiscriminator(num_classes=32).to(device)

    # 测试输入
    batch_size = 4
    z = torch.randn(batch_size, 128).to(device)

    # 测试生成器
    logits = generator(z)
    indices = generator.generate(z)

    print(f"生成器:")
    print(f"  输入: z={z.shape}")
    print(f"  输出logits: {logits.shape}")
    print(f"  输出indices: {indices.shape}")
    print(f"  参数量: {sum(p.numel() for p in generator.parameters()):,}")

    # 测试判别器
    validity = discriminator(indices)
    print(f"\n判别器:")
    print(f"  输入: indices={indices.shape}")
    print(f"  输出: {validity.shape}")
    print(f"  参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

    # 测试索引范围
    print(f"\n索引范围: [{indices.min().item()}, {indices.max().item()}]")


if __name__ == "__main__":
    test_models()

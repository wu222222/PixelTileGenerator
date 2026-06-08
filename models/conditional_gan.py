"""
Conditional GAN 模型

功能:
- 基于AutoEncoder的Conditional GAN
- 根据类别标签生成32×32像素瓦片
"""

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """条件生成器: 基于AutoEncoder的Decoder"""

    def __init__(self, latent_dim=128, num_classes=16, embed_dim=32, pretrained_decoder=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

        # 映射层: latent + embed → decoder输入
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # 使用预训练的Decoder
        if pretrained_decoder is not None:
            self.decoder = pretrained_decoder
        else:
            # 如果没有预训练，创建新的Decoder
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.autoencoder_resnet import Decoder
            self.decoder = Decoder(latent_dim)

    def forward(self, z, labels):
        """
        Args:
            z: 随机噪声 [batch_size, latent_dim]
            labels: 类别标签 [batch_size]
        Returns:
            生成的图片 [batch_size, 4, 32, 32]
        """
        # 获取类别嵌入
        class_embed = self.class_embedding(labels)  # [batch_size, embed_dim]

        # 拼接
        x = torch.cat([z, class_embed], dim=1)  # [batch_size, latent_dim + embed_dim]

        # 映射
        x = self.mapping(x)  # [batch_size, latent_dim]

        # 解码
        return self.decoder(x)

    def generate(self, label, num_samples=1, device="cuda"):
        """生成指定类别的图片"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), label, dtype=torch.long).to(device)
            return self.forward(z, labels)


class ConditionalDiscriminator(nn.Module):
    """条件判别器"""

    def __init__(self, num_classes=16, embed_dim=32):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

        # CNN提取图片特征
        self.conv = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 8×8 → 4×4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 4×4 → 2×2
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 判断层: 图片特征 + 类别嵌入
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2 + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        """
        Args:
            img: 图片 [batch_size, 4, 32, 32]
            labels: 类别标签 [batch_size]
        Returns:
            真实概率 [batch_size, 1]
        """
        # 提取图片特征
        img_feat = self.conv(img)  # [batch_size, 512, 2, 2]
        img_feat = img_feat.view(img_feat.size(0), -1)  # [batch_size, 512*2*2]

        # 获取类别嵌入
        class_embed = self.class_embedding(labels)  # [batch_size, embed_dim]

        # 拼接
        x = torch.cat([img_feat, class_embed], dim=1)  # [batch_size, 512*2*2 + embed_dim]

        # 判断
        return self.fc(x)


def test_models():
    """测试模型"""
    # 创建模型
    generator = ConditionalGenerator(latent_dim=128, num_classes=16)
    discriminator = ConditionalDiscriminator(num_classes=16)

    # 测试输入
    batch_size = 4
    z = torch.randn(batch_size, 128)
    labels = torch.randint(0, 16, (batch_size,))
    fake_images = torch.randn(batch_size, 4, 32, 32)

    # 测试生成器
    generated = generator(z, labels)
    print(f"生成器:")
    print(f"  输入: z={z.shape}, labels={labels.shape}")
    print(f"  输出: {generated.shape}")
    print(f"  参数量: {sum(p.numel() for p in generator.parameters()):,}")

    # 测试判别器
    validity = discriminator(fake_images, labels)
    print(f"\n判别器:")
    print(f"  输入: img={fake_images.shape}, labels={labels.shape}")
    print(f"  输出: {validity.shape}")
    print(f"  参数量: {sum(p.numel() for p in discriminator.parameters()):,}")


if __name__ == "__main__":
    test_models()

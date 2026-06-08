"""
Terrain WGAN-GP 训练脚本

功能:
- 无条件生成地形纹理
- 使用WGAN-GP训练
- 只用terrain数据
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 配置
CONFIG = {
    # 数据
    "data_dir": "datasets/classified/pixel_32_v2/terrain",

    # 模型
    "latent_dim": 128,

    # 训练参数
    "batch_size": 64,
    "epochs": 300,
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP参数
    "lambda_gp": 10.0,
    "n_critic": 5,

    # 保存
    "checkpoint_dir": "checkpoints/terrain_wgan",
    "save_every": 50,
    "sample_every": 25,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class TerrainDataset(Dataset):
    """地形纹理数据集"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 获取所有图片
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
        self.image_files = [
            f for f in self.data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        print(f"加载数据集: {len(self.image_files)} 张图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGBA")

        if self.transform:
            img = self.transform(img)

        return img


class Generator(nn.Module):
    """无条件生成器"""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

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
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 2, 2)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    """无条件判别器"""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
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

    def forward(self, img):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def gradient_penalty(discriminator, real_images, fake_images, device):
    """计算梯度惩罚"""
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def get_transforms():
    """获取数据变换（包含数据增强）"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[0, 270]),  # 0-270度随机旋转
        transforms.ToTensor(),
    ])


def save_samples(generator, device, save_dir, epoch, fixed_noise, num_samples=16):
    """保存生成样本"""
    generator.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        fake = generator(fixed_noise[:num_samples])

        # 保存单张图片
        for i in range(num_samples):
            img = transforms.ToPILImage()(fake[i].cpu())
            img_large = img.resize((128, 128), Image.Resampling.NEAREST)
            img_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png")

        # 保存网格图
        grid_size = int(num_samples ** 0.5)
        grid_img = Image.new("RGBA", (grid_size * 128, grid_size * 128))

        for i in range(num_samples):
            img = transforms.ToPILImage()(fake[i].cpu())
            img_large = img.resize((128, 128), Image.Resampling.NEAREST)
            row = i // grid_size
            col = i % grid_size
            grid_img.paste(img_large, (col * 128, row * 128))

        grid_img.save(samples_dir / f"epoch_{epoch:03d}_grid.png")

    generator.train()


def create_timeline(save_dir, epochs):
    """创建时间轴对比图"""
    samples_dir = Path(save_dir) / "samples"

    # 收集所有grid图
    grid_images = []
    for epoch in epochs:
        grid_path = samples_dir / f"epoch_{epoch:03d}_grid.png"
        if grid_path.exists():
            grid_images.append((epoch, Image.open(grid_path)))

    if not grid_images:
        return

    # 创建时间轴图
    img_width = grid_images[0][1].width
    img_height = grid_images[0][1].height
    padding = 10
    label_height = 30

    timeline_width = len(grid_images) * (img_width + padding) + padding
    timeline_height = img_height + label_height + padding * 2

    timeline = Image.new("RGB", (timeline_width, timeline_height), (40, 40, 40))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(timeline)

    for i, (epoch, img) in enumerate(grid_images):
        x = padding + i * (img_width + padding)
        y = label_height + padding

        timeline.paste(img, (x, y))
        draw.text((x + 10, 5), f"Epoch {epoch}", fill="white")

    timeline.save(save_dir / "timeline.png")
    print(f"时间轴已保存: {save_dir / 'timeline.png'}")


def train():
    """训练函数"""
    print("=" * 60)
    print("Terrain WGAN-GP 训练 (无条件)")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Latent维度: {CONFIG['latent_dim']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    # 创建保存目录
    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 数据变换
    transform = get_transforms()

    # 加载数据集
    dataset = TerrainDataset(project_root / CONFIG["data_dir"], transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 创建模型
    generator = Generator(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])
    discriminator = Discriminator().to(CONFIG["device"])

    # 优化器
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=CONFIG["lr_g"],
        betas=(CONFIG["beta1"], CONFIG["beta2"]),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=CONFIG["lr_d"],
        betas=(CONFIG["beta1"], CONFIG["beta2"]),
    )

    # 固定噪声（用于生成样本）
    fixed_noise = torch.randn(16, CONFIG["latent_dim"]).to(CONFIG["device"])

    # 打印模型信息
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")
    print(f"数据集大小: {len(dataset)}")
    print(f"Batch数量: {len(dataloader)}")
    print()

    # 训练历史
    history = {
        "g_loss": [],
        "d_loss": [],
        "gp_loss": [],
    }

    # 训练循环
    best_g_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_gp_loss = 0.0
        num_batches = 0

        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            for _ in range(CONFIG["n_critic"]):
                optimizer_d.zero_grad()

                # 真图片
                d_real = discriminator(real_images)

                # 假图片
                z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
                fake_images = generator(z)
                d_fake = discriminator(fake_images.detach())

                # 梯度惩罚
                gp = gradient_penalty(discriminator, real_images, fake_images.detach(), CONFIG["device"])

                # WGAN-GP判别器损失
                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            # 假图片
            z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
            fake_images = generator(z)
            d_fake = discriminator(fake_images)

            # 生成器损失
            g_loss = -d_fake.mean()

            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_gp_loss += gp.item()
            num_batches += 1

        # 计算平均损失
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_gp_loss = epoch_gp_loss / num_batches
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["gp_loss"].append(avg_gp_loss)

        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"G_loss: {avg_g_loss:.4f} "
              f"D_loss: {avg_d_loss:.4f} "
              f"GP: {avg_gp_loss:.4f} "
              f"Time: {elapsed_time:.1f}s")

        # 保存样本
        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(generator, CONFIG["device"], checkpoint_dir, epoch + 1, fixed_noise)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"terrain_wgan_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": avg_g_loss,
                "d_loss": avg_d_loss,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_path = checkpoint_dir / "terrain_wgan_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": best_g_loss,
                "d_loss": avg_d_loss,
                "config": CONFIG,
            }, best_path)

    # 保存最终模型
    final_path = checkpoint_dir / "terrain_wgan_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "g_loss": avg_g_loss,
        "d_loss": avg_d_loss,
        "config": CONFIG,
    }, final_path)

    # 保存训练历史
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 创建时间轴
    timeline_epochs = list(range(CONFIG["sample_every"], CONFIG["epochs"] + 1, CONFIG["sample_every"]))
    create_timeline(checkpoint_dir, timeline_epochs)

    # 打印总结
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"最佳G_loss: {best_g_loss:.4f}")
    print(f"最终G_loss: {avg_g_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

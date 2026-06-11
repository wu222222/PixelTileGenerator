"""
Latent Space GAN 训练脚本

功能:
- 在AutoEncoder的latent space训练WGAN-GP
- 生成新的latent z
- 用Decoder生成图片
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.autoencoder_resnet import ResNetAutoEncoder


# 配置
CONFIG = {
    # 数据
    "latent_data": "datasets/latent_data",
    "autoencoder_checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",

    # 模型
    "latent_dim": 128,

    # 训练参数
    "batch_size": 64,
    "epochs": 500,
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP参数
    "lambda_gp": 10.0,
    "n_critic": 5,

    # 保存
    "checkpoint_dir": "checkpoints/latent_gan",
    "save_every": 50,
    "sample_every": 25,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class LatentDataset(Dataset):
    """Latent vector数据集"""

    def __init__(self, latent_data_path):
        self.latents = np.load(latent_data_path / "latents.npy")
        print(f"加载数据集: {len(self.latents)} 个latent vectors")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.latents[idx])


class LatentGenerator(nn.Module):
    """Latent空间生成器"""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z):
        return self.fc(z)


class LatentDiscriminator(nn.Module):
    """Latent空间判别器"""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.fc(z)


def gradient_penalty(discriminator, real_z, fake_z, device):
    """计算梯度惩罚"""
    batch_size = real_z.size(0)
    alpha = torch.rand(batch_size, 1).to(device)

    interpolated = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)
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


def save_samples(generator, autoencoder, device, save_dir, epoch, fixed_noise):
    """保存生成样本"""
    from torchvision import transforms

    generator.eval()
    autoencoder.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # 生成latent vectors
        fake_z = generator(fixed_noise)

        # 用Decoder生成图片
        fake_images = autoencoder.decode(fake_z)

        # 保存图片
        for i in range(min(16, len(fixed_noise))):
            img_tensor = fake_images[i].cpu()
            img = transforms.ToPILImage()(img_tensor)
            img_large = img.resize((128, 128), Image.Resampling.NEAREST)
            img_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png")

    generator.train()


def train():
    """训练函数"""
    print("=" * 60)
    print("Latent Space GAN 训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Latent维度: {CONFIG['latent_dim']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    # 创建保存目录
    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 加载latent数据
    latent_data_path = project_root / CONFIG["latent_data"]
    dataset = LatentDataset(latent_data_path)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 加载AutoEncoder（只用Decoder）
    ae_checkpoint = torch.load(
        project_root / CONFIG["autoencoder_checkpoint"],
        map_location=CONFIG["device"]
    )
    ae_config = ae_checkpoint.get("config", {})
    autoencoder = ResNetAutoEncoder(latent_dim=ae_config.get("latent_dim", 128)).to(CONFIG["device"])
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()

    print(f"\nAutoEncoder加载成功")

    # 创建GAN模型
    generator = LatentGenerator(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])
    discriminator = LatentDiscriminator(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])

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

    # 固定噪声
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

        for batch_idx, real_z in enumerate(dataloader):
            batch_size = real_z.size(0)
            real_z = real_z.to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            for _ in range(CONFIG["n_critic"]):
                optimizer_d.zero_grad()

                # 真实latent
                d_real = discriminator(real_z)

                # 假latent
                z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
                fake_z = generator(z)
                d_fake = discriminator(fake_z.detach())

                # 梯度惩罚
                gp = gradient_penalty(discriminator, real_z, fake_z.detach(), CONFIG["device"])

                # WGAN-GP判别器损失
                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            # 假latent
            z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
            fake_z = generator(z)
            d_fake = discriminator(fake_z)

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
            save_samples(generator, autoencoder, CONFIG["device"], checkpoint_dir, epoch + 1, fixed_noise)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"latent_gan_epoch_{epoch+1:03d}.pth"
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
            best_path = checkpoint_dir / "latent_gan_best.pth"
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
    final_path = checkpoint_dir / "latent_gan_final.pth"
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

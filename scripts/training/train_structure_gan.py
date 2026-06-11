"""
Structure GAN 训练脚本

功能:
- 训练Structure Generator生成索引图
- 使用WGAN-GP训练
- 支持调色板映射
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

from models.structure_generator import StructureGenerator, StructureDiscriminator


# 配置
CONFIG = {
    # 数据
    "data_dir": "datasets/structure_data",

    # 模型
    "latent_dim": 128,
    "num_classes": 32,

    # 训练参数
    "batch_size": 32,
    "epochs": 300,
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP参数
    "lambda_gp": 10.0,
    "n_critic": 5,

    # 保存
    "checkpoint_dir": "checkpoints/structure_gan",
    "save_every": 50,
    "sample_every": 25,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class StructureDataset(Dataset):
    """Structure Map数据集"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # 加载数据集信息
        info_path = self.data_dir / "dataset_info.json"
        with open(info_path, "r") as f:
            self.dataset_info = json.load(f)

        self.samples = self.dataset_info["samples"]
        print(f"加载数据集: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载Structure Map
        structure_path = self.data_dir / sample["structure_path"]
        indices = np.load(structure_path)

        # 转换为tensor
        indices_tensor = torch.LongTensor(indices)

        return indices_tensor, sample["name"]


def gradient_penalty(discriminator, real_indices, fake_indices, device):
    """计算梯度惩罚（在one-hot空间中）"""
    batch_size = real_indices.size(0)

    # 转换为one-hot编码
    real_one_hot = torch.zeros(batch_size, CONFIG["num_classes"], 32, 32).to(device)
    real_one_hot.scatter_(1, real_indices.unsqueeze(1), 1)

    fake_one_hot = torch.zeros(batch_size, CONFIG["num_classes"], 32, 32).to(device)
    fake_one_hot.scatter_(1, fake_indices.unsqueeze(1), 1)

    # 随机插值系数
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    # 在one-hot空间中插值
    interpolated = (alpha * real_one_hot + (1 - alpha) * fake_one_hot).requires_grad_(True)

    # 判别器输出（直接使用one-hot输入）
    d_interpolated = discriminator_with_onehot(discriminator, interpolated)

    # 计算梯度
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


def discriminator_with_onehot(discriminator, one_hot):
    """使用one-hot编码的判别器"""
    # 直接使用one-hot作为输入（不转换为索引）
    x = discriminator.conv(one_hot)
    x = x.view(x.size(0), -1)
    return discriminator.fc(x)


def save_samples(generator, device, save_dir, epoch, fixed_noise, palette=None):
    """保存生成样本"""
    generator.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        indices = generator.generate(fixed_noise)

        # 保存索引图
        for i in range(min(16, len(fixed_noise))):
            idx = indices[i].cpu().numpy()

            # 保存索引图可视化（灰度图）
            idx_img = Image.fromarray((idx * 8).astype(np.uint8), mode='L')
            idx_img_large = idx_img.resize((128, 128), Image.Resampling.NEAREST)
            idx_img_large.save(samples_dir / f"epoch_{epoch:03d}_idx_{i:02d}.png")

            # 如果有调色板，保存彩色图
            if palette is not None:
                color_img = Image.new("RGB", (32, 32))
                for y in range(32):
                    for x in range(32):
                        color_idx = idx[y, x]
                        if color_idx < len(palette):
                            color_img.putpixel((x, y), tuple(palette[color_idx]))
                color_img_large = color_img.resize((128, 128), Image.Resampling.NEAREST)
                color_img_large.save(samples_dir / f"epoch_{epoch:03d}_color_{i:02d}.png")

    generator.train()


def train():
    """训练函数"""
    print("=" * 60)
    print("Structure GAN 训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Latent维度: {CONFIG['latent_dim']}")
    print(f"类别数: {CONFIG['num_classes']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    # 创建保存目录
    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = StructureDataset(project_root / CONFIG["data_dir"])

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 创建模型
    generator = StructureGenerator(
        latent_dim=CONFIG["latent_dim"],
        num_classes=CONFIG["num_classes"],
    ).to(CONFIG["device"])

    discriminator = StructureDiscriminator(
        num_classes=CONFIG["num_classes"],
    ).to(CONFIG["device"])

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

        for batch_idx, (real_indices, names) in enumerate(dataloader):
            batch_size = real_indices.size(0)
            real_indices = real_indices.to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            for _ in range(CONFIG["n_critic"]):
                optimizer_d.zero_grad()

                # 真实索引图
                d_real = discriminator(real_indices)

                # 假索引图
                z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
                fake_logits = generator(z)
                fake_indices = torch.argmax(fake_logits, dim=1)
                d_fake = discriminator(fake_indices.detach())

                # 梯度惩罚
                gp = gradient_penalty(discriminator, real_indices, fake_indices.detach(), CONFIG["device"])

                # WGAN-GP判别器损失
                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            # 假索引图
            z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
            fake_logits = generator(z)
            fake_indices = torch.argmax(fake_logits, dim=1)
            d_fake = discriminator(fake_indices)

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
            checkpoint_path = checkpoint_dir / f"structure_gan_epoch_{epoch+1:03d}.pth"
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
            best_path = checkpoint_dir / "structure_gan_best.pth"
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
    final_path = checkpoint_dir / "structure_gan_final.pth"
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

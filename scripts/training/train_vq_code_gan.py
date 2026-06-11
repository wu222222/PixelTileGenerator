"""
VQ Code GAN 训练脚本

功能:
- 学习离散code indices的分布
- 生成新的indices
- 用VQ-VAE Decoder生成新纹理
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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_code_gan import VQCodeGAN
from models.vq_vae_v2 import VQVAEv2
from torchvision import transforms
from PIL import Image


# 配置
CONFIG = {
    # 数据
    "latent_data": "datasets/vqvae_latent_data",
    "vqvae_checkpoint": "checkpoints/vqvae_v5/vqvae_v5_best.pth",

    # 模型
    "num_embeddings": 256,
    "latent_size": 16,

    # 训练参数
    "batch_size": 32,
    "epochs": 500,
    "lr_g": 2e-4,  # 生成器学习率更大
    "lr_d": 1e-4,  # 判别器学习率更小
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP参数
    "lambda_gp": 10.0,
    "n_critic": 3,  # 减少判别器训练次数

    # 保存
    "checkpoint_dir": "checkpoints/vq_code_gan",
    "save_every": 50,
    "sample_every": 25,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class IndicesDataset(Dataset):
    """Indices数据集"""

    def __init__(self, latent_data_path):
        self.indices = np.load(latent_data_path / "indices.npy")
        print(f"加载数据集: {len(self.indices)} 个indices")
        print(f"Indices形状: {self.indices.shape}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.LongTensor(self.indices[idx])


def gradient_penalty(discriminator, real_indices, fake_indices, device):
    """计算梯度惩罚（在one-hot空间中）"""
    batch_size = real_indices.size(0)

    # 转换为one-hot
    real_one_hot = torch.zeros(batch_size, CONFIG["num_embeddings"], 16, 16).to(device)
    real_one_hot.scatter_(1, real_indices.unsqueeze(1), 1)

    fake_one_hot = torch.zeros(batch_size, CONFIG["num_embeddings"], 16, 16).to(device)
    fake_one_hot.scatter_(1, fake_indices.unsqueeze(1), 1)

    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real_one_hot + (1 - alpha) * fake_one_hot).requires_grad_(True)

    # 判别器输出（直接使用one-hot，不转换为indices）
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


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    """对图片进行颜色量化"""
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def save_samples(code_gan, vqvae, device, save_dir, epoch, fixed_noise):
    """保存生成样本"""
    code_gan.eval()
    vqvae.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # 生成indices
        fake_indices = code_gan.generate(fixed_noise)

        # 用codebook lookup得到z_q
        z_q = vqvae.vq.embedding(fake_indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [batch, embedding_dim, h, w]

        # 用Decoder生成图片
        fake_images = vqvae.decode(z_q)

        # 保存图片
        for i in range(min(16, len(fixed_noise))):
            img_tensor = fake_images[i].cpu()

            # 量化到32色
            img_pil = transforms.ToPILImage()(img_tensor)
            img_quantized = quantize_image(img_pil, colors=32)
            img_large = img_quantized.resize((128, 128), Image.Resampling.NEAREST)
            img_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png")

    code_gan.train()


def train():
    """训练函数"""
    print("=" * 60)
    print("VQ Code GAN 训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Indices形状: [{CONFIG['latent_size']}, {CONFIG['latent_size']}]")
    print(f"码本大小: {CONFIG['num_embeddings']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    # 创建保存目录
    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 加载indices数据
    latent_data_path = project_root / CONFIG["latent_data"]
    dataset = IndicesDataset(latent_data_path)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 加载VQ-VAE
    vqvae_checkpoint = torch.load(
        project_root / CONFIG["vqvae_checkpoint"],
        map_location=CONFIG["device"]
    )
    vqvae_config = vqvae_checkpoint.get("config", {})
    vqvae = VQVAEv2(
        in_channels=vqvae_config.get("in_channels", 4),
        hidden_channels=vqvae_config.get("hidden_channels", 256),
        embedding_dim=vqvae_config.get("embedding_dim", 64),
        num_embeddings=vqvae_config.get("num_embeddings", 256),
    ).to(CONFIG["device"])
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae.eval()

    print(f"\nVQ-VAE加载成功")

    # 创建Code GAN
    code_gan = VQCodeGAN(
        num_embeddings=CONFIG["num_embeddings"],
        latent_size=CONFIG["latent_size"],
    ).to(CONFIG["device"])

    # 优化器
    optimizer_g = optim.Adam(
        code_gan.generator.parameters(),
        lr=CONFIG["lr_g"],
        betas=(CONFIG["beta1"], CONFIG["beta2"]),
    )

    optimizer_d = optim.Adam(
        code_gan.discriminator.parameters(),
        lr=CONFIG["lr_d"],
        betas=(CONFIG["beta1"], CONFIG["beta2"]),
    )

    # 固定噪声
    fixed_noise = torch.randn(16, CONFIG["num_embeddings"]).to(CONFIG["device"])

    # 打印模型信息
    g_params = sum(p.numel() for p in code_gan.generator.parameters())
    d_params = sum(p.numel() for p in code_gan.discriminator.parameters())
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
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0

        for batch_idx, real_indices in enumerate(dataloader):
            batch_size = real_indices.size(0)
            real_indices = real_indices.to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            for _ in range(CONFIG["n_critic"]):
                optimizer_d.zero_grad()

                # 真实indices
                d_real = code_gan.discriminate(real_indices)
                d_real_mean = d_real.mean().item()

                # 假indices
                noise = torch.randn(batch_size, CONFIG["num_embeddings"]).to(CONFIG["device"])
                fake_indices = code_gan.generate(noise)
                d_fake = code_gan.discriminate(fake_indices.detach())
                d_fake_mean = d_fake.mean().item()

                # 梯度惩罚
                gp = gradient_penalty(code_gan.discriminator, real_indices, fake_indices.detach(), CONFIG["device"])

                # WGAN-GP判别器损失
                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            # 假indices
            noise = torch.randn(batch_size, CONFIG["num_embeddings"]).to(CONFIG["device"])
            fake_indices = code_gan.generate(noise)
            d_fake = code_gan.discriminate(fake_indices)

            # 生成器损失
            g_loss = -d_fake.mean()

            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_gp_loss += gp.item()
            epoch_d_real += d_real_mean
            epoch_d_fake += d_fake_mean
            num_batches += 1

        # 计算平均损失
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_gp_loss = epoch_gp_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["gp_loss"].append(avg_gp_loss)

        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"G_loss: {avg_g_loss:.4f} "
              f"D_loss: {avg_d_loss:.4f} "
              f"GP: {avg_gp_loss:.4f} "
              f"D_real: {avg_d_real:.4f} "
              f"D_fake: {avg_d_fake:.4f} "
              f"Time: {elapsed_time:.1f}s")

        # 保存样本
        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(code_gan, vqvae, CONFIG["device"], checkpoint_dir, epoch + 1, fixed_noise)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"vq_code_gan_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": code_gan.generator.state_dict(),
                "discriminator_state_dict": code_gan.discriminator.state_dict(),
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
            best_path = checkpoint_dir / "vq_code_gan_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": code_gan.generator.state_dict(),
                "discriminator_state_dict": code_gan.discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": best_g_loss,
                "d_loss": avg_d_loss,
                "config": CONFIG,
            }, best_path)

    # 保存最终模型
    final_path = checkpoint_dir / "vq_code_gan_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "generator_state_dict": code_gan.generator.state_dict(),
        "discriminator_state_dict": code_gan.discriminator.state_dict(),
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

"""
ZQ Latent GAN 训练脚本 (v3 — 无 Tanh + WGAN-GP + 死通道过滤)

改进 (相对 v2):
- 去掉 Tanh: G 输出无界值，D 学习实际范围
- WGAN-GP: 替代 hinge loss
- 死通道过滤: 移除 15 个 std < 0.05 的通道
- 推理时: padding 回 128 通道再做 codebook lookup
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.zq_latent_gan import ZQLatentGAN
from models.vq_vae_v4 import VQVAEv4
from torchvision import transforms
from PIL import Image


CONFIG = {
    # 数据
    "zq_data": "datasets/vqvae_v9_zq_data",
    "vqvae_checkpoint": "checkpoints/vqvae_v9/vqvae_v9_best.pth",

    # 模型
    "noise_dim": 128,
    "embed_dim": 128,  # 保留 128，死通道在数据预处理中处理
    "latent_size": 16,

    # 训练参数
    "batch_size": 32,
    "epochs": 500,
    "lr_g": 1e-4,
    "lr_d": 4e-5,
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP 参数
    "lambda_gp": 10.0,
    "n_critic": 2,

    # 保存
    "checkpoint_dir": "checkpoints/zq_latent_gan_v9",
    "save_every": 50,
    "sample_every": 25,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# 死通道索引 (std < 0.05)
DEAD_CHANNELS = [11, 14, 30, 31, 62, 66, 74, 84, 91, 98, 103, 109, 119, 126, 127]
GOOD_CHANNELS = [i for i in range(128) if i not in DEAD_CHANNELS]  # 113 个


class ZQDataset(Dataset):
    """z_q 数据集 (死通道置零, 有效通道归一化)"""

    def __init__(self, data_path):
        zq_raw = np.load(data_path / "z_q.npy")  # [N, 128, 16, 16]

        # 有效通道归一化
        zq_good = zq_raw[:, GOOD_CHANNELS, :, :]
        self.mean = zq_good.mean(axis=(0, 2, 3))
        self.std = zq_good.std(axis=(0, 2, 3))

        zq_norm = np.zeros_like(zq_raw, dtype=np.float32)
        zq_norm[:, GOOD_CHANNELS, :, :] = (zq_good - self.mean.reshape(1, -1, 1, 1)) / (self.std.reshape(1, -1, 1, 1) + 1e-8)

        self.zq = zq_norm
        print(f"加载数据集: {len(self.zq)} 个 z_q")
        print(f"有效通道: {len(GOOD_CHANNELS)}, 死通道: {len(DEAD_CHANNELS)}")
        print(f"归一化后有效通道 std 范围: [{self.std.min():.4f}, {self.std.max():.4f}]")
        print(f"整体范围: [{self.zq.min():.4f}, {self.zq.max():.4f}]")

    def __len__(self):
        return len(self.zq)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.zq[idx])


def gradient_penalty(discriminator, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True,
    )[0]
    return ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()


def quantize_image(img, colors=32):
    if img.mode == "RGBA":
        q = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return q.convert("RGBA")


def save_samples(gan, vqvae, codebook, device, save_dir, epoch, fixed_noise, dataset):
    """保存生成样本

    推理: G(noise) → z_q (128ch, 死通道=0) → nearest codebook → decoder
    """
    gan.eval()
    vqvae.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.FloatTensor(dataset.mean).to(device)
    std = torch.FloatTensor(dataset.std).to(device)

    with torch.no_grad():
        z_q_norm = gan.generate(fixed_noise)

        # 反归一化有效通道, 死通道保持为 0
        z_q_full = z_q_norm.clone()
        z_q_good = z_q_norm[:, GOOD_CHANNELS, :, :]
        z_q_good = z_q_good * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        z_q_full[:, GOOD_CHANNELS, :, :] = z_q_good

        # nearest codebook projection
        B, C, H, W = z_q_full.shape
        z_flat = z_q_full.permute(0, 2, 3, 1).reshape(-1, C)
        cb = codebook.weight.data  # [1024, 128]
        dist = torch.cdist(z_flat, cb)
        nearest = dist.argmin(dim=1)
        z_proj = cb[nearest].reshape(B, H, W, C).permute(0, 3, 1, 2)

        # decoder
        fake_images = vqvae.decode(z_proj)

        for i in range(min(8, len(fixed_noise))):
            img_tensor = fake_images[i].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            img_q = quantize_image(img_pil, colors=32)
            img_large = img_q.resize((256, 256), Image.Resampling.NEAREST)
            img_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png")

    gan.train()


def train():
    print("=" * 60)
    print("ZQ Latent GAN v3 (无Tanh + WGAN-GP + 死通道过滤)")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Batch: {CONFIG['batch_size']}  Epochs: {CONFIG['epochs']}")
    print(f"LR: G={CONFIG['lr_g']}, D={CONFIG['lr_d']}")
    print(f"死通道: {DEAD_CHANNELS}")
    print("=" * 60)

    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    zq_data_path = project_root / CONFIG["zq_data"]
    dataset = ZQDataset(zq_data_path)

    # 保存归一化参数
    norm_path = checkpoint_dir / "norm_params.json"
    with open(norm_path, "w") as f:
        json.dump({
            "mean": dataset.mean.tolist(),
            "std": dataset.std.tolist(),
            "dead_channels": DEAD_CHANNELS,
            "good_channels": GOOD_CHANNELS,
        }, f)

    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
    )

    # 加载 VQ-VAE
    vqvae_checkpoint = torch.load(
        project_root / CONFIG["vqvae_checkpoint"],
        map_location=CONFIG["device"], weights_only=False,
    )
    vqvae_config = vqvae_checkpoint.get("config", {})
    vqvae = VQVAEv4(
        in_channels=vqvae_config.get("in_channels", 4),
        hidden_channels=vqvae_config.get("hidden_channels", 256),
        embedding_dim=vqvae_config.get("embedding_dim", 128),
        num_embeddings=vqvae_config.get("num_embeddings", 1024),
        latent_size=vqvae_config.get("latent_size", 16),
    ).to(CONFIG["device"])
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae.eval()

    codebook = vqvae.vq.embedding
    print(f"\nVQ-VAE 加载成功 (epoch {vqvae_checkpoint.get('epoch', '?')})")

    # 创建 GAN
    gan = ZQLatentGAN(
        noise_dim=CONFIG["noise_dim"],
        embed_dim=CONFIG["embed_dim"],
        latent_size=CONFIG["latent_size"],
    ).to(CONFIG["device"])

    optimizer_g = optim.Adam(gan.generator.parameters(), lr=CONFIG["lr_g"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=CONFIG["lr_d"], betas=(CONFIG["beta1"], CONFIG["beta2"]))

    fixed_noise = torch.randn(8, CONFIG["noise_dim"]).to(CONFIG["device"])

    g_params = sum(p.numel() for p in gan.generator.parameters())
    d_params = sum(p.numel() for p in gan.discriminator.parameters())
    print(f"\nG 参数量: {g_params:,}")
    print(f"D 参数量: {d_params:,}")
    print(f"数据集: {len(dataset)}  Batches: {len(dataloader)}")
    print()

    history = {"g_loss": [], "d_loss": [], "gp_loss": []}
    best_g_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_gp = 0.0
        num_batches = 0

        for real_zq in dataloader:
            batch_size = real_zq.size(0)
            real_zq = real_zq.to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            for _ in range(CONFIG["n_critic"]):
                optimizer_d.zero_grad()

                d_real = gan.discriminate(real_zq)
                noise = torch.randn(batch_size, CONFIG["noise_dim"]).to(CONFIG["device"])
                fake_zq = gan.generate(noise)
                d_fake = gan.discriminate(fake_zq.detach())

                gp = gradient_penalty(gan.discriminator, real_zq, fake_zq.detach(), CONFIG["device"])
                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, CONFIG["noise_dim"]).to(CONFIG["device"])
            fake_zq = gan.generate(noise)
            d_fake = gan.discriminate(fake_zq)
            g_loss = -d_fake.mean()
            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_gp += gp.item()
            num_batches += 1

        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        avg_gp = epoch_gp / num_batches
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["gp_loss"].append(avg_gp)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"G: {avg_g:.4f} D: {avg_d:.4f} GP: {avg_gp:.4f} "
              f"T: {elapsed:.0f}s")

        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(gan, vqvae, codebook, CONFIG["device"],
                        checkpoint_dir, epoch + 1, fixed_noise, dataset)

        if (epoch + 1) % CONFIG["save_every"] == 0:
            path = checkpoint_dir / f"zq_gan_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": gan.generator.state_dict(),
                "discriminator_state_dict": gan.discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": avg_g, "d_loss": avg_d,
                "config": CONFIG,
            }, path)
            print(f"  Saved: {path}")

        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": gan.generator.state_dict(),
                "discriminator_state_dict": gan.discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": best_g_loss, "d_loss": avg_d,
                "config": CONFIG,
            }, checkpoint_dir / "zq_gan_best.pth")

    torch.save({
        "epoch": CONFIG["epochs"],
        "generator_state_dict": gan.generator.state_dict(),
        "discriminator_state_dict": gan.discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "g_loss": avg_g, "d_loss": avg_d,
        "config": CONFIG,
    }, checkpoint_dir / "zq_gan_final.pth")

    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"训练完成! 时间: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"最佳 G loss: {best_g_loss:.4f}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

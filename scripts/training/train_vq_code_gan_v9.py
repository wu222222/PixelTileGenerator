"""
VQ Code GAN 训练脚本 (v9 适配)

功能:
- 学习离散 code indices 的分布
- 生成新的 indices
- 用 VQ-VAEv4 Decoder 生成 64x64 纹理
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

from models.vq_code_gan import VQCodeGAN
from models.vq_vae_v4 import VQVAEv4
from torchvision import transforms
from PIL import Image


CONFIG = {
    # 数据
    "latent_data": "datasets/vqvae_v9_latent_data",
    "vqvae_checkpoint": "checkpoints/vqvae_v9/vqvae_v9_best.pth",

    # 模型
    "num_embeddings": 1024,
    "latent_size": 16,

    # 训练参数
    "batch_size": 32,
    "epochs": 500,
    "lr_g": 5e-4,
    "lr_d": 5e-5,
    "beta1": 0.0,
    "beta2": 0.9,

    # WGAN-GP 参数
    "lambda_gp": 10.0,
    "n_critic": 1,

    # 保存
    "checkpoint_dir": "checkpoints/vq_code_gan_v9",
    "save_every": 50,
    "sample_every": 25,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class IndicesDataset(Dataset):
    """Indices 数据集"""

    def __init__(self, latent_data_path):
        self.indices = np.load(latent_data_path / "indices.npy")
        print(f"加载数据集: {len(self.indices)} 个 indices")
        print(f"Indices形状: {self.indices.shape}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.LongTensor(self.indices[idx])


def gradient_penalty(discriminator, real_indices, fake_indices, device):
    """计算梯度惩罚（在 one-hot 空间中）"""
    batch_size = real_indices.size(0)
    num_embeddings = CONFIG["num_embeddings"]
    latent_size = CONFIG["latent_size"]

    real_one_hot = torch.zeros(batch_size, num_embeddings, latent_size, latent_size).to(device)
    real_one_hot.scatter_(1, real_indices.unsqueeze(1), 1)

    fake_one_hot = torch.zeros(batch_size, num_embeddings, latent_size, latent_size).to(device)
    fake_one_hot.scatter_(1, fake_indices.unsqueeze(1), 1)

    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real_one_hot + (1 - alpha) * fake_one_hot).requires_grad_(True)

    # 给插值样本加噪声（与 D forward 中的 input_noise 一致）
    if discriminator.input_noise > 0:
        noise = torch.randn_like(interpolated) * discriminator.input_noise
        interpolated = interpolated + noise

    d_interpolated = discriminator_with_onehot(discriminator, interpolated)

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
    """使用 one-hot 编码的判别器 (直接前向，不走 indices 转换)"""
    x = discriminator.conv(one_hot)
    x = x.view(x.size(0), -1)
    return discriminator.fc(x)


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
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
        fake_indices = code_gan.generate(fixed_noise)

        z_q = vqvae.vq.embedding(fake_indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        fake_images = vqvae.decode(z_q)

        for i in range(min(16, len(fixed_noise))):
            img_tensor = fake_images[i].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            img_quantized = quantize_image(img_pil, colors=32)
            # 64x64 -> 256x256 放大预览
            img_large = img_quantized.resize((256, 256), Image.Resampling.NEAREST)
            img_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i:02d}.png")

    code_gan.train()


def train():
    print("=" * 60)
    print("VQ Code GAN 训练 (v9 适配)")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Indices形状: [{CONFIG['latent_size']}, {CONFIG['latent_size']}]")
    print(f"码本大小: {CONFIG['num_embeddings']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 加载 indices 数据
    latent_data_path = project_root / CONFIG["latent_data"]
    dataset = IndicesDataset(latent_data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 加载 VQ-VAE (只用 Decoder)
    vqvae_checkpoint = torch.load(
        project_root / CONFIG["vqvae_checkpoint"],
        map_location=CONFIG["device"],
        weights_only=False,
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

    print(f"\nVQ-VAE 加载成功 (epoch {vqvae_checkpoint.get('epoch', '?')})")

    # 创建 Code GAN
    code_gan = VQCodeGAN(
        num_embeddings=CONFIG["num_embeddings"],
        latent_size=CONFIG["latent_size"],
    ).to(CONFIG["device"])

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

    fixed_noise = torch.randn(16, CONFIG["num_embeddings"]).to(CONFIG["device"])

    g_params = sum(p.numel() for p in code_gan.generator.parameters())
    d_params = sum(p.numel() for p in code_gan.discriminator.parameters())
    print(f"\n生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")
    print(f"数据集大小: {len(dataset)}")
    print(f"Batch数量: {len(dataloader)}")
    print()

    history = {
        "g_loss": [],
        "d_loss": [],
        "gp_loss": [],
    }

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

                d_real = code_gan.discriminate(real_indices)
                d_real_mean = d_real.mean().item()

                noise = torch.randn(batch_size, CONFIG["num_embeddings"]).to(CONFIG["device"])
                fake_indices = code_gan.generate(noise)
                d_fake = code_gan.discriminate(fake_indices.detach())
                d_fake_mean = d_fake.mean().item()

                gp = gradient_penalty(code_gan.discriminator, real_indices, fake_indices.detach(), CONFIG["device"])

                d_loss = d_fake.mean() - d_real.mean() + CONFIG["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size, CONFIG["num_embeddings"]).to(CONFIG["device"])
            fake_indices = code_gan.generate(noise)
            d_fake = code_gan.discriminate(fake_indices)

            g_loss = -d_fake.mean()

            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_gp_loss += gp.item()
            epoch_d_real += d_real_mean
            epoch_d_fake += d_fake_mean
            num_batches += 1

        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_gp_loss = epoch_gp_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["gp_loss"].append(avg_gp_loss)

        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"G: {avg_g_loss:.4f} "
              f"D: {avg_d_loss:.4f} "
              f"GP: {avg_gp_loss:.4f} "
              f"D_real: {avg_d_real:.4f} "
              f"D_fake: {avg_d_fake:.4f} "
              f"T: {elapsed_time:.0f}s")

        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(code_gan, vqvae, CONFIG["device"], checkpoint_dir, epoch + 1, fixed_noise)

        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"vq_code_gan_v9_epoch_{epoch+1:03d}.pth"
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
            print(f"  Saved: {checkpoint_path}")

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_path = checkpoint_dir / "vq_code_gan_v9_best.pth"
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

    final_path = checkpoint_dir / "vq_code_gan_v9_final.pth"
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

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总时间: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"最佳G_loss: {best_g_loss:.4f}")
    print(f"最终G_loss: {avg_g_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

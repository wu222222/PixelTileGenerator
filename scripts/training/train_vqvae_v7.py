"""
VQ-VAE v7 训练脚本

实验目的: 纯数据规模实验
- 架构完全复制 v5 (16×16 latent, 256码本)
- 唯一变量: 数据量 314 → 1778
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


CONFIG = {
    "data_dir": "datasets/classified/pixel_32_quantized",
    "in_channels": 4,
    "hidden_channels": 256,
    "embedding_dim": 64,
    "num_embeddings": 256,
    "commitment_cost": 0.25,
    "latent_size": 16,
    "batch_size": 32,
    "epochs": 500,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "mse_weight": 0.5,
    "l1_weight": 0.5,
    "checkpoint_dir": "checkpoints/vqvae_v7",
    "save_every": 50,
    "sample_every": 25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class TileDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
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
        return img, img_path.name


def get_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.Lambda(lambda x: x),
            transforms.Lambda(lambda x: x.rotate(90)),
            transforms.Lambda(lambda x: x.rotate(180)),
            transforms.Lambda(lambda x: x.rotate(270)),
        ]),
        transforms.ToTensor(),
    ])


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def save_samples(model, dataset, device, save_dir, epoch, num_samples=8):
    model.eval()
    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    indices = torch.randperm(len(dataset))[:num_samples]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, name = dataset[idx]
            img = img.unsqueeze(0).to(device)
            recon, _, _ = model(img)

            original_pil = transforms.ToPILImage()(img.squeeze(0).cpu())
            recon_pil = transforms.ToPILImage()(recon.squeeze(0).cpu())
            recon_quantized = quantize_image(recon_pil, colors=32)

            original_large = original_pil.resize((128, 128), Image.Resampling.NEAREST)
            recon_large = recon_quantized.resize((128, 128), Image.Resampling.NEAREST)

            original_large.save(samples_dir / f"epoch_{epoch:03d}_original_{i}.png")
            recon_large.save(samples_dir / f"epoch_{epoch:03d}_recon_{i}.png")

            comparison = Image.new("RGBA", (128 * 2 + 10, 128))
            comparison.paste(original_large, (0, 0))
            comparison.paste(recon_large, (128 + 10, 0))
            comparison.save(samples_dir / f"epoch_{epoch:03d}_comparison_{i}.png")

    model.train()


def train():
    print("=" * 60)
    print("VQ-VAE v7 训练 (数据规模实验)")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"架构: 完全复制 v5 (16×16 latent, 256码本)")
    print(f"唯一变量: 数据量 (314 → 1778)")
    print(f"损失函数: MSE({CONFIG['mse_weight']}) + L1({CONFIG['l1_weight']})")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms()
    dataset = TileDataset(project_root / CONFIG["data_dir"], transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
    )

    model = VQVAEv2(
        in_channels=CONFIG["in_channels"],
        hidden_channels=CONFIG["hidden_channels"],
        embedding_dim=CONFIG["embedding_dim"],
        num_embeddings=CONFIG["num_embeddings"],
        commitment_cost=CONFIG["commitment_cost"],
        latent_size=CONFIG["latent_size"],
    ).to(CONFIG["device"])

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"数据集大小: {len(dataset)}")
    print(f"Batch数量: {len(dataloader)}")
    print()

    history = {
        "total_loss": [], "recon_loss": [], "mse_loss": [],
        "l1_loss": [], "vq_loss": [], "codebook_usage": [],
    }

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_usage = 0.0
        num_batches = 0

        for batch_idx, (images, names) in enumerate(dataloader):
            images = images.to(CONFIG["device"])
            recon, vq_loss, indices = model(images)

            mse = mse_loss(recon, images)
            l1 = l1_loss(recon, images)
            recon_loss = CONFIG["mse_weight"] * mse + CONFIG["l1_weight"] * l1
            total_loss = recon_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            usage, _ = model.vq.get_codebook_usage(indices)

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_mse_loss += mse.item()
            epoch_l1_loss += l1.item()
            epoch_vq_loss += vq_loss.item()
            epoch_usage += usage
            num_batches += 1

        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_mse_loss = epoch_mse_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        avg_usage = epoch_usage / num_batches
        history["total_loss"].append(avg_total_loss)
        history["recon_loss"].append(avg_recon_loss)
        history["mse_loss"].append(avg_mse_loss)
        history["l1_loss"].append(avg_l1_loss)
        history["vq_loss"].append(avg_vq_loss)
        history["codebook_usage"].append(avg_usage)

        scheduler.step()

        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Total: {avg_total_loss:.4f} "
              f"Recon: {avg_recon_loss:.4f} "
              f"MSE: {avg_mse_loss:.4f} "
              f"L1: {avg_l1_loss:.4f} "
              f"VQ: {avg_vq_loss:.4f} "
              f"Usage: {avg_usage:.2%} "
              f"Time: {elapsed_time:.1f}s")

        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(model, dataset, CONFIG["device"], checkpoint_dir, epoch + 1)

        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"vqvae_v7_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total_loss,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_path = checkpoint_dir / "vqvae_v7_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": CONFIG,
            }, best_path)

    final_path = checkpoint_dir / "vqvae_v7_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_total_loss,
        "config": CONFIG,
    }, final_path)

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最终损失: {avg_total_loss:.4f}")
    print(f"最终码本使用率: {avg_usage:.2%}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

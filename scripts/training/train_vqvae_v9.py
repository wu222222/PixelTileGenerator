"""
VQ-VAE v9 训练脚本

改进 (相对v8):
- 码本 1024 + embedding_dim 128 (容量翻倍)
- Perceptual Loss (VGG16 特征匹配, 帮助保留高频细节)
- Codebook Reset (自动重置低使用率码字)
- 损失: 0.5*MSE + 0.3*L1 + 0.2*Perceptual + VQ
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v4 import VQVAEv4


CONFIG = {
    "data_dir": "datasets/classified/pixel_64_quantized",
    "in_channels": 4,
    "hidden_channels": 256,
    "embedding_dim": 128,
    "num_embeddings": 1024,
    "commitment_cost": 0.25,
    "latent_size": 16,
    "batch_size": 16,
    "epochs": 600,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "mse_weight": 0.5,
    "l1_weight": 0.3,
    "perceptual_weight": 0.2,
    "checkpoint_dir": "checkpoints/vqvae_v9",
    "save_every": 50,
    "sample_every": 10,
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
        print(f"Loaded: {len(self.image_files)} images")

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


class PerceptualLoss(nn.Module):
    """VGG16 感知损失 — 帮助保留高频纹理细节"""

    def __init__(self, device="cpu"):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # RGBA→RGB: 取前3通道
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, x, y):
        # RGBA → RGB, 归一化
        x_rgb = x[:, :3, :, :]
        y_rgb = y[:, :3, :, :]
        x_norm = (x_rgb - self.mean) / self.std
        y_norm = (y_rgb - self.mean) / self.std

        # 64x64 → 224x224 (VGG 输入尺寸)
        x_up = nn.functional.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False)
        y_up = nn.functional.interpolate(y_norm, size=(224, 224), mode='bilinear', align_corners=False)

        feat_x = self.vgg(x_up)
        feat_y = self.vgg(y_up)

        return nn.functional.mse_loss(feat_x, feat_y)


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

            original_large = original_pil.resize((256, 256), Image.Resampling.NEAREST)
            recon_large = recon_quantized.resize((256, 256), Image.Resampling.NEAREST)

            original_large.save(samples_dir / f"epoch_{epoch:03d}_original_{i}.png")
            recon_large.save(samples_dir / f"epoch_{epoch:03d}_recon_{i}.png")

            comparison = Image.new("RGBA", (256 * 3 + 20, 256))
            comparison.paste(original_large, (0, 0))
            comparison.paste(recon_large, (256 + 10, 0))

            # 差分图 (放大差异便于观察)
            diff = Image.new("RGB", (64, 64))
            orig_px = original_pil.convert("RGB").load()
            recon_px = recon_quantized.convert("RGB").load()
            diff_px = diff.load()
            for yy in range(64):
                for xx in range(64):
                    r1, g1, b1 = orig_px[xx, yy]
                    r2, g2, b2 = recon_px[xx, yy]
                    diff_px[xx, yy] = (min(255, abs(r1-r2)*4), min(255, abs(g1-g2)*4), min(255, abs(b1-b2)*4))
            diff_large = diff.resize((256, 256), Image.Resampling.NEAREST)
            comparison.paste(diff_large, (512 + 20, 0))
            comparison.save(samples_dir / f"epoch_{epoch:03d}_comparison_{i}.png")

    model.train()


def train():
    print("=" * 60)
    print("VQ-VAE v9 (codebook reset + perceptual loss)")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Codebook: {CONFIG['num_embeddings']} x {CONFIG['embedding_dim']}d")
    print(f"Latent: {CONFIG['latent_size']}x{CONFIG['latent_size']} ({CONFIG['latent_size']**2} tokens)")
    print(f"Loss: {CONFIG['mse_weight']}*MSE + {CONFIG['l1_weight']}*L1 + {CONFIG['perceptual_weight']}*Perceptual")
    print(f"Batch: {CONFIG['batch_size']}  Epochs: {CONFIG['epochs']}")
    print("=" * 60)

    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms()
    dataset = TileDataset(project_root / CONFIG["data_dir"], transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
    )

    model = VQVAEv4(
        in_channels=CONFIG["in_channels"],
        hidden_channels=CONFIG["hidden_channels"],
        embedding_dim=CONFIG["embedding_dim"],
        num_embeddings=CONFIG["num_embeddings"],
        commitment_cost=CONFIG["commitment_cost"],
        latent_size=CONFIG["latent_size"],
    ).to(CONFIG["device"])

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(device=CONFIG["device"])

    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParams: {total_params:,}")
    print(f"Dataset: {len(dataset)}")
    print(f"Batches: {len(dataloader)}")
    print()

    history = {
        "total_loss": [], "recon_loss": [], "mse_loss": [],
        "l1_loss": [], "perceptual": [], "vq_loss": [], "codebook_usage": [],
    }

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_perceptual = 0.0
        epoch_vq = 0.0
        epoch_usage = 0.0
        num_batches = 0

        for images, names in dataloader:
            images = images.to(CONFIG["device"])
            recon, vq_loss, indices = model(images)

            mse = mse_loss(recon, images)
            l1 = l1_loss(recon, images)
            perc = perceptual_loss(recon, images)

            recon_loss = CONFIG["mse_weight"] * mse + CONFIG["l1_weight"] * l1 + CONFIG["perceptual_weight"] * perc
            total_loss = recon_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            usage, _ = model.vq.get_codebook_usage(indices)

            epoch_total += total_loss.item()
            epoch_mse += mse.item()
            epoch_l1 += l1.item()
            epoch_perceptual += perc.item()
            epoch_vq += vq_loss.item()
            epoch_usage += usage
            num_batches += 1

        avg_total = epoch_total / num_batches
        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_perc = epoch_perceptual / num_batches
        avg_vq = epoch_vq / num_batches
        avg_usage = epoch_usage / num_batches

        history["total_loss"].append(avg_total)
        history["mse_loss"].append(avg_mse)
        history["l1_loss"].append(avg_l1)
        history["perceptual"].append(avg_perc)
        history["vq_loss"].append(avg_vq)
        history["codebook_usage"].append(avg_usage)

        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Total: {avg_total:.4f} "
              f"MSE: {avg_mse:.4f} "
              f"L1: {avg_l1:.4f} "
              f"Perc: {avg_perc:.4f} "
              f"VQ: {avg_vq:.4f} "
              f"CB: {avg_usage:.1%} "
              f"T: {elapsed:.0f}s")

        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(model, dataset, CONFIG["device"], checkpoint_dir, epoch + 1)

        if (epoch + 1) % CONFIG["save_every"] == 0:
            path = checkpoint_dir / f"vqvae_v9_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
                "config": CONFIG,
            }, path)
            print(f"  Saved: {path}")

        if avg_total < best_loss:
            best_loss = avg_total
            best_path = checkpoint_dir / "vqvae_v9_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": CONFIG,
            }, best_path)

    final_path = checkpoint_dir / "vqvae_v9_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_total,
        "config": CONFIG,
    }, final_path)

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_total:.4f}")
    print(f"Codebook usage: {avg_usage:.1%}")
    print(f"Saved to: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

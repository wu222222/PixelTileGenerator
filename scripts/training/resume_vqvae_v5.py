"""
VQ-VAE v5 续跑脚本

功能:
- 加载最新的checkpoint
- 继续训练
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

from models.vq_vae_v2 import VQVAEv2


# 配置
CONFIG = {
    # 数据
    "data_dir": "datasets/classified/pixel_32_quantized",

    # 模型
    "in_channels": 4,
    "hidden_channels": 256,
    "embedding_dim": 64,
    "num_embeddings": 256,
    "commitment_cost": 0.25,

    # 训练参数
    "batch_size": 32,
    "additional_epochs": 500,  # 额外训练的epoch数
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,

    # 损失函数权重
    "mse_weight": 0.5,
    "l1_weight": 0.5,

    # 保存
    "checkpoint_dir": "checkpoints/vqvae_v5",
    "save_every": 50,
    "sample_every": 25,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class TileDataset(Dataset):
    """瓦片数据集"""

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
    """获取数据变换（带增强）"""
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
    """对图片进行颜色量化"""
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def save_samples(model, dataset, device, save_dir, epoch, num_samples=8):
    """保存重建样本"""
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


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint"""
    checkpoint_files = list(checkpoint_dir.glob("vqvae_v5_epoch_*.pth"))

    if not checkpoint_files:
        return None

    # 按epoch排序
    def get_epoch(path):
        try:
            return int(path.stem.split("_")[-1])
        except:
            return 0

    checkpoint_files.sort(key=get_epoch)
    return checkpoint_files[-1]


def train():
    """训练函数"""
    print("=" * 60)
    print("VQ-VAE v5 续跑训练")
    print("=" * 60)

    # 路径
    checkpoint_dir = project_root / CONFIG["checkpoint_dir"]

    # 查找最新checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        print("[错误] 没有找到checkpoint文件")
        print("请先运行: python scripts/training/train_vqvae_v5.py")
        return

    print(f"加载checkpoint: {latest_checkpoint}")

    # 加载checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=CONFIG["device"])
    start_epoch = checkpoint["epoch"]
    total_epochs = start_epoch + CONFIG["additional_epochs"]

    print(f"从epoch {start_epoch} 继续训练")
    print(f"目标epoch: {total_epochs}")
    print(f"设备: {CONFIG['device']}")
    print("=" * 60)

    # 数据变换
    transform = get_transforms()

    # 加载数据集
    dataset = TileDataset(project_root / CONFIG["data_dir"], transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 创建模型
    model = VQVAEv2(
        in_channels=CONFIG["in_channels"],
        hidden_channels=CONFIG["hidden_channels"],
        embedding_dim=CONFIG["embedding_dim"],
        num_embeddings=CONFIG["num_embeddings"],
        commitment_cost=CONFIG["commitment_cost"],
    ).to(CONFIG["device"])

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 损失函数
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # 加载优化器状态
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["additional_epochs"],
        last_epoch=-1
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"数据集大小: {len(dataset)}")
    print(f"Batch数量: {len(dataloader)}")
    print()

    # 加载或创建训练历史
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {
            "total_loss": [],
            "recon_loss": [],
            "mse_loss": [],
            "l1_loss": [],
            "vq_loss": [],
            "codebook_usage": [],
        }

    # 训练循环
    best_loss = checkpoint.get("loss", float("inf"))
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_usage = 0.0
        num_batches = 0

        for batch_idx, (images, names) in enumerate(dataloader):
            images = images.to(CONFIG["device"])

            # 前向传播
            recon, vq_loss, indices = model(images)

            # 计算重建损失
            mse = mse_loss(recon, images)
            l1 = l1_loss(recon, images)
            recon_loss = CONFIG["mse_weight"] * mse + CONFIG["l1_weight"] * l1

            # 总损失
            total_loss = recon_loss + vq_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 统计
            usage, _ = model.vq.get_codebook_usage(indices)

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_mse_loss += mse.item()
            epoch_l1_loss += l1.item()
            epoch_vq_loss += vq_loss.item()
            epoch_usage += usage
            num_batches += 1

        # 计算平均损失
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_mse_loss = epoch_mse_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        avg_usage = epoch_usage / num_batches

        # 更新历史
        history["total_loss"].append(avg_total_loss)
        history["recon_loss"].append(avg_recon_loss)
        history["mse_loss"].append(avg_mse_loss)
        history["l1_loss"].append(avg_l1_loss)
        history["vq_loss"].append(avg_vq_loss)
        history["codebook_usage"].append(avg_usage)

        # 更新学习率
        scheduler.step()

        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{total_epochs}] "
              f"Total: {avg_total_loss:.4f} "
              f"Recon: {avg_recon_loss:.4f} "
              f"MSE: {avg_mse_loss:.4f} "
              f"L1: {avg_l1_loss:.4f} "
              f"VQ: {avg_vq_loss:.4f} "
              f"Usage: {avg_usage:.2%} "
              f"Time: {elapsed_time:.1f}s")

        # 保存样本
        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(model, dataset, CONFIG["device"], checkpoint_dir, epoch + 1)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"vqvae_v5_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total_loss,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_path = checkpoint_dir / "vqvae_v5_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": CONFIG,
            }, best_path)

    # 保存最终模型
    final_path = checkpoint_dir / "vqvae_v5_final.pth"
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_total_loss,
        "config": CONFIG,
    }, final_path)

    # 保存训练历史
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 打印总结
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("续跑训练完成!")
    print("=" * 60)
    print(f"总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"起始epoch: {start_epoch}")
    print(f"结束epoch: {total_epochs}")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最终损失: {avg_total_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

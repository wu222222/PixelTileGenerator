"""
AutoEncoder 训练脚本

功能:
- 训练CNN AutoEncoder学习32×32像素瓦片的latent space
- 支持快速验证和正式训练
- 保存训练历史和模型检查点
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.autoencoder import AutoEncoder
from models.autoencoder_resnet import ResNetAutoEncoder


# 配置
CONFIG = {
    # 数据
    "data_dir": "datasets/classified/pixel_32_quantized",
    "image_size": 32,
    "channels": 4,  # RGBA

    # 模型
    "model_type": "resnet",  # "simple" 或 "resnet"
    "latent_dim": 128,

    # 训练参数
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,

    # 损失函数
    "loss_type": "l1",  # "mse", "l1", 或 "mse_l1"

    # 保存
    "checkpoint_dir": "checkpoints/autoencoder_resnet",
    "save_every": 10,  # 每10个epoch保存一次
    "sample_every": 5,  # 每5个epoch生成样本

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class TileDataset(Dataset):
    """瓦片数据集"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 获取所有图片文件
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

        # 加载图片
        img = Image.open(img_path).convert("RGBA")

        # 应用变换
        if self.transform:
            img = self.transform(img)

        return img, img_path.name


def get_transforms():
    """获取数据变换"""
    return transforms.Compose([
        transforms.ToTensor(),  # 转换为 [0, 1] 范围
    ])


def save_samples(model, dataset, device, save_dir, epoch, num_samples=8):
    """保存重建样本"""
    model.eval()

    # 创建保存目录
    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 获取随机样本
    indices = torch.randperm(len(dataset))[:num_samples]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, name = dataset[idx]
            img = img.unsqueeze(0).to(device)

            # 重建
            reconstruction = model(img)

            # 保存原图和重建图
            original = img.squeeze(0).cpu()
            recon = reconstruction.squeeze(0).cpu()

            # 转换为PIL图片
            original_pil = transforms.ToPILImage()(original)
            recon_pil = transforms.ToPILImage()(recon)

            # 创建对比图
            comparison = Image.new("RGBA", (32 * 2 + 10, 32))
            comparison.paste(original_pil, (0, 0))
            comparison.paste(recon_pil, (32 + 10, 0))

            # 放大保存
            comparison_large = comparison.resize((128 * 2 + 40, 128), Image.Resampling.NEAREST)
            comparison_large.save(samples_dir / f"epoch_{epoch:03d}_sample_{i}.png")

    model.train()


def train():
    """训练函数"""
    print("=" * 60)
    print("AutoEncoder 训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")
    print(f"Latent维度: {CONFIG['latent_dim']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}")
    print("=" * 60)

    # 创建保存目录
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 数据变换
    transform = get_transforms()

    # 加载数据集
    dataset = TileDataset(CONFIG["data_dir"], transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows下设为0
        pin_memory=True,
    )

    # 创建模型
    if CONFIG["model_type"] == "resnet":
        model = ResNetAutoEncoder(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])
    else:
        model = AutoEncoder(latent_dim=CONFIG["latent_dim"]).to(CONFIG["device"])

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 损失函数
    if CONFIG["loss_type"] == "mse":
        criterion = nn.MSELoss()
    elif CONFIG["loss_type"] == "l1":
        criterion = nn.L1Loss()
    elif CONFIG["loss_type"] == "mse_l1":
        mse_criterion = nn.MSELoss()
        l1_criterion = nn.L1Loss()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"数据集大小: {len(dataset)}")
    print(f"Batch数量: {len(dataloader)}")
    print(f"损失函数: {CONFIG['loss_type'].upper()}")
    print()

    # 训练历史
    history = {
        "train_loss": [],
        "learning_rate": [],
    }

    # 训练循环
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, names) in enumerate(dataloader):
            images = images.to(CONFIG["device"])

            # 前向传播
            reconstructions = model(images)

            # 计算损失
            if CONFIG["loss_type"] == "mse_l1":
                mse_loss = mse_criterion(reconstructions, images)
                l1_loss = l1_criterion(reconstructions, images)
                loss = mse_loss + 0.1 * l1_loss
            else:
                loss = criterion(reconstructions, images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        history["learning_rate"].append(scheduler.get_last_lr()[0])

        # 更新学习率
        scheduler.step()

        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {avg_loss:.6f} "
              f"LR: {scheduler.get_last_lr()[0]:.6f} "
              f"Time: {elapsed_time:.1f}s")

        # 保存样本
        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(model, dataset, CONFIG["device"], checkpoint_dir, epoch + 1)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"autoencoder_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "autoencoder_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": CONFIG,
            }, best_path)

    # 保存最终模型
    final_path = checkpoint_dir / "autoencoder_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
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
    print(f"最佳损失: {best_loss:.6f}")
    print(f"最终损失: {avg_loss:.6f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 60)


def main():
    train()


if __name__ == "__main__":
    main()

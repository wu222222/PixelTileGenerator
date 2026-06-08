"""
Conditional GAN 训练脚本

功能:
- 训练Conditional GAN根据类别生成32×32像素瓦片
- 使用预训练AutoEncoder的Decoder
- 保存训练历史和模型检查点
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

from models.autoencoder_resnet import ResNetAutoEncoder
from models.conditional_gan import ConditionalGenerator, ConditionalDiscriminator


# 配置
CONFIG = {
    # 数据
    "data_dir": "datasets/classified/pixel_32_filtered",
    "labels_file": "datasets/classified/pixel_32_filtered/class_labels.json",
    "autoencoder_checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",

    # 模型
    "latent_dim": 128,
    "embed_dim": 32,

    # 训练参数
    "batch_size": 64,
    "epochs": 200,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,

    # 损失权重
    "adversarial_weight": 1.0,
    "reconstruction_weight": 10.0,

    # 保存
    "checkpoint_dir": "checkpoints/conditional_gan",
    "save_every": 20,
    "sample_every": 10,

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class ConditionalTileDataset(Dataset):
    """带类别标签的瓦片数据集"""

    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 加载标签
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        self.categories = labels_data["categories"]
        self.category_counts = labels_data["category_counts"]

        # 创建类别到ID的映射
        self.category_names = sorted(self.category_counts.keys())
        self.category_to_id = {name: i for i, name in enumerate(self.category_names)}
        self.num_classes = len(self.category_names)

        # 获取图片文件列表
        self.image_files = []
        for img_name in self.categories.keys():
            img_path = self.data_dir / img_name
            if img_path.exists():
                self.image_files.append(img_path)

        print(f"加载数据集: {len(self.image_files)} 张图片, {self.num_classes} 个类别")
        print(f"类别映射:")
        for name, idx in self.category_to_id.items():
            print(f"  {idx}: {name} ({self.category_counts[name]}张)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # 加载图片
        img = Image.open(img_path).convert("RGBA")

        # 应用变换
        if self.transform:
            img = self.transform(img)

        # 获取类别ID
        category_name = self.categories[img_path.name]
        category_id = self.category_to_id[category_name]

        return img, category_id, img_path.name


def get_transforms():
    """获取数据变换"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def save_samples(generator, dataset, device, save_dir, epoch, num_per_class=2):
    """保存生成样本"""
    generator.eval()

    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for class_id in range(min(dataset.num_classes, 8)):  # 最多显示8个类别
            z = torch.randn(num_per_class, generator.latent_dim).to(device)
            labels = torch.full((num_per_class,), class_id, dtype=torch.long).to(device)

            generated = generator(z, labels)

            for i, img in enumerate(generated):
                img_pil = transforms.ToPILImage()(img.cpu())
                img_large = img_pil.resize((128, 128), Image.Resampling.NEAREST)
                class_name = dataset.category_names[class_id]
                img_large.save(samples_dir / f"epoch_{epoch:03d}_{class_name}_{i}.png")

    generator.train()


def train():
    """训练函数"""
    print("=" * 60)
    print("Conditional GAN 训练")
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
    dataset = ConditionalTileDataset(
        project_root / CONFIG["data_dir"],
        project_root / CONFIG["labels_file"],
        transform=transform
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 加载预训练AutoEncoder
    ae_checkpoint = torch.load(
        project_root / CONFIG["autoencoder_checkpoint"],
        map_location=CONFIG["device"]
    )
    ae_config = ae_checkpoint.get("config", {})
    ae_latent_dim = ae_config.get("latent_dim", 128)

    autoencoder = ResNetAutoEncoder(latent_dim=ae_latent_dim).to(CONFIG["device"])
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()

    print(f"\n预训练AutoEncoder加载成功")

    # 创建Conditional GAN
    generator = ConditionalGenerator(
        latent_dim=CONFIG["latent_dim"],
        num_classes=dataset.num_classes,
        embed_dim=CONFIG["embed_dim"],
        pretrained_decoder=autoencoder.decoder,
    ).to(CONFIG["device"])

    discriminator = ConditionalDiscriminator(
        num_classes=dataset.num_classes,
        embed_dim=CONFIG["embed_dim"],
    ).to(CONFIG["device"])

    # 损失函数
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

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
    }

    # 训练循环
    best_g_loss = float("inf")
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for batch_idx, (real_images, labels, names) in enumerate(dataloader):
            batch_size = real_images.size(0)

            real_images = real_images.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            # 标签
            real_labels = torch.ones(batch_size, 1).to(CONFIG["device"])
            fake_labels = torch.zeros(batch_size, 1).to(CONFIG["device"])

            # ==================== 训练判别器 ====================
            optimizer_d.zero_grad()

            # 真图片
            d_real = discriminator(real_images, labels)
            d_real_loss = adversarial_loss(d_real, real_labels)

            # 假图片
            z = torch.randn(batch_size, CONFIG["latent_dim"]).to(CONFIG["device"])
            fake_images = generator(z, labels)
            d_fake = discriminator(fake_images.detach(), labels)
            d_fake_loss = adversarial_loss(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # ==================== 训练生成器 ====================
            optimizer_g.zero_grad()

            # 对抗损失
            d_fake = discriminator(fake_images, labels)
            g_adversarial_loss = adversarial_loss(d_fake, real_labels)

            # 重建损失 (用AutoEncoder的latent)
            with torch.no_grad():
                real_latent = autoencoder.encode(real_images)
            fake_from_real_latent = generator(real_latent, labels)
            g_reconstruction_loss = reconstruction_loss(fake_from_real_latent, real_images)

            g_loss = (CONFIG["adversarial_weight"] * g_adversarial_loss +
                      CONFIG["reconstruction_weight"] * g_reconstruction_loss)

            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        # 计算平均损失
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)

        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"G_loss: {avg_g_loss:.4f} "
              f"D_loss: {avg_d_loss:.4f} "
              f"Time: {elapsed_time:.1f}s")

        # 保存样本
        if (epoch + 1) % CONFIG["sample_every"] == 0:
            save_samples(generator, dataset, CONFIG["device"], checkpoint_dir, epoch + 1)

        # 保存检查点
        if (epoch + 1) % CONFIG["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"conditional_gan_epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": avg_g_loss,
                "d_loss": avg_d_loss,
                "config": CONFIG,
                "category_names": dataset.category_names,
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_path = checkpoint_dir / "conditional_gan_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": best_g_loss,
                "d_loss": avg_d_loss,
                "config": CONFIG,
                "category_names": dataset.category_names,
            }, best_path)

    # 保存最终模型
    final_path = checkpoint_dir / "conditional_gan_final.pth"
    torch.save({
        "epoch": CONFIG["epochs"],
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "g_loss": avg_g_loss,
        "d_loss": avg_d_loss,
        "config": CONFIG,
        "category_names": dataset.category_names,
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

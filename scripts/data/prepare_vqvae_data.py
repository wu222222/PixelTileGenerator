"""
VQ-VAE数据准备脚本

功能:
- 加载量化后的图片
- 数据增强（翻转、旋转）
- 保存为训练数据
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "data_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "datasets/vqvae_data",
    "augment_factor": 4,  # 增强倍数
}


class TileDataset(Dataset):
    """瓦片数据集（带数据增强）"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 获取所有图片
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


def get_transforms(augment=True):
    """获取数据变换"""
    if augment:
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
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


def main():
    print("=" * 60)
    print("VQ-VAE数据准备")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    data_dir = base_dir / CONFIG["data_dir"]
    output_dir = base_dir / CONFIG["output_dir"]

    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据变换（带增强）
    transform = get_transforms(augment=True)

    # 加载数据集
    dataset = TileDataset(data_dir, transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # 收集所有数据
    print("\n收集数据...")
    all_images = []
    all_names = []

    for batch_idx, (images, names) in enumerate(dataloader):
        all_images.append(images)
        all_names.extend(names)

        if (batch_idx + 1) % 10 == 0:
            print(f"  进度: {(batch_idx + 1) * 32}/{len(dataset)}")

    # 合并所有数据
    all_images = torch.cat(all_images, dim=0)
    print(f"\n数据形状: {all_images.shape}")

    # 保存数据
    print("保存数据...")
    torch.save(all_images, output_dir / "images.pt")

    with open(output_dir / "names.json", "w") as f:
        json.dump(all_names, f)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "data_dir": str(data_dir),
            "num_samples": len(all_images),
            "image_size": 32,
            "channels": 4,
            "augmented": True,
        }, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print(f"数据形状: {all_images.shape}")
    print(f"输出目录: {output_dir}")
    print(f"文件:")
    print(f"  - images.pt (图片数据)")
    print(f"  - names.json (文件名)")
    print(f"  - config.json (配置)")
    print("=" * 60)


if __name__ == "__main__":
    main()

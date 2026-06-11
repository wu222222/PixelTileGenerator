"""
Latent GAN 评估脚本

功能:
- 加载训练好的Latent GAN
- 生成latent vectors
- 用Decoder生成图片
- 颜色量化
- 创建评估图
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.autoencoder_resnet import ResNetAutoEncoder
from scripts.training.train_latent_gan import LatentGenerator


# 配置
CONFIG = {
    "latent_gan_checkpoint": "checkpoints/latent_gan/latent_gan_best.pth",
    "autoencoder_checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",
    "output_dir": "checkpoints/latent_gan/evaluation",
    "num_samples": 25,
    "palette_colors": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    """对图片进行颜色量化"""
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def main():
    print("=" * 60)
    print("Latent GAN 评估")
    print("=" * 60)

    # 路径
    latent_gan_path = project_root / CONFIG["latent_gan_checkpoint"]
    autoencoder_path = project_root / CONFIG["autoencoder_checkpoint"]
    output_dir = project_root / CONFIG["output_dir"]

    if not latent_gan_path.exists():
        print(f"[错误] Latent GAN模型不存在: {latent_gan_path}")
        return

    if not autoencoder_path.exists():
        print(f"[错误] AutoEncoder模型不存在: {autoencoder_path}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载Latent GAN
    print(f"加载Latent GAN: {latent_gan_path}")
    latent_checkpoint = torch.load(latent_gan_path, map_location=CONFIG["device"])
    latent_config = latent_checkpoint.get("config", {})
    latent_dim = latent_config.get("latent_dim", 128)

    latent_generator = LatentGenerator(latent_dim=latent_dim).to(CONFIG["device"])
    latent_generator.load_state_dict(latent_checkpoint["generator_state_dict"])
    latent_generator.eval()

    # 加载AutoEncoder
    print(f"加载AutoEncoder: {autoencoder_path}")
    ae_checkpoint = torch.load(autoencoder_path, map_location=CONFIG["device"])
    ae_config = ae_checkpoint.get("config", {})
    ae_latent_dim = ae_config.get("latent_dim", 128)

    autoencoder = ResNetAutoEncoder(latent_dim=ae_latent_dim).to(CONFIG["device"])
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()

    # 生成样本
    print(f"\n生成 {CONFIG['num_samples']} 个样本...")
    z = torch.randn(CONFIG["num_samples"], latent_dim).to(CONFIG["device"])

    with torch.no_grad():
        # 生成latent vectors
        fake_z = latent_generator(z)

        # 用Decoder生成图片
        fake_images = autoencoder.decode(fake_z)

    # 转换为PIL图片
    original_images = []
    quantized_images = []

    for i in range(CONFIG["num_samples"]):
        img_tensor = fake_images[i].cpu()
        img_pil = transforms.ToPILImage()(img_tensor)

        # 量化
        img_quantized = quantize_image(img_pil, CONFIG["palette_colors"])

        original_images.append(img_pil)
        quantized_images.append(img_quantized)

    # 保存单张图片
    print("保存图片...")
    for i in range(CONFIG["num_samples"]):
        # 原图
        img_large = original_images[i].resize((128, 128), Image.Resampling.NEAREST)
        img_large.save(output_dir / f"original_{i:02d}.png")

        # 量化后
        img_large = quantized_images[i].resize((128, 128), Image.Resampling.NEAREST)
        img_large.save(output_dir / f"quantized_{i:02d}.png")

    # 创建网格图
    print("创建网格图...")
    create_grid(original_images, output_dir / "grid_original.png", "Original")
    create_grid(quantized_images, output_dir / "grid_quantized.png", f"Quantized ({CONFIG['palette_colors']} colors)")

    # 创建对比图
    print("创建对比图...")
    create_comparison(original_images, quantized_images, output_dir / "comparison.png")

    # 打印总结
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n生成的文件:")
    print(f"  - original_XX.png     (原图)")
    print(f"  - quantized_XX.png    (量化后)")
    print(f"  - grid_original.png   (原图网格)")
    print(f"  - grid_quantized.png  (量化网格)")
    print(f"  - comparison.png      (对比图)")
    print("=" * 60)


def create_grid(images, output_path, title, cols=5):
    """创建网格图"""
    import math

    num_images = len(images)
    rows = math.ceil(num_images / cols)
    cell_size = 128
    padding = 5

    total_width = cols * (cell_size + padding) + padding
    total_height = rows * (cell_size + padding) + padding + 40

    grid_img = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid_img)

    draw.text((padding, 10), title, fill="white")

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = padding + col * (cell_size + padding)
        y = 40 + row * (cell_size + padding)

        img_large = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        grid_img.paste(img_large, (x, y))

    grid_img.save(output_path)


def create_comparison(originals, quantizeds, output_path, num_show=16):
    """创建对比图"""
    import math

    num_images = min(num_show, len(originals))
    cols = 4
    rows = math.ceil(num_images / cols)
    cell_size = 64
    padding = 5

    section_width = cols * (cell_size + padding) + padding
    total_width = section_width * 2 + 100
    total_height = rows * (cell_size + padding) + padding + 40

    comparison = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(comparison)

    draw.text((10, 10), "Original", fill="white")
    draw.text((section_width + 100, 10), "Quantized", fill="white")

    for i in range(num_images):
        row = i // cols
        col = i % cols

        x1 = padding + col * (cell_size + padding)
        y1 = 40 + row * (cell_size + padding)

        x2 = section_width + 100 + col * (cell_size + padding)
        y2 = y1

        img1 = originals[i].resize((cell_size, cell_size), Image.Resampling.NEAREST)
        img2 = quantizeds[i].resize((cell_size, cell_size), Image.Resampling.NEAREST)

        comparison.paste(img1, (x1, y1))
        comparison.paste(img2, (x2, y2))

    comparison.save(output_path)


if __name__ == "__main__":
    main()

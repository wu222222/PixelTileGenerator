"""
Latent空间插值测试

功能:
- 选择两张图片
- 在latent空间中插值
- 生成插值结果
- 验证latent空间连续性
"""

import sys
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v5/vqvae_v5_best.pth",
    "data_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "checkpoints/vqvae_v5/interpolation",
    "num_interpolations": 10,
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
    print("Latent空间插值测试")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    data_dir = project_root / CONFIG["data_dir"]
    output_dir = project_root / CONFIG["output_dir"]

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载VQ-VAE
    print(f"加载VQ-VAE: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})

    vqvae = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
    ).to(CONFIG["device"])

    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    # 选择两张不同的图片
    if len(image_files) < 2:
        print("[错误] 需要至少2张图片")
        return

    img_path1 = image_files[0]
    img_path2 = image_files[100] if len(image_files) > 100 else image_files[1]

    print(f"图片1: {img_path1.name}")
    print(f"图片2: {img_path2.name}")

    # 加载图片
    transform = transforms.Compose([transforms.ToTensor()])
    img1 = transform(Image.open(img_path1).convert("RGBA")).unsqueeze(0).to(CONFIG["device"])
    img2 = transform(Image.open(img_path2).convert("RGBA")).unsqueeze(0).to(CONFIG["device"])

    # 提取latent
    with torch.no_grad():
        z1, _ = vqvae.encode(img1)
        z2, _ = vqvae.encode(img2)

    print(f"Latent形状: {z1.shape}")

    # 插值
    print(f"\n生成 {CONFIG['num_interpolations']} 个插值...")
    interpolation_images = []

    for i in range(CONFIG["num_interpolations"] + 1):
        alpha = i / CONFIG["num_interpolations"]

        # 线性插值
        z_interp = (1 - alpha) * z1 + alpha * z2

        # 解码
        with torch.no_grad():
            img_interp = vqvae.decode(z_interp)

        # 转换为PIL图片
        img_tensor = img_interp.squeeze(0).cpu()
        img_pil = transforms.ToPILImage()(img_tensor)
        img_quantized = quantize_image(img_pil, colors=32)

        interpolation_images.append(img_quantized)

        print(f"  alpha={alpha:.2f}")

    # 创建插值网格
    print("\n创建插值网格...")
    cell_size = 128
    padding = 5
    num_images = len(interpolation_images)

    total_width = num_images * (cell_size + padding) + padding
    total_height = cell_size + padding * 2 + 40

    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), f"Latent Interpolation: {img_path1.name} → {img_path2.name}", fill="white")

    # 绘制图片
    for i, img in enumerate(interpolation_images):
        x = padding + i * (cell_size + padding)
        y = 40 + padding

        img_large = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        grid.paste(img_large, (x, y))

        # 绘制alpha值
        alpha = i / CONFIG["num_interpolations"]
        draw.text((x, y + cell_size + 5), f"{alpha:.2f}", fill="white")

    # 保存
    grid.save(output_dir / "interpolation_grid.png")

    # 保存单张图片
    for i, img in enumerate(interpolation_images):
        img.save(output_dir / f"interp_{i:02d}.png")

    print("\n" + "=" * 60)
    print("插值测试完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n请查看插值结果:")
    print("  - 如果过渡自然 → latent空间连续，GAN问题可解决")
    print("  - 如果有突变/模糊 → latent空间不连续，需要其他方案")
    print("=" * 60)


if __name__ == "__main__":
    main()

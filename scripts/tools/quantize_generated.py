"""
生成样本颜色量化脚本

功能:
- 加载训练好的WGAN-GP模型
- 生成样本
- 对生成样本做颜色量化
- 创建对比图
"""

import sys
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 配置
CONFIG = {
    "checkpoint": "checkpoints/terrain_wgan/terrain_wgan_best.pth",
    "output_dir": "checkpoints/terrain_wgan/quantized_samples",
    "num_samples": 16,
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


def load_generator(checkpoint_path, device):
    """加载生成器"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    latent_dim = config.get("latent_dim", 128)

    # 导入生成器类
    from scripts.training.train_terrain_wgan import Generator
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    return generator, latent_dim


def main():
    print("=" * 60)
    print("生成样本颜色量化")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["checkpoint"]
    output_dir = project_root / CONFIG["output_dir"]

    if not checkpoint_path.exists():
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    generator, latent_dim = load_generator(checkpoint_path, CONFIG["device"])
    print(f"Latent维度: {latent_dim}")

    # 生成样本
    print(f"\n生成 {CONFIG['num_samples']} 个样本...")
    torch.manual_seed(42)  # 固定种子，方便对比
    z = torch.randn(CONFIG["num_samples"], latent_dim).to(CONFIG["device"])

    with torch.no_grad():
        generated = generator(z)

    # 转换为PIL图片
    original_images = []
    quantized_images = []

    for i in range(CONFIG["num_samples"]):
        img_tensor = generated[i].cpu()
        img_pil = transforms.ToPILImage()(img_tensor)

        # 量化
        img_quantized = quantize_image(img_pil, CONFIG["palette_colors"])

        original_images.append(img_pil)
        quantized_images.append(img_quantized)

    # 保存单张图片
    print("\n保存图片...")
    for i in range(CONFIG["num_samples"]):
        # 原图
        img_large = original_images[i].resize((128, 128), Image.Resampling.NEAREST)
        img_large.save(output_dir / f"original_{i:02d}.png")

        # 量化后
        img_large = quantized_images[i].resize((128, 128), Image.Resampling.NEAREST)
        img_large.save(output_dir / f"quantized_{i:02d}.png")

    # 创建对比图
    print("创建对比图...")
    grid_size = int(CONFIG["num_samples"] ** 0.5)
    cell_size = 128
    padding = 10

    # 对比图宽度: 原图 + 间距 + 量化图
    comparison_width = grid_size * cell_size * 2 + padding * 3
    comparison_height = grid_size * cell_size + padding * 2

    comparison = Image.new("RGBA", (comparison_width, comparison_height), (40, 40, 40, 255))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(comparison)

    # 标题
    draw.text((padding, 5), "Original", fill="white")
    draw.text((grid_size * cell_size + padding * 2, 5), f"Quantized ({CONFIG['palette_colors']} colors)", fill="white")

    for i in range(CONFIG["num_samples"]):
        row = i // grid_size
        col = i % grid_size

        # 原图位置
        x1 = padding + col * cell_size
        y1 = padding + row * cell_size

        # 量化图位置
        x2 = grid_size * cell_size + padding * 2 + col * cell_size
        y2 = y1

        # 粘贴图片
        comparison.paste(original_images[i].resize((cell_size, cell_size), Image.Resampling.NEAREST), (x1, y1))
        comparison.paste(quantized_images[i].resize((cell_size, cell_size), Image.Resampling.NEAREST), (x2, y2))

    # 保存对比图
    comparison.save(output_dir / "comparison.png")

    # 创建单独的量化网格图
    quantized_grid = Image.new("RGBA", (grid_size * cell_size + padding * 2, grid_size * cell_size + padding * 2), (40, 40, 40, 255))
    for i in range(CONFIG["num_samples"]):
        row = i // grid_size
        col = i % grid_size
        x = padding + col * cell_size
        y = padding + row * cell_size
        quantized_grid.paste(quantized_images[i].resize((cell_size, cell_size), Image.Resampling.NEAREST), (x, y))

    quantized_grid.save(output_dir / "quantized_grid.png")

    # 打印总结
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  - original_XX.png    (原图)")
    print(f"  - quantized_XX.png   (量化后)")
    print(f"  - comparison.png     (对比图)")
    print(f"  - quantized_grid.png (量化网格)")
    print("=" * 60)
    print("\n请对比查看:")
    print("  1. 量化后是否更清晰?")
    print("  2. 是否能看出草地/石头/泥土纹理?")
    print("  3. 边缘是否更锐利?")


if __name__ == "__main__":
    main()

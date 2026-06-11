"""
Structure GAN 评估脚本

功能:
- 加载训练好的模型
- 生成Structure Map样本
- 用不同调色板渲染
- 创建评估对比图
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.structure_generator import StructureGenerator


# 配置
CONFIG = {
    "checkpoint": "checkpoints/structure_gan/structure_gan_best.pth",
    "output_dir": "checkpoints/structure_gan/evaluation",
    "num_samples": 25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 预定义调色板
PALETTES = {
    "grass": [
        [34, 139, 34],    # 深绿
        [50, 205, 50],    # 绿色
        [144, 238, 144],  # 浅绿
        [152, 251, 152],  # 淡绿
        [85, 107, 47],    # 暗绿
        [107, 142, 35],   # 橄榄绿
        [60, 179, 113],   # 中绿
        [46, 139, 87],    # 海绿
        [0, 128, 0],      # 绿色
        [0, 100, 0],      # 深绿
        [127, 255, 0],    # 黄绿
        [173, 255, 47],   # 黄绿
        [50, 120, 50],    # 深绿
        [80, 160, 80],    # 中绿
        [100, 200, 100],  # 浅绿
        [120, 220, 120],  # 淡绿
    ],
    "stone": [
        [128, 128, 128],  # 灰色
        [169, 169, 169],  # 深灰
        [192, 192, 192],  # 银色
        [211, 211, 211],  # 浅灰
        [105, 105, 105],  # 暗灰
        [139, 137, 137],  # 灰色
        [160, 160, 160],  # 灰色
        [180, 180, 180],  # 灰色
        [145, 142, 140],  # 灰色
        [155, 153, 150],  # 灰色
        [170, 168, 165],  # 灰色
        [130, 128, 125],  # 灰色
        [140, 138, 135],  # 灰色
        [150, 148, 145],  # 灰色
        [175, 173, 170],  # 灰色
        [165, 163, 160],  # 灰色
    ],
    "dirt": [
        [139, 90, 43],    # 棕色
        [160, 82, 45],    # 深棕
        [184, 115, 51],   # 中棕
        [205, 133, 63],   # 浅棕
        [120, 60, 30],    # 深棕
        [150, 100, 50],   # 中棕
        [170, 120, 70],   # 浅棕
        [190, 140, 90],   # 淡棕
        [139, 90, 43],    # 棕色
        [120, 80, 40],    # 深棕
        [160, 100, 50],   # 中棕
        [180, 120, 60],   # 浅棕
        [140, 90, 45],    # 棕色
        [155, 105, 55],   # 中棕
        [175, 125, 65],   # 浅棕
        [195, 145, 85],   # 淡棕
    ],
    "lava": [
        [255, 0, 0],      # 红色
        [255, 69, 0],     # 红橙
        [255, 99, 71],    # 番茄
        [255, 140, 0],    # 深橙
        [255, 165, 0],    # 橙色
        [255, 215, 0],    # 金色
        [255, 255, 0],    # 黄色
        [255, 255, 224],  # 浅黄
        [200, 0, 0],      # 深红
        [220, 20, 20],    # 红色
        [240, 40, 40],    # 红色
        [255, 80, 0],     # 橙红
        [255, 120, 0],    # 橙色
        [255, 180, 0],    # 金橙
        [255, 220, 50],   # 黄橙
        [255, 240, 100],  # 浅黄
    ],
    "snow": [
        [255, 250, 250],  # 雪白
        [240, 248, 255],  # 爱丽丝蓝
    [230, 230, 250],  # 薰衣草
        [220, 220, 220],  # 浅灰
        [210, 210, 210],  # 灰色
        [200, 200, 200],  # 灰色
        [190, 190, 190],  # 灰色
        [180, 180, 180],  # 灰色
        [245, 245, 245],  # 白烟
        [235, 235, 235],  # 浅灰
        [225, 225, 225],  # 灰色
        [215, 215, 215],  # 灰色
        [205, 205, 205],  # 灰色
        [195, 195, 195],  # 灰色
        [185, 185, 185],  # 灰色
        [175, 175, 175],  # 灰色
    ],
}


def indices_to_image(indices, palette):
    """将索引图转换为图像"""
    h, w = indices.shape
    img = Image.new("RGB", (w, h))

    for y in range(h):
        for x in range(w):
            idx = indices[y, x]
            if idx < len(palette):
                img.putpixel((x, y), tuple(palette[idx]))

    return img


def create_evaluation_grid(generator, device, num_samples=25):
    """创建评估网格图"""
    # 生成样本
    z = torch.randn(num_samples, generator.latent_dim).to(device)

    with torch.no_grad():
        indices = generator.generate(z).cpu().numpy()

    # 为每个调色板创建网格
    results = {}

    for palette_name, palette in PALETTES.items():
        # 计算网格大小
        import math
        grid_size = math.ceil(math.sqrt(num_samples))
        cell_size = 128
        padding = 5

        # 计算总尺寸
        total_width = grid_size * (cell_size + padding) + padding
        total_height = grid_size * (cell_size + padding) + padding + 40  # 40 for title

        # 创建图片
        grid_img = Image.new("RGB", (total_width, total_height), (40, 40, 40))
        draw = ImageDraw.Draw(grid_img)

        # 标题
        draw.text((padding, 10), f"{palette_name.upper()} ({num_samples} samples)", fill="white")

        # 绘制图片
        for i in range(num_samples):
            row = i // grid_size
            col = i % grid_size

            x = padding + col * (cell_size + padding)
            y = 40 + row * (cell_size + padding)

            # 转换为图像
            tile_img = indices_to_image(indices[i], palette)
            tile_img_large = tile_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

            grid_img.paste(tile_img_large, (x, y))

        results[palette_name] = grid_img

    return results, indices


def main():
    print("=" * 60)
    print("Structure GAN 评估")
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
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})

    generator = StructureGenerator(
        latent_dim=config.get("latent_dim", 128),
        num_classes=config.get("num_classes", 32),
    ).to(CONFIG["device"])

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # 生成评估图
    print(f"\n生成 {CONFIG['num_samples']} 个样本...")
    results, indices = create_evaluation_grid(generator, CONFIG["device"], CONFIG["num_samples"])

    # 保存结果
    print("\n保存评估图...")
    for palette_name, grid_img in results.items():
        save_path = output_dir / f"evaluation_{palette_name}.png"
        grid_img.save(save_path)
        print(f"  {palette_name}: {save_path}")

    # 保存索引图
    indices_dir = output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    for i in range(CONFIG["num_samples"]):
        idx = indices[i]
        idx_img = Image.fromarray((idx * 8).astype(np.uint8), mode='L')
        idx_img_large = idx_img.resize((128, 128), Image.Resampling.NEAREST)
        idx_img_large.save(indices_dir / f"index_{i:02d}.png")

    # 创建对比图（索引图 vs 渲染图）
    print("\n创建对比图...")
    create_comparison(indices, output_dir)

    # 打印总结
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n生成的文件:")
    print(f"  - evaluation_grass.png    (草地调色板)")
    print(f"  - evaluation_stone.png    (石头调色板)")
    print(f"  - evaluation_dirt.png     (泥土调色板)")
    print(f"  - evaluation_lava.png     (岩浆调色板)")
    print(f"  - evaluation_snow.png     (雪地调色板)")
    print(f"  - indices/               (索引图)")
    print(f"  - comparison.png         (对比图)")
    print("=" * 60)


def create_comparison(indices, output_dir):
    """创建索引图和渲染图的对比"""
    num_samples = min(16, len(indices))
    cell_size = 64
    padding = 5

    # 计算布局
    cols = 4
    rows = (num_samples + cols - 1) // cols

    # 总尺寸
    section_width = cols * (cell_size + padding) + padding
    total_width = section_width * 2 + 100  # 左边索引图，右边渲染图
    total_height = rows * (cell_size + padding) + padding + 40

    # 创建图片
    comparison = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(comparison)

    # 标题
    draw.text((10, 10), "Index Map", fill="white")
    draw.text((section_width + 100, 10), "Rendered (Grass)", fill="white")

    # 使用grass调色板
    palette = PALETTES["grass"]

    # 绘制图片
    for i in range(num_samples):
        row = i // cols
        col = i % cols

        # 索引图位置
        x1 = padding + col * (cell_size + padding)
        y1 = 40 + row * (cell_size + padding)

        # 渲染图位置
        x2 = section_width + 100 + col * (cell_size + padding)
        y2 = y1

        # 索引图
        idx = indices[i]
        idx_img = Image.fromarray((idx * 8).astype(np.uint8), mode='L')
        idx_img_large = idx_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        comparison.paste(idx_img_large, (x1, y1))

        # 渲染图
        rendered = indices_to_image(idx, palette)
        rendered_large = rendered.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        comparison.paste(rendered_large, (x2, y2))

    # 保存
    comparison.save(output_dir / "comparison.png")


if __name__ == "__main__":
    main()

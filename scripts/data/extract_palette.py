"""
提取调色板脚本

功能:
- 从数据集提取调色板
- 使用K-means聚类找到主要颜色
- 保存调色板到JSON文件
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "data_dir": "datasets/classified/pixel_32_filtered",
    "output_file": "datasets/classified/pixel_32_filtered/palette.json",
    "num_colors": 32,
}


def extract_colors_from_image(img_path):
    """从单张图片提取颜色"""
    try:
        img = Image.open(img_path).convert("RGBA")
        pixels = np.array(img)

        # 只保留不透明的像素
        if pixels.shape[2] == 4:  # RGBA
            mask = pixels[:, :, 3] > 128  # alpha > 128
            pixels = pixels[mask][:, :3]  # 只取RGB
        else:
            pixels = pixels.reshape(-1, 3)

        return pixels
    except Exception as e:
        print(f"  [错误] {img_path.name}: {e}")
        return np.array([])


def extract_palette():
    """提取调色板"""
    base_dir = CONFIG["project_root"]
    data_dir = base_dir / CONFIG["data_dir"]
    output_path = base_dir / CONFIG["output_file"]

    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        return

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"找到 {len(image_files)} 张图片")

    # 提取所有颜色
    print("提取颜色...")
    all_colors = []

    for i, img_path in enumerate(image_files):
        colors = extract_colors_from_image(img_path)
        if len(colors) > 0:
            # 随机采样，避免内存溢出
            if len(colors) > 100:
                indices = np.random.choice(len(colors), 100, replace=False)
                colors = colors[indices]
            all_colors.append(colors)

        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(image_files)}")

    # 合并所有颜色
    all_colors = np.vstack(all_colors)
    print(f"总颜色数: {len(all_colors)}")

    # K-means聚类
    print(f"K-means聚类 (k={CONFIG['num_colors']})...")
    kmeans = KMeans(n_clusters=CONFIG["num_colors"], random_state=42, n_init=10)
    kmeans.fit(all_colors)

    # 获取调色板
    palette = kmeans.cluster_centers_.astype(int)
    palette = np.clip(palette, 0, 255)

    # 统计每个颜色的使用频率
    labels = kmeans.labels_
    color_counts = np.bincount(labels, minlength=CONFIG["num_colors"])

    # 按频率排序
    sorted_indices = np.argsort(-color_counts)
    palette = palette[sorted_indices]
    color_counts = color_counts[sorted_indices]

    # 保存调色板
    palette_data = {
        "num_colors": CONFIG["num_colors"],
        "palette": palette.tolist(),
        "color_counts": color_counts.tolist(),
        "total_pixels": len(all_colors),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(palette_data, f, indent=2)

    # 打印调色板
    print(f"\n调色板 (前10个颜色):")
    for i in range(min(10, len(palette))):
        r, g, b = palette[i]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        percentage = color_counts[i] / len(all_colors) * 100
        print(f"  {i+1}. RGB({r},{g},{b}) {hex_color} - {percentage:.1f}%")

    print(f"\n调色板已保存: {output_path}")


def visualize_palette():
    """可视化调色板"""
    base_dir = CONFIG["project_root"]
    palette_path = base_dir / CONFIG["output_file"]

    if not palette_path.exists():
        print(f"[错误] 调色板文件不存在: {palette_path}")
        return

    # 加载调色板
    with open(palette_path, "r") as f:
        palette_data = json.load(f)

    palette = np.array(palette_data["palette"])
    num_colors = len(palette)

    # 创建调色板图片
    cell_size = 40
    cols = 8
    rows = (num_colors + cols - 1) // cols

    img_width = cols * cell_size
    img_height = rows * cell_size

    img = Image.new("RGB", (img_width, img_height), (40, 40, 40))

    for i, color in enumerate(palette):
        row = i // cols
        col = i % cols
        x = col * cell_size
        y = row * cell_size

        # 绘制颜色块
        for dx in range(cell_size):
            for dy in range(cell_size):
                img.putpixel((x + dx, y + dy), tuple(color))

    # 保存
    output_path = base_dir / CONFIG["output_file"].replace(".json", ".png")
    img.save(output_path)
    print(f"调色板图片已保存: {output_path}")


def main():
    print("=" * 60)
    print("提取调色板")
    print("=" * 60)
    print(f"目标颜色数: {CONFIG['num_colors']}")
    print("=" * 60)

    extract_palette()
    visualize_palette()


if __name__ == "__main__":
    main()

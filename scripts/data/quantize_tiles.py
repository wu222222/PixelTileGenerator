"""
瓦片颜色量化脚本

功能:
- 对pixel_32目录中的所有图片进行颜色量化
- 生成量化后的图片到新目录
- 生成对比图
- 用JSON记录量化状态
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "datasets/classified/pixel_32",
    "output_dir": "datasets/classified/pixel_32_quantized",
    "comparison_dir": "datasets/classified/quantization_comparisons",
    "palette_colors": 32,
    "tiles_per_comparison": 50,  # 每张对比图显示50个瓦片
    "comparison_cols": 5,  # 5列
    "comparison_rows": 10,  # 10行
}


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    """对图片进行颜色量化"""
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def create_comparison(original: Image.Image, quantized: Image.Image, scale: int = 4) -> Image.Image:
    """创建对比图"""
    img_size = original.width * scale
    padding = 10
    label_height = 30

    # 创建对比图
    comparison_width = img_size * 2 + padding * 3
    comparison_height = img_size + label_height + padding * 2

    comparison = Image.new("RGBA", (comparison_width, comparison_height), (40, 40, 40, 255))
    draw = ImageDraw.Draw(comparison)

    # 标题
    draw.text((padding, 5), "Original", fill="white")
    draw.text((img_size + padding * 2, 5), f"Quantized ({CONFIG['palette_colors']} colors)", fill="white")

    # 放大图片
    original_large = original.resize((img_size, img_size), Image.Resampling.NEAREST)
    quantized_large = quantized.resize((img_size, img_size), Image.Resampling.NEAREST)

    # 粘贴图片
    comparison.paste(original_large, (padding, label_height + padding))
    comparison.paste(quantized_large, (img_size + padding * 2, label_height + padding))

    return comparison


def main():
    print("=" * 60)
    print("瓦片颜色量化")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    output_dir = base_dir / CONFIG["output_dir"]
    comparison_dir = base_dir / CONFIG["comparison_dir"]

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"找到 {len(image_files)} 张图片")

    # 量化状态记录
    quantize_status = {}

    # 处理每张图片
    print(f"\n量化中 (目标颜色数: {CONFIG['palette_colors']})...")

    for i, img_path in enumerate(image_files):
        try:
            # 打开原图
            original = Image.open(img_path).convert("RGBA")

            # 统计原始颜色数
            original_colors = len(original.getcolors(maxcolors=1024) or [])

            # 量化
            quantized = quantize_image(original, CONFIG["palette_colors"])

            # 统计量化后颜色数
            quantized_colors = len(quantized.getcolors(maxcolors=1024) or [])

            # 保存量化后的图片
            quantized.save(output_dir / img_path.name)

            # 生成对比图
            comparison = create_comparison(original, quantized)
            comparison.save(comparison_dir / f"compare_{img_path.stem}.png")

            # 记录状态
            quantize_status[img_path.name] = {
                "original_colors": original_colors,
                "quantized_colors": quantized_colors,
                "quantized": True,
            }

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(image_files)}")

        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")
            quantize_status[img_path.name] = {
                "quantized": False,
                "error": str(e),
            }

    # 保存量化状态
    status_path = output_dir / "quantize_status.json"
    status_data = {
        "config": {
            "palette_colors": CONFIG["palette_colors"],
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
        "total_images": len(image_files),
        "quantized_images": sum(1 for v in quantize_status.values() if v.get("quantized")),
        "images": quantize_status,
    }

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)

    # 创建网格对比图
    print("\n创建网格对比图...")
    import math

    tiles_per_page = CONFIG["tiles_per_comparison"]
    total_pages = math.ceil(len(image_files) / tiles_per_page)

    comparison_files = []
    for page in range(total_pages):
        result = create_grid_comparison(
            image_files,
            output_dir,
            comparison_dir,
            page,
            tiles_per_page,
            CONFIG["comparison_cols"],
            CONFIG["comparison_rows"]
        )
        if result:
            comparison_files.append(result)
            print(f"  已生成: {result.name}")

    # 打印总结
    print("\n" + "=" * 60)
    print("量化完成!")
    print("=" * 60)
    print(f"原始图片: {input_dir}")
    print(f"量化图片: {output_dir}")
    print(f"对比图目录: {comparison_dir}")
    print(f"状态文件: {status_path}")
    print(f"对比图数量: {len(comparison_files)} 张")
    print("=" * 60)


def create_grid_comparison(image_files, quantized_dir, output_dir, page_num, tiles_per_page=50, cols=5, rows=10):
    """创建网格对比图：左边原图，右边量化图"""
    import math

    # 计算起始和结束索引
    start_idx = page_num * tiles_per_page
    end_idx = min(start_idx + tiles_per_page, len(image_files))
    selected_files = image_files[start_idx:end_idx]
    num_images = len(selected_files)

    if num_images == 0:
        return None

    # 实际行数
    actual_rows = math.ceil(num_images / cols)

    # 计算尺寸
    cell_size = 48
    padding = 4
    label_width = 60
    title_height = 40

    # 每个区域的宽度（左边原图，右边量化图）
    section_width = cols * (cell_size + padding) + padding

    # 总尺寸
    total_width = label_width + section_width * 2 + padding * 3
    total_height = actual_rows * (cell_size + padding) + padding + title_height

    # 创建图片
    grid_img = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid_img)

    # 标题
    draw.text((label_width + 10, 10), f"Original (Page {page_num + 1})", fill="white")
    draw.text((label_width + section_width + padding + 10, 10), f"Quantized ({CONFIG['palette_colors']} colors)", fill="white")

    # 绘制图片
    for i, img_path in enumerate(selected_files):
        row = i // cols
        col = i % cols

        # 原图位置
        x1 = label_width + padding + col * (cell_size + padding)
        y1 = title_height + row * (cell_size + padding)

        # 量化图位置
        x2 = label_width + section_width + padding * 2 + col * (cell_size + padding)
        y2 = y1

        try:
            # 原图
            original = Image.open(img_path).convert("RGBA")
            original_small = original.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            grid_img.paste(original_small, (x1, y1))

            # 量化图
            quantized_path = quantized_dir / img_path.name
            if quantized_path.exists():
                quantized = Image.open(quantized_path).convert("RGBA")
                quantized_small = quantized.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                grid_img.paste(quantized_small, (x2, y2))

            # 行号标签
            if col == 0:
                draw.text((10, y1 + cell_size // 2 - 8), f"R{row + 1}", fill="gray")

        except Exception as e:
            print(f"  [错误] 处理 {img_path.name}: {e}")

    # 保存
    output_path = output_dir / f"comparison_page_{page_num + 1:03d}.png"
    grid_img.save(output_path)
    return output_path


if __name__ == "__main__":
    main()

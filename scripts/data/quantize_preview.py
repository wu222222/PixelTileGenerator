"""
颜色量化预览脚本

功能:
- 随机选择5-10张图片进行量化
- 生成原图vs量化后的对比图
- 检查量化后的质量
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent,
    "input_dir": "datasets/classified/pixel_32",
    "output_dir": "datasets/classified/pixel_32_quantized",
    "preview_dir": "datasets/classified/quantize_preview",
    "target_colors": 32,
    "preview_count": 8,
}


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    """对图片进行颜色量化"""
    # 确保是RGBA或RGB模式
    if img.mode == "P":
        img = img.convert("RGBA")

    # RGBA图片需要使用FASTOCTREE方法
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    # 转换回RGBA
    return quantized.convert("RGBA")


def create_comparison(original: Image.Image, quantized: Image.Image, name: str) -> Image.Image:
    """创建对比图"""
    # 放大倍数（方便查看）
    scale = 4

    # 计算对比图尺寸
    width = original.width * scale * 2 + 20  # 中间留20像素间距
    height = original.height * scale + 40  # 上方留40像素写标题

    # 创建对比图
    comparison = Image.new("RGBA", (width, height), (40, 40, 40, 255))
    draw = ImageDraw.Draw(comparison)

    # 绘制标题
    draw.text((10, 10), f"原图 ({name})", fill="white")
    draw.text((original.width * scale + 30, 10), f"量化后 ({CONFIG['target_colors']}色)", fill="white")

    # 放大并粘贴原图
    original_scaled = original.resize(
        (original.width * scale, original.height * scale),
        Image.Resampling.NEAREST
    )
    comparison.paste(original_scaled, (10, 40))

    # 放大并粘贴量化后的图
    quantized_scaled = quantized.resize(
        (quantized.width * scale, quantized.height * scale),
        Image.Resampling.NEAREST
    )
    comparison.paste(quantized_scaled, (original.width * scale + 30, 40))

    return comparison


def preview_quantize():
    """预览量化效果"""
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    preview_dir = base_dir / CONFIG["preview_dir"]

    if not input_dir.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    # 创建预览目录
    preview_dir.mkdir(parents=True, exist_ok=True)

    # 图片扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    # 获取所有图片
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if len(image_files) == 0:
        print("[错误] 没有找到图片文件")
        return

    # 随机选择图片
    sample_count = min(CONFIG["preview_count"], len(image_files))
    sample_files = random.sample(image_files, sample_count)

    print(f"随机选择 {sample_count} 张图片进行预览")
    print("=" * 60)

    results = []

    for img_path in sample_files:
        try:
            # 打开原图
            with Image.open(img_path) as original:
                original_rgba = original.convert("RGBA")

                # 量化
                quantized = quantize_image(original_rgba, CONFIG["target_colors"])

                # 统计颜色数
                original_colors = len(original_rgba.getcolors(maxcolors=256) or [])
                quantized_colors = len(quantized.getcolors(maxcolors=256) or [])

                # 创建对比图
                comparison = create_comparison(original_rgba, quantized, img_path.name)

                # 保存对比图
                preview_path = preview_dir / f"preview_{img_path.name}"
                comparison.save(preview_path, "PNG")

                results.append({
                    "name": img_path.name,
                    "original_colors": original_colors,
                    "quantized_colors": quantized_colors,
                    "preview_path": str(preview_path),
                })

                print(f"  [完成] {img_path.name}")
                print(f"         原始颜色: {original_colors} → 量化后: {quantized_colors}")

        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")

    # 打印总结
    print("\n" + "=" * 60)
    print("预览完成!")
    print("=" * 60)
    print(f"预览图片保存在: {preview_dir}")
    print("\n请查看预览效果，确认后运行批量量化脚本")

    # 保存结果
    import json
    result_path = preview_dir / "preview_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {result_path}")


def main():
    print("=" * 60)
    print("颜色量化预览")
    print(f"目标颜色数: {CONFIG['target_colors']}")
    print("=" * 60)
    preview_quantize()


if __name__ == "__main__":
    main()

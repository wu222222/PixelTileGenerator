"""
批量颜色量化脚本

功能:
- 对pixel_32目录中所有图片进行颜色量化
- 量化到32色
- 保存到pixel_32_quantized目录
- 生成统计报告
"""

import json
from pathlib import Path
from PIL import Image


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "datasets/classified/pixel_64",
    "output_dir": "datasets/classified/pixel_64_quantized",
    "target_colors": 32,
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


def batch_quantize():
    """批量量化图片"""
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    output_dir = base_dir / CONFIG["output_dir"]

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图片扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    # 获取所有图片
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    total = len(image_files)

    print(f"找到 {total} 个图片")
    print(f"目标颜色数: {CONFIG['target_colors']}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 统计
    stats = {
        "total": total,
        "success": 0,
        "errors": 0,
        "color_distribution": {},
    }

    # 处理每个图片
    for i, img_path in enumerate(image_files, 1):
        try:
            # 打开原图
            with Image.open(img_path) as original:
                original_rgba = original.convert("RGBA")

                # 量化
                quantized = quantize_image(original_rgba, CONFIG["target_colors"])

                # 统计量化后的颜色数
                quantized_colors = len(quantized.getcolors(maxcolors=256) or [])
                stats["color_distribution"][quantized_colors] = stats["color_distribution"].get(quantized_colors, 0) + 1

                # 保存量化后的图片
                output_path = output_dir / img_path.name
                quantized.save(output_path, "PNG")

                stats["success"] += 1

                # 打印进度
                if i % 100 == 0 or i == total:
                    print(f"  进度: {i}/{total} ({i/total*100:.1f}%)")

        except Exception as e:
            stats["errors"] += 1
            print(f"  [错误] {img_path.name}: {e}")

    # 打印报告
    print("\n" + "=" * 60)
    print("批量量化完成!")
    print("=" * 60)
    print(f"总计: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"错误: {stats['errors']}")
    print(f"输出目录: {output_dir}")

    # 颜色数分布
    print(f"\n量化后颜色数分布:")
    for colors, count in sorted(stats["color_distribution"].items()):
        percentage = count / stats["success"] * 100 if stats["success"] > 0 else 0
        print(f"  {colors}色: {count} ({percentage:.1f}%)")

    print("=" * 60)

    # 保存报告
    report_path = output_dir / "quantize_report.json"
    report = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "target_colors": CONFIG["target_colors"],
        },
        "stats": stats,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_path}")


def main():
    print("=" * 60)
    print("批量颜色量化")
    print("=" * 60)
    batch_quantize()


if __name__ == "__main__":
    main()

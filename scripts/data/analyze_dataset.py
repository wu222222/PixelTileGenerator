"""
数据集统计分析脚本

功能:
- 统计pixel_32目录中的图片数量
- 分析文件名前缀（来源）
- 分析颜色数分布
- 生成统计报告
"""

import json
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent,
    "input_dir": "datasets/classified/pixel_32",
}


def analyze_dataset():
    """分析数据集"""
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]

    if not input_dir.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    # 图片扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    # 统计数据
    stats = {
        "total": 0,
        "sources": defaultdict(int),  # 来源统计
        "colors": [],  # 颜色数列表
        "sizes": Counter(),  # 尺寸统计
    }

    # 扫描所有图片
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    stats["total"] = len(image_files)

    print(f"扫描完成: 找到 {len(image_files)} 个图片")
    print("=" * 60)
    print("分析中...")

    for img_path in image_files:
        try:
            # 分析文件名来源
            # 文件名格式通常是: 来源_序号.png 或 来源_hash_序号.png
            stem = img_path.stem

            # 提取来源（第一个下划线前的部分）
            parts = stem.split("_")
            if len(parts) >= 2:
                # 尝试提取有意义的来源名
                source = parts[0]
                # 如果来源太短或太长，尝试合并前几个部分
                if len(source) < 3 and len(parts) > 2:
                    source = "_".join(parts[:2])
                stats["sources"][source] += 1
            else:
                stats["sources"]["unknown"] += 1

            # 分析图片属性
            with Image.open(img_path) as img:
                width, height = img.size
                stats["sizes"][f"{width}×{height}"] += 1

                # 计算颜色数
                try:
                    colors = len(img.getcolors(maxcolors=256))
                except TypeError:
                    colors = 257
                stats["colors"].append(colors)

        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")

    # 计算颜色数统计
    if stats["colors"]:
        avg_colors = sum(stats["colors"]) / len(stats["colors"])
        min_colors = min(stats["colors"])
        max_colors = max(stats["colors"])
    else:
        avg_colors = min_colors = max_colors = 0

    # 打印报告
    print("\n" + "=" * 60)
    print("数据集统计报告")
    print("=" * 60)

    print(f"\n📊 总计: {stats['total']} 张图片")

    print(f"\n📏 尺寸分布:")
    for size, count in stats["sizes"].most_common(10):
        percentage = count / stats["total"] * 100
        print(f"  {size}: {count} ({percentage:.1f}%)")

    print(f"\n🎨 颜色数统计:")
    print(f"  平均: {avg_colors:.1f}")
    print(f"  最小: {min_colors}")
    print(f"  最大: {max_colors}")

    # 颜色数分布
    color_ranges = {
        "1-16": 0,
        "17-32": 0,
        "33-64": 0,
        "65-128": 0,
        "129-256": 0,
        "257+": 0,
    }
    for c in stats["colors"]:
        if c <= 16:
            color_ranges["1-16"] += 1
        elif c <= 32:
            color_ranges["17-32"] += 1
        elif c <= 64:
            color_ranges["33-64"] += 1
        elif c <= 128:
            color_ranges["65-128"] += 1
        elif c <= 256:
            color_ranges["129-256"] += 1
        else:
            color_ranges["257+"] += 1

    print(f"\n  颜色数分布:")
    for range_name, count in color_ranges.items():
        percentage = count / stats["total"] * 100
        print(f"    {range_name}: {count} ({percentage:.1f}%)")

    print(f"\n📁 来源分布 (前20):")
    for source, count in sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True)[:20]:
        percentage = count / stats["total"] * 100
        print(f"  {source}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 60)

    # 保存详细报告
    report_path = base_dir / CONFIG["input_dir"] / "dataset_analysis.json"
    report = {
        "total": stats["total"],
        "sizes": dict(stats["sizes"]),
        "colors": {
            "average": avg_colors,
            "min": min_colors,
            "max": max_colors,
            "distribution": color_ranges,
        },
        "sources": dict(sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True)),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"详细报告已保存: {report_path}")


def main():
    print("=" * 60)
    print("数据集统计分析")
    print("=" * 60)
    analyze_dataset()


if __name__ == "__main__":
    main()

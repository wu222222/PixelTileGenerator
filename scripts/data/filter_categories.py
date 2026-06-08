"""
过滤类别脚本

功能:
- 加载类别标签
- 过滤出样本数 >= min_count 的类别
- 复制对应的图片到新目录
- 生成新的类别标签文件
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "data_dir": "datasets/classified/pixel_32_quantized",
    "labels_file": "datasets/classified/pixel_32_quantized/class_labels.json",
    "output_dir": "datasets/classified/pixel_32_filtered",
    "output_labels": "datasets/classified/pixel_32_filtered/class_labels.json",
    "min_count": 96,  # 最少96张
}


def main():
    print("=" * 60)
    print("过滤类别")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    data_dir = base_dir / CONFIG["data_dir"]
    labels_path = base_dir / CONFIG["labels_file"]
    output_dir = base_dir / CONFIG["output_dir"]
    output_labels_path = base_dir / CONFIG["output_labels"]

    if not labels_path.exists():
        print(f"[错误] 标签文件不存在: {labels_path}")
        return

    # 加载标签
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    categories = labels_data["categories"]
    print(f"总图片数: {len(categories)}")
    print(f"总类别数: {labels_data['total_categories']}")

    # 统计类别
    category_counts = defaultdict(int)
    for cat in categories.values():
        category_counts[cat] += 1

    # 过滤类别
    valid_categories = {cat for cat, count in category_counts.items() if count >= CONFIG["min_count"]}
    print(f"\n保留类别 (>= {CONFIG['min_count']} 张):")
    for cat in sorted(valid_categories):
        print(f"  {cat}: {category_counts[cat]}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片
    print(f"\n复制图片到 {output_dir}...")
    filtered_categories = {}
    copied_count = 0

    for img_name, category in categories.items():
        if category in valid_categories:
            src = data_dir / img_name
            dst = output_dir / img_name

            if src.exists():
                shutil.copy2(src, dst)
                filtered_categories[img_name] = category
                copied_count += 1

    # 统计过滤后的类别
    filtered_counts = defaultdict(int)
    for cat in filtered_categories.values():
        filtered_counts[cat] += 1

    # 保存新的标签文件
    output_data = {
        "categories": filtered_categories,
        "category_counts": dict(filtered_counts),
        "total_images": len(filtered_categories),
        "total_categories": len(filtered_counts),
    }

    with open(output_labels_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("过滤完成!")
    print("=" * 60)
    print(f"原始: {len(categories)} 张, {labels_data['total_categories']} 类")
    print(f"过滤后: {len(filtered_categories)} 张, {len(filtered_counts)} 类")
    print(f"\n类别分布:")
    for cat, count in sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    print(f"\n输出目录: {output_dir}")
    print(f"标签文件: {output_labels_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

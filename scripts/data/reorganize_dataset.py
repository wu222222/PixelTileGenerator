"""
重组数据集脚本

功能:
- 建立新的类别体系 (terrain/structure/cloth/discard)
- 合并相关类别到terrain
- 丢弃unknown_cluster_2
- 生成新的数据集和标签
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "data_dir": "datasets/classified/pixel_32_filtered",
    "labels_file": "datasets/classified/pixel_32_filtered/class_labels.json",
    "output_dir": "datasets/classified/pixel_32_v2",
    "output_labels": "datasets/classified/pixel_32_v2/class_labels.json",
}

# 新的类别映射
CATEGORY_MAPPING = {
    # terrain (地形纹理)
    "terrain": "terrain",
    "unknown_cluster_3": "terrain",
    "ground": "terrain",
    "stone": "terrain",
    "floor-dirt": "terrain",
    "floor-sand": "terrain",
    "floor-grass": "terrain",
    "floor-stone": "terrain",
    "elements": "terrain",

    # structure (结构纹理)
    "grating": "structure",
    "floor-metal": "structure",
    "floor-glass": "structure",
    "wall-brick": "structure",
    "brick": "structure",
    "metal": "structure",

    # fabric (布料)
    "cloth": "cloth",
    "weave": "cloth",

    # discard (丢弃)
    "unknown_cluster_2": "discard",
    "unknown_cluster_0": "discard",
    "unknown_cluster_1": "discard",
    "unknown_cluster_4": "discard",
}


def main():
    print("=" * 60)
    print("重组数据集")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    data_dir = base_dir / CONFIG["data_dir"]
    labels_path = base_dir / CONFIG["labels_file"]
    output_dir = base_dir / CONFIG["output_dir"]
    output_labels_path = base_dir / CONFIG["output_labels"]

    # 加载标签
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    categories = labels_data["categories"]
    print(f"原始数据: {len(categories)} 张")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["terrain", "structure", "cloth", "discard"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # 统计
    new_categories = {}
    category_counts = defaultdict(int)
    unmapped = defaultdict(int)

    # 处理每张图片
    for img_name, old_category in categories.items():
        # 映射到新类别
        new_category = CATEGORY_MAPPING.get(old_category)

        if new_category is None:
            # 未映射的类别
            unmapped[old_category] += 1
            new_category = "discard"  # 默认丢弃

        # 复制文件
        src = data_dir / img_name
        dst = output_dir / new_category / img_name

        if src.exists():
            shutil.copy2(src, dst)
            new_categories[img_name] = new_category
            category_counts[new_category] += 1

    # 保存新标签
    output_data = {
        "categories": new_categories,
        "category_counts": dict(category_counts),
        "total_images": len(new_categories),
        "total_categories": len(category_counts),
        "mapping": CATEGORY_MAPPING,
    }

    with open(output_labels_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 打印统计
    print("\n" + "=" * 60)
    print("重组完成!")
    print("=" * 60)
    print(f"新数据集: {len(new_categories)} 张")
    print(f"\n类别分布:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    if unmapped:
        print(f"\n未映射的类别 (已放入discard):")
        for cat, count in sorted(unmapped.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

    print(f"\n输出目录: {output_dir}")
    print(f"标签文件: {output_labels_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
抽样unknown_cluster图片

功能:
- 从unknown_cluster_2和unknown_cluster_3随机抽样20张
- 保存到指定目录供人工查看
"""

import json
import random
import shutil
from pathlib import Path


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "data_dir": "datasets/classified/pixel_32_filtered",
    "labels_file": "datasets/classified/pixel_32_filtered/class_labels.json",
    "output_dir": "datasets/classified/unknown_samples",
    "sample_count": 20,
}


def main():
    print("=" * 60)
    print("抽样unknown_cluster图片")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    data_dir = base_dir / CONFIG["data_dir"]
    labels_path = base_dir / CONFIG["labels_file"]
    output_dir = base_dir / CONFIG["output_dir"]

    # 加载标签
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    categories = labels_data["categories"]

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个unknown_cluster
    for cluster_name in ["unknown_cluster_2", "unknown_cluster_3"]:
        # 获取该类别的所有图片
        cluster_images = [name for name, cat in categories.items() if cat == cluster_name]

        print(f"\n{cluster_name}: {len(cluster_images)} 张")

        # 随机抽样
        sample_count = min(CONFIG["sample_count"], len(cluster_images))
        sampled = random.sample(cluster_images, sample_count)

        # 复制到输出目录
        cluster_dir = output_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for img_name in sampled:
            src = data_dir / img_name
            dst = cluster_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)

        print(f"  已抽样 {sample_count} 张到 {cluster_dir}")

    print("\n" + "=" * 60)
    print("抽样完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n请查看这些图片，判断里面的内容:")
    print("  - 如果是单一纹理 (如全是grass) → 可以用作训练")
    print("  - 如果是混合内容 → 需要重新分类或删除")


if __name__ == "__main__":
    main()

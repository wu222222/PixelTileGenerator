"""
提取grass类别图片

功能:
- 从terrain中提取grass相关图片
- 复制到单独的grass目录
- 统计数量
"""

import shutil
from pathlib import Path
from collections import defaultdict


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "terrain_dir": "datasets/classified/pixel_32_v2/terrain",
    "output_dir": "datasets/classified/pixel_32_v2/grass",
    "grass_keywords": ["grass", "green", "lawn", "meadow"],
}


def main():
    print("=" * 60)
    print("提取grass类别图片")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    terrain_dir = base_dir / CONFIG["terrain_dir"]
    output_dir = base_dir / CONFIG["output_dir"]

    if not terrain_dir.exists():
        print(f"[错误] terrain目录不存在: {terrain_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in terrain_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"terrain总图片数: {len(image_files)}")

    # 筛选grass相关图片
    grass_files = []
    other_files = []

    for img_path in image_files:
        filename_lower = img_path.name.lower()
        is_grass = any(keyword in filename_lower for keyword in CONFIG["grass_keywords"])

        if is_grass:
            grass_files.append(img_path)
        else:
            other_files.append(img_path)

    print(f"grass相关图片: {len(grass_files)}")
    print(f"其他图片: {len(other_files)}")

    # 复制grass图片
    print(f"\n复制到 {output_dir}...")
    for img_path in grass_files:
        dst = output_dir / img_path.name
        shutil.copy2(img_path, dst)

    # 打印总结
    print("\n" + "=" * 60)
    print("提取完成!")
    print("=" * 60)
    print(f"grass类别: {len(grass_files)} 张")
    print(f"输出目录: {output_dir}")

    # 检查是否足够
    if len(grass_files) < 100:
        print("\n[警告] grass图片太少，可能需要手动添加更多")
    elif len(grass_files) < 300:
        print("\n[提示] grass图片数量一般，可以尝试训练")
    else:
        print("\n[✓] grass图片数量充足，可以开始训练")

    print("=" * 60)


if __name__ == "__main__":
    main()

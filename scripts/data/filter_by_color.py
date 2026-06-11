"""
按颜色筛选图片

功能:
- 按绿色/棕色/灰色占比排序
- 导出候选图片供人工筛选
"""

import shutil
from pathlib import Path
from PIL import Image
import numpy as np


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "datasets/classified/pixel_32_v2/terrain",
    "output_base": "datasets/classified/candidates",
    "top_k": 500,
}


def calculate_green_ratio(img_array):
    """计算绿色占比"""
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    # 绿色条件: G > R+10 且 G > B+10
    green_mask = (g > r + 10) & (g > b + 10)
    return green_mask.mean()


def calculate_brown_ratio(img_array):
    """计算棕色/泥土占比"""
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    # 棕色条件: R > G > B, 且都不是极值
    brown_mask = (r > g) & (g > b) & (r > 50) & (r < 200)
    return brown_mask.mean()


def calculate_gray_ratio(img_array):
    """计算灰色/石头占比"""
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    # 灰色条件: R≈G≈B
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)

    gray_mask = (diff_rg < 30) & (diff_rb < 30) & (diff_gb < 30) & (r > 30) & (r < 200)
    return gray_mask.mean()


def main():
    print("=" * 60)
    print("按颜色筛选图片")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    output_base = base_dir / CONFIG["output_base"]

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"总图片数: {len(image_files)}")

    # 计算每种颜色的占比
    print("\n分析颜色...")
    green_scores = []
    brown_scores = []
    gray_scores = []

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            green_scores.append((img_path, calculate_green_ratio(arr)))
            brown_scores.append((img_path, calculate_brown_ratio(arr)))
            gray_scores.append((img_path, calculate_gray_ratio(arr)))

            if (i + 1) % 500 == 0:
                print(f"  进度: {i+1}/{len(image_files)}")

        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")

    # 排序
    green_scores.sort(key=lambda x: x[1], reverse=True)
    brown_scores.sort(key=lambda x: x[1], reverse=True)
    gray_scores.sort(key=lambda x: x[1], reverse=True)

    # 导出候选图片
    top_k = CONFIG["top_k"]

    for category, scores in [("grass", green_scores), ("dirt", brown_scores), ("stone", gray_scores)]:
        output_dir = output_base / category
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n导出 {category} 候选 (前{top_k}张):")

        exported = 0
        for rank, (img_path, score) in enumerate(scores[:top_k]):
            if score > 0.01:  # 至少1%的颜色符合
                new_name = f"{rank:04d}_{score:.3f}_{img_path.name}"
                shutil.copy2(img_path, output_dir / new_name)
                exported += 1

        print(f"  导出: {exported} 张")

        # 打印前5个
        print(f"  前5个:")
        for rank, (img_path, score) in enumerate(scores[:5]):
            print(f"    {rank+1}. {img_path.name} - {score:.3f}")

    # 打印总结
    print("\n" + "=" * 60)
    print("筛选完成!")
    print("=" * 60)
    print(f"输出目录: {output_base}")
    print(f"\n请人工检查每个目录:")
    print(f"  - grass/: 删除非草地图片")
    print(f"  - dirt/: 删除非泥土图片")
    print(f"  - stone/: 删除非石头图片")
    print(f"\n文件名格式: 排序_分数_原文件名")
    print(f"  - 排序越小，越可能是该类别")
    print(f"  - 分数越高，颜色特征越明显")
    print("=" * 60)


if __name__ == "__main__":
    main()

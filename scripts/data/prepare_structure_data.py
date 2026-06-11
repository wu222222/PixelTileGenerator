"""
准备Structure Generator训练数据

功能:
- 从量化后的图片提取Structure Map（索引图）
- 保存每张图的调色板
- 数据增强（翻转、旋转）
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "datasets/structure_data",
    "augment_factor": 4,  # 增强倍数（翻转+旋转）
}


def extract_palette_and_indices(img: Image.Image):
    """提取调色板和索引图"""
    # 转换为RGBA
    img_rgba = img.convert("RGBA")
    pixels = np.array(img_rgba)

    # 提取唯一颜色
    h, w, c = pixels.shape
    pixels_flat = pixels.reshape(-1, c)

    # 找到唯一颜色并创建映射
    unique_colors = []
    color_to_idx = {}
    indices = np.zeros(h * w, dtype=np.int32)

    for i, pixel in enumerate(pixels_flat):
        color_tuple = tuple(int(x) for x in pixel[:3])  # RGB，转换为Python int
        if color_tuple not in color_to_idx:
            color_to_idx[color_tuple] = len(unique_colors)
            unique_colors.append(list(color_tuple))
        indices[i] = color_to_idx[color_tuple]

    indices = indices.reshape(h, w)
    palette = unique_colors

    return indices, palette


def augment_data(indices: np.ndarray, palette: list):
    """数据增强：翻转和旋转"""
    augmented = []

    # 原始
    augmented.append((indices, palette))

    # 水平翻转
    augmented.append((np.flip(indices, axis=1).copy(), palette))

    # 垂直翻转
    augmented.append((np.flip(indices, axis=0).copy(), palette))

    # 90度旋转
    augmented.append((np.rot90(indices).copy(), palette))

    return augmented


def main():
    print("=" * 60)
    print("准备Structure Generator训练数据")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    output_dir = base_dir / CONFIG["output_dir"]

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure_maps").mkdir(parents=True, exist_ok=True)
    (output_dir / "palettes").mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    # 获取所有量化后的图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions and f.name != "quantize_status.json"]

    print(f"找到 {len(image_files)} 张图片")

    # 处理每张图片
    all_samples = []

    for img_path in image_files:
        try:
            # 加载图片
            img = Image.open(img_path).convert("RGBA")

            # 提取Structure Map和调色板
            indices, palette = extract_palette_and_indices(img)

            # 数据增强
            augmented = augment_data(indices, palette)

            # 保存增强后的数据
            for aug_idx, (aug_indices, aug_palette) in enumerate(augmented):
                sample_name = f"{img_path.stem}_aug{aug_idx}"

                # 保存Structure Map（索引图）
                structure_path = output_dir / "structure_maps" / f"{sample_name}.npy"
                np.save(structure_path, aug_indices)

                # 保存调色板
                palette_path = output_dir / "palettes" / f"{sample_name}.json"
                with open(palette_path, "w") as f:
                    json.dump({"palette": aug_palette, "num_colors": len(aug_palette)}, f)

                # 保存预览图（用调色板渲染）
                preview = Image.new("RGB", (32, 32))
                for y in range(32):
                    for x in range(32):
                        color_idx = aug_indices[y, x]
                        if color_idx < len(aug_palette):
                            preview.putpixel((x, y), tuple(aug_palette[color_idx]))
                preview_path = output_dir / "previews" / f"{sample_name}.png"
                preview_large = preview.resize((128, 128), Image.Resampling.NEAREST)
                preview_large.save(preview_path)

                all_samples.append({
                    "name": sample_name,
                    "original": img_path.name,
                    "augmentation": aug_idx,
                    "num_colors": len(aug_palette),
                    "structure_path": str(structure_path.relative_to(output_dir)).replace("\\", "/"),
                    "palette_path": str(palette_path.relative_to(output_dir)).replace("\\", "/"),
                    "preview_path": str(preview_path.relative_to(output_dir)).replace("\\", "/"),
                })

            print(f"  [完成] {img_path.name} → {len(augmented)} 个样本")

        except Exception as e:
            print(f"  [错误] {img_path.name}: {e}")

    # 保存数据集信息（转换Path为字符串）
    config_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}

    dataset_info = {
        "config": config_serializable,
        "total_original": len(image_files),
        "total_samples": len(all_samples),
        "samples": all_samples,
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print(f"原始图片: {len(image_files)} 张")
    print(f"增强后样本: {len(all_samples)} 个")
    print(f"输出目录: {output_dir}")
    print(f"数据集信息: {info_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

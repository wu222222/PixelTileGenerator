"""
数据集质量验证

检查:
1. 图片尺寸 (应为 32×32)
2. 颜色数量 (应 ≤ 32)
3. 感知哈希 (pHash) 查重
4. 近似重复检测 (汉明距离)
5. 损坏图片检测
6. 生成质量报告和重复对比图
"""

import sys
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent


def phash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
    """
    感知哈希 (pHash)
    基于 DCT 的感知哈希，对缩放、轻微颜色变化鲁棒
    """
    # 转灰度，缩放到 hash_size+1 × hash_size+1
    gray = img.convert('L').resize((hash_size + 1, hash_size + 1), Image.Resampling.LANCZOS)
    pixels = np.array(gray, dtype=np.float64)

    # 简化版: 用均值比较代替 DCT
    # 左上角 hash_size×hash_size 区域
    block = pixels[:hash_size, :hash_size]
    mean_val = block.mean()
    hash_bits = (block > mean_val).flatten()

    return hash_bits


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """计算两个哈希的汉明距离"""
    return int(np.sum(h1 != h2))


def ahash(img: Image.Image, hash_size: int = 8) -> str:
    """平均哈希 (aHash)，更简单快速"""
    gray = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(gray, dtype=np.float64)
    mean_val = pixels.mean()
    bits = (pixels > mean_val).flatten()
    return ''.join(['1' if b else '0' for b in bits])


def check_image(img_path: Path) -> dict:
    """检查单张图片质量"""
    result = {
        "path": str(img_path),
        "name": img_path.name,
        "valid": True,
        "errors": [],
        "width": 0,
        "height": 0,
        "mode": "",
        "unique_colors": 0,
        "file_size": 0,
    }

    try:
        img = Image.open(img_path)
        result["width"] = img.width
        result["height"] = img.height
        result["mode"] = img.mode
        result["file_size"] = img_path.stat().st_size

        # 尺寸检查
        if img.width != 32 or img.height != 32:
            result["errors"].append(f"尺寸异常: {img.width}×{img.height}")

        # 颜色数量
        img_rgba = img.convert("RGBA")
        pixels = np.array(img_rgba)
        unique = np.unique(pixels.reshape(-1, 4), axis=0)
        result["unique_colors"] = len(unique)
        if len(unique) > 32:
            result["errors"].append(f"颜色过多: {len(unique)}")

        # 完全透明检查
        if img.mode == "RGBA":
            alpha = pixels[:, :, 3]
            if (alpha == 0).all():
                result["errors"].append("完全透明")
            elif (alpha == 0).sum() > pixels.shape[0] * pixels.shape[1] * 0.5:
                result["errors"].append("超过50%透明像素")

        # 单色检查
        if len(unique) <= 1:
            result["errors"].append("纯色图片")

        if result["errors"]:
            result["valid"] = False

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"无法打开: {e}")

    return result


def find_duplicates(results: list, threshold: int = 5) -> list:
    """
    查找近似重复 (汉明距离 < threshold)
    """
    duplicates = []
    n = len(results)

    # 计算所有哈希
    hashes = []
    for r in results:
        try:
            img = Image.open(r["path"])
            h = phash(img)
            hashes.append(h)
        except:
            hashes.append(None)

    # 两两比较
    for i in range(n):
        if hashes[i] is None:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            dist = hamming_distance(hashes[i], hashes[j])
            if dist < threshold:
                duplicates.append({
                    "img1": results[i]["name"],
                    "img2": results[j]["name"],
                    "distance": dist,
                    "idx1": i,
                    "idx2": j,
                })

    return duplicates


def generate_report(results: list, duplicates: list, output_dir: Path):
    """生成质量报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    invalid = total - valid

    # 统计
    sizes = Counter()
    color_counts = []
    errors_all = []

    for r in results:
        sizes[f"{r['width']}×{r['height']}"] += 1
        color_counts.append(r["unique_colors"])
        errors_all.extend(r["errors"])

    # 报告文本
    report = []
    report.append("=" * 60)
    report.append("数据集质量报告")
    report.append("=" * 60)
    report.append(f"总图片数: {total}")
    report.append(f"有效: {valid}")
    report.append(f"无效: {invalid}")
    report.append(f"近似重复对: {len(duplicates)}")
    report.append("")
    report.append("--- 尺寸分布 ---")
    for size, count in sizes.most_common():
        report.append(f"  {size}: {count} 张")
    report.append("")
    report.append("--- 颜色数量分布 ---")
    cc = np.array(color_counts)
    report.append(f"  最少: {cc.min()}")
    report.append(f"  最多: {cc.max()}")
    report.append(f"  平均: {cc.mean():.1f}")
    report.append(f"  中位数: {np.median(cc):.0f}")
    report.append("")
    report.append("--- 错误统计 ---")
    error_counts = Counter(errors_all)
    for err, count in error_counts.most_common():
        report.append(f"  {err}: {count}")
    report.append("")

    if invalid > 0:
        report.append("--- 无效图片列表 ---")
        for r in results:
            if not r["valid"]:
                report.append(f"  {r['name']}: {', '.join(r['errors'])}")
        report.append("")

    if duplicates:
        report.append(f"--- 近似重复 (汉明距离 < 5) ---")
        for d in duplicates[:50]:  # 最多显示50对
            report.append(f"  {d['img1']} ↔ {d['img2']} (距离: {d['distance']})")
        if len(duplicates) > 50:
            report.append(f"  ... 还有 {len(duplicates) - 50} 对")
        report.append("")

    report_text = "\n".join(report)
    print(report_text)

    # 保存报告
    with open(output_dir / "quality_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 生成重复对比图
    if duplicates:
        generate_duplicate_grid(duplicates, results, output_dir, max_pairs=20)

    # 颜色分布直方图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(color_counts, bins=range(0, max(color_counts) + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Unique Colors")
    ax.set_ylabel("Count")
    ax.set_title(f"Color Distribution (n={total})")
    ax.axvline(x=32, color='r', linestyle='--', label='32 color limit')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "color_distribution.png", dpi=150)
    plt.close()

    print(f"\n报告已保存: {output_dir / 'quality_report.txt'}")
    print(f"颜色分布图: {output_dir / 'color_distribution.png'}")
    if duplicates:
        print(f"重复对比图: {output_dir / 'duplicate_grid.png'}")


def generate_duplicate_grid(duplicates: list, results: list, output_dir: Path, max_pairs: int = 20):
    """生成重复对比图"""
    pairs = duplicates[:max_pairs]
    n = len(pairs)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
    if rows == 1 and cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif rows == 1:
        axes = axes.reshape(2, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, pair in enumerate(pairs):
        r = idx // cols
        c = idx % cols

        try:
            img1 = Image.open(results[pair["idx1"]]["path"]).convert("RGBA")
            img2 = Image.open(results[pair["idx2"]]["path"]).convert("RGBA")

            # 放大显示
            img1_large = img1.resize((128, 128), Image.Resampling.NEAREST)
            img2_large = img2.resize((128, 128), Image.Resampling.NEAREST)

            axes[r * 2, c].imshow(np.array(img1_large))
            axes[r * 2, c].set_title(f"{pair['img1']}\n(colors:{results[pair['idx1']]['unique_colors']})", fontsize=7)
            axes[r * 2, c].axis('off')

            axes[r * 2 + 1, c].imshow(np.array(img2_large))
            axes[r * 2 + 1, c].set_title(f"{pair['img2']}\n(dist:{pair['distance']}, colors:{results[pair['idx2']]['unique_colors']})", fontsize=7)
            axes[r * 2 + 1, c].axis('off')
        except Exception as e:
            axes[r * 2, c].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            axes[r * 2, c].axis('off')
            axes[r * 2 + 1, c].axis('off')

    # 隐藏多余子图
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r * 2, c].axis('off')
        axes[r * 2 + 1, c].axis('off')

    plt.suptitle(f"Duplicate Pairs (pHash distance < 5)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "duplicate_grid.png", dpi=150)
    plt.close()


def main():
    data_dir = project_root / "datasets" / "classified" / "pixel_32_quantized"
    output_dir = project_root / "checkpoints" / "dataset_quality"

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    print(f"扫描 {len(image_files)} 张图片...")
    print()

    # 1. 逐张检查
    results = []
    for i, f in enumerate(image_files):
        r = check_image(f)
        results.append(r)
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(image_files)}")

    # 2. 查找重复
    print("\n查找近似重复 (pHash)...")
    duplicates = find_duplicates(results, threshold=5)
    print(f"找到 {len(duplicates)} 对近似重复")

    # 3. 生成报告
    print()
    generate_report(results, duplicates, output_dir)


if __name__ == "__main__":
    main()

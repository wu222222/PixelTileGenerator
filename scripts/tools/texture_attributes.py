"""
Texture Attribute Space 挖掘

为每个样本计算连续属性:
1. Structure Score: 边缘能量 (规则结构 vs 随机纹理)
2. Entropy Score: 纹理随机性
3. Edge Density: 边缘密度
4. Color Variance: 颜色方差
5. Brightness: 平均亮度
6. Periodicity: 周期性强度 (FFT分析)

然后基于属性重新聚类，替代 tileset 标签
"""

import sys
import json
import numpy as np
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "checkpoints" / "v7_attribute_space"


def compute_attributes(img_array):
    """
    计算单张图片的纹理属性

    Args:
        img_array: (H, W, 4) RGBA uint8

    Returns:
        dict of attributes
    """
    rgb = img_array[:, :, :3].astype(float)
    gray = rgb.mean(axis=2)

    # 1. Structure Score: Sobel 边缘能量
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_energy = float(gx.mean() + gy.mean())

    # 2. Edge Density: 边缘像素占比 (阈值化)
    edge_mag = np.sqrt(
        np.pad(gx, ((0, 0), (0, 1))) ** 2 +
        np.pad(gy, ((0, 1), (0, 0))) ** 2
    )
    edge_threshold = edge_mag.mean() + edge_mag.std()
    edge_density = float((edge_mag > edge_threshold).mean())

    # 3. Entropy: 灰度直方图熵
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # 4. Color Variance: RGB 各通道方差的平均
    color_var = float(np.mean([rgb[:, :, c].var() for c in range(3)]))

    # 5. Brightness: 平均亮度
    brightness = float(gray.mean())

    # 6. Periodicity: FFT 周期性
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    # 中心区域 vs 外圈
    inner = magnitude[r < min(h, w) // 4].mean()
    outer = magnitude[r >= min(h, w) // 4].mean()
    periodicity = float(inner / (outer + 1e-8))

    # 7. Local Contrast: 局部对比度
    local_std = float(np.std(gray))

    # 8. Color Count: 唯一颜色数
    unique_colors = len(np.unique(img_array.reshape(-1, 4), axis=0))

    return {
        "structure": edge_energy,
        "edge_density": edge_density,
        "entropy": entropy,
        "color_variance": color_var,
        "brightness": brightness,
        "periodicity": periodicity,
        "local_contrast": local_std,
        "color_count": unique_colors,
    }


def main():
    print("=" * 60)
    print("Texture Attribute Space 挖掘")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载图片
    data_dir = project_root / "datasets" / "classified" / "pixel_32_quantized"
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    print(f"加载 {len(image_files)} 张图片")

    # 加载原始标签
    latent_dir = project_root / "datasets" / "vqvae_latent_data_v7"
    latent_names = json.load(open(latent_dir / "names.json"))
    latents = np.load(latent_dir / "latents.npy")

    def get_category(name):
        stem = Path(name).stem
        if "_from_" in stem:
            return stem.split("_from_")[0]
        return stem.split("_")[0]

    # 计算属性
    print("计算纹理属性...")
    all_attrs = []
    all_names = []
    for i, img_path in enumerate(image_files):
        img = np.array(Image.open(img_path).convert("RGBA"))
        attrs = compute_attributes(img)
        all_attrs.append(attrs)
        all_names.append(img_path.name)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(image_files)}")

    # 转为矩阵
    attr_keys = list(all_attrs[0].keys())
    attr_matrix = np.array([[a[k] for k in attr_keys] for a in all_attrs])
    print(f"属性矩阵: {attr_matrix.shape} ({len(attr_keys)} 属性)")

    # 标准化
    scaler = StandardScaler()
    attr_scaled = scaler.fit_transform(attr_matrix)

    # 基于属性的 KMeans 聚类
    print("\n基于属性空间的 KMeans 聚类 (k=8)...")
    kmeans_attr = KMeans(n_clusters=8, random_state=42, n_init=10)
    attr_labels = kmeans_attr.fit_predict(attr_scaled)

    # 原始标签
    orig_labels = [get_category(n) for n in all_names]

    # 对比: 原始标签 vs 属性聚类
    print("\n" + "=" * 80)
    print("属性聚类 vs 原始标签 交叉表")
    print("=" * 80)

    from collections import defaultdict
    cross_table = defaultdict(lambda: defaultdict(int))
    for i in range(len(all_names)):
        cross_table[attr_labels[i]][orig_labels[i]] += 1

    # 打印交叉表
    all_orig_cats = sorted(set(orig_labels))
    header = f"{'Cluster':>8} " + " ".join(f"{c:>10}" for c in all_orig_cats) + f" {'Total':>6}"
    print(header)
    print("-" * len(header))

    for c in range(8):
        row = f"{'C' + str(c):>8} "
        total = 0
        for cat in all_orig_cats:
            count = cross_table[c][cat]
            total += count
            row += f"{count:>10} "
        row += f"{total:>6}"
        print(row)

    # 每个属性聚类簇的特征
    print("\n" + "=" * 80)
    print("属性聚类簇特征")
    print("=" * 80)

    cluster_profiles = []
    for c in range(8):
        mask = attr_labels == c
        profile = {}
        for j, key in enumerate(attr_keys):
            vals = attr_matrix[mask, j]
            profile[key] = float(vals.mean())
        # 主要原始标签
        cats = defaultdict(int)
        for i in np.where(mask)[0]:
            cats[orig_labels[i]] += 1
        top_cats = sorted(cats.items(), key=lambda x: -x[1])
        profile["primary_label"] = top_cats[0][0] if top_cats else "?"
        profile["size"] = int(mask.sum())
        profile["id"] = c
        cluster_profiles.append(profile)

    for p in cluster_profiles:
        print(f"\nC{p['id']} ({p['size']}张, 原始标签: {p['primary_label']}):")
        for key in attr_keys:
            print(f"  {key:16s}: {p[key]:.3f}")

    # 给每个簇命名 (基于属性)
    print("\n" + "=" * 80)
    print("属性驱动的语义命名")
    print("=" * 80)

    def name_cluster(profile):
        """根据属性值给簇起语义名"""
        s = profile["structure"]
        e = profile["entropy"]
        b = profile["brightness"]
        p = profile["periodicity"]

        # 规则性
        if s > 25 and p > 2.5:
            base = "Brick"
        elif s > 20:
            base = "Stone"
        elif e > 4.5 and b > 120:
            base = "Light-Sand"
        elif e > 4.5 and b < 100:
            base = "Dark-Mud"
        elif b > 150:
            base = "Bright"
        elif b < 80:
            base = "Dark"
        else:
            base = "Mixed"

        # 颜色倾向
        if profile["color_variance"] > 3000:
            base += "-Colorful"
        elif profile["color_variance"] < 1000:
            base += "-Muted"

        return base

    for p in cluster_profiles:
        semantic_name = name_cluster(p)
        print(f"  C{p['id']:2d} → {semantic_name:20s} (原: {p['primary_label']}, n={p['size']})")

    # 可视化: 属性空间的 PCA 2D
    print("\n生成可视化...")
    pca2d = PCA(n_components=2)
    attr_2d = pca2d.fit_transform(attr_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 属性聚类着色
    scatter = axes[0, 0].scatter(attr_2d[:, 0], attr_2d[:, 1], c=attr_labels, cmap='tab10', s=8, alpha=0.6)
    axes[0, 0].set_title("Attribute Clusters")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")

    # 2. 原始标签着色
    cat_to_int = {cat: i for i, cat in enumerate(all_orig_cats)}
    orig_ints = [cat_to_int[c] for c in orig_labels]
    axes[0, 1].scatter(attr_2d[:, 0], attr_2d[:, 1], c=orig_ints, cmap='tab10', s=8, alpha=0.6)
    axes[0, 1].set_title("Original Labels (tileset)")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    # 3-8. 各属性着色
    attr_display = [
        ("structure", "Structure"),
        ("entropy", "Entropy"),
        ("brightness", "Brightness"),
        ("periodicity", "Periodicity"),
        ("color_variance", "Color Variance"),
        ("local_contrast", "Local Contrast"),
    ]
    for idx, (key, title) in enumerate(attr_display):
        row = (idx + 2) // 3
        col = (idx + 2) % 3
        if row < 2:
            j = attr_keys.index(key)
            sc = axes[row, col].scatter(attr_2d[:, 0], attr_2d[:, 1],
                                         c=attr_matrix[:, j], cmap='viridis', s=8, alpha=0.6)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("PC1")
            axes[row, col].set_ylabel("PC2")
            plt.colorbar(sc, ax=axes[row, col])

    plt.suptitle("Texture Attribute Space — PixelTileGAN v7", fontsize=14)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "attribute_space.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"属性空间图: {save_path}")

    # 保存属性数据
    attr_data = {
        "names": all_names,
        "attributes": {key: attr_matrix[:, i].tolist() for i, key in enumerate(attr_keys)},
        "attr_labels": attr_labels.tolist(),
        "cluster_profiles": cluster_profiles,
    }
    with open(OUTPUT_DIR / "attributes.json", "w") as f:
        json.dump(attr_data, f, indent=2)

    # 保存标准化器参数
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": attr_keys,
    }
    with open(OUTPUT_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"属性数据: {OUTPUT_DIR / 'attributes.json'}")

    # 对比属性聚类 vs 原始标签的纯度
    print("\n" + "=" * 60)
    print("聚类纯度分析")
    print("=" * 60)
    for c in range(8):
        mask = attr_labels == c
        cats = defaultdict(int)
        for i in np.where(mask)[0]:
            cats[orig_labels[i]] += 1
        top = sorted(cats.items(), key=lambda x: -x[1])
        purity = top[0][1] / mask.sum() if mask.sum() > 0 else 0
        cat_str = ", ".join(f"{k}:{v}" for k, v in top[:3])
        print(f"  C{c}: purity={purity:.1%}  ({cat_str})")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

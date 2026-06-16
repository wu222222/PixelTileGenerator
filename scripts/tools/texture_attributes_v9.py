"""
Texture Attribute Space 挖掘 (v9 适配)

为 64x64 像素画瓦片计算纹理属性:
1. Structure Score: 边缘能量
2. Entropy Score: 纹理随机性
3. Edge Density: 边缘密度
4. Color Variance: 颜色方差
5. Brightness: 平均亮度
6. Periodicity: 周期性强度 (FFT)

然后基于属性聚类 + 保存标准化参数 (供 attribute_generator 使用)
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

OUTPUT_DIR = project_root / "checkpoints" / "v9_attribute_space"

STRUCTURE_TYPES = ["flat", "grass", "ice", "rock"]


def name_structure_types(profiles):
    """基于排名 + 色相的结构类型命名 (4类: flat/grass/ice/rock)"""
    n = len(profiles)
    names = [""] * n

    def rank(key, reverse=False):
        vals = [(i, profiles[i][key]) for i in range(n)]
        vals.sort(key=lambda x: x[1], reverse=reverse)
        return {i: r for r, (i, _) in enumerate(vals)}

    r_struct = rank("structure", reverse=True)

    for i in range(n):
        if profiles[i]["green_ratio"] > 0.5:
            names[i] = "grass"
            break

    flat_score = [(i, (n - 1 - r_struct[i])) for i in range(n) if not names[i]]
    if flat_score:
        flat_score.sort(key=lambda x: x[1])
        names[flat_score[0][0]] = "flat"

    remaining = [i for i in range(n) if not names[i]]
    if len(remaining) >= 2:
        a, b = remaining[0], remaining[1]
        if profiles[a]["hue_dominant"] > profiles[b]["hue_dominant"]:
            names[a] = "ice"
            names[b] = "rock"
        else:
            names[b] = "ice"
            names[a] = "rock"
    elif len(remaining) == 1:
        names[remaining[0]] = "rock"

    return names


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

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_energy = float(gx.mean() + gy.mean())

    edge_mag = np.sqrt(
        np.pad(gx, ((0, 0), (0, 1))) ** 2 +
        np.pad(gy, ((0, 1), (0, 0))) ** 2
    )
    edge_threshold = edge_mag.mean() + edge_mag.std()
    edge_density = float((edge_mag > edge_threshold).mean())

    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    color_var = float(np.mean([rgb[:, :, c].var() for c in range(3)]))
    brightness = float(gray.mean())

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    inner = magnitude[r < min(h, w) // 4].mean()
    outer = magnitude[r >= min(h, w) // 4].mean()
    periodicity = float(inner / (outer + 1e-8))

    local_std = float(np.std(gray))
    unique_colors = len(np.unique(img_array.reshape(-1, 4), axis=0))

    r, g, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b_ch)
    min_c = np.minimum(np.minimum(r, g), b_ch)
    delta = max_c - min_c + 1e-8

    hue = np.zeros_like(r)
    mask_r = max_c == r
    mask_g = max_c == g
    mask_b = max_c == b_ch
    hue[mask_r] = 60 * (((g[mask_r] - b_ch[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * ((b_ch[mask_g] - r[mask_g]) / delta[mask_g] + 2)
    hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

    sat = np.where(max_c > 0, delta / (max_c + 1e-8), 0)

    non_gray = delta > 10
    hue_mean = float(hue[non_gray].mean()) if non_gray.sum() > 0 else 0.0

    if non_gray.sum() > 10:
        hue_hist, _ = np.histogram(hue[non_gray], bins=36, range=(0, 360))
        hue_dominant = float((np.argmax(hue_hist) * 10 + 5))
    else:
        hue_dominant = 0.0

    sat_mean = float(sat.mean())
    green_ratio = float(((hue > 60) & (hue < 180) & non_gray).mean())
    warm_ratio = float(((hue < 60) | (hue > 300)).mean())

    return {
        "structure": edge_energy,
        "edge_density": edge_density,
        "entropy": entropy,
        "color_variance": color_var,
        "brightness": brightness,
        "periodicity": periodicity,
        "local_contrast": local_std,
        "color_count": unique_colors,
        "hue_mean": hue_mean,
        "hue_dominant": hue_dominant,
        "saturation_mean": sat_mean,
        "green_ratio": green_ratio,
        "warm_ratio": warm_ratio,
    }


def main():
    print("=" * 60)
    print("Texture Attribute Space 挖掘 (v9)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载图片 (64x64)
    data_dir = project_root / "datasets" / "classified" / "pixel_64_quantized"
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    print(f"加载 {len(image_files)} 张图片")

    # 加载原始标签
    latent_dir = project_root / "datasets" / "vqvae_v9_zq_data"
    latent_names = json.load(open(latent_dir / "names.json"))

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
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(image_files)}")

    attr_keys = list(all_attrs[0].keys())
    attr_matrix = np.array([[a[k] for k in attr_keys] for a in all_attrs])
    print(f"属性矩阵: {attr_matrix.shape} ({len(attr_keys)} 属性)")

    # 标准化
    scaler = StandardScaler()
    attr_scaled = scaler.fit_transform(attr_matrix)

    # 原始标签
    orig_labels = [get_category(n) for n in all_names]
    all_orig_cats = sorted(set(orig_labels))

    # KMeans 聚类 (k=8)
    print("\n基于属性空间的 KMeans 聚类 (k=8)...")
    kmeans_attr = KMeans(n_clusters=8, random_state=42, n_init=10)
    attr_labels = kmeans_attr.fit_predict(attr_scaled)

    # 交叉表
    print("\n" + "=" * 80)
    print("属性聚类 vs 原始标签 交叉表")
    print("=" * 80)

    from collections import defaultdict
    cross_table = defaultdict(lambda: defaultdict(int))
    for i in range(len(all_names)):
        cross_table[attr_labels[i]][orig_labels[i]] += 1

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

    # 每个簇的特征
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

    # 语义命名
    print("\n" + "=" * 80)
    print("属性驱动的语义命名")
    print("=" * 80)

    def name_cluster(profile):
        s = profile["structure"]
        e = profile["entropy"]
        b = profile["brightness"]
        p = profile["periodicity"]
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
        if profile["color_variance"] > 3000:
            base += "-Colorful"
        elif profile["color_variance"] < 1000:
            base += "-Muted"
        return base

    for p in cluster_profiles:
        semantic_name = name_cluster(p)
        print(f"  C{p['id']:2d} → {semantic_name:20s} (原: {p['primary_label']}, n={p['size']})")

    # 可视化
    print("\n生成可视化...")
    pca2d = PCA(n_components=2)
    attr_2d = pca2d.fit_transform(attr_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    scatter = axes[0, 0].scatter(attr_2d[:, 0], attr_2d[:, 1], c=attr_labels, cmap='tab10', s=8, alpha=0.6)
    axes[0, 0].set_title("Attribute Clusters")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")

    cat_to_int = {cat: i for i, cat in enumerate(all_orig_cats)}
    orig_ints = [cat_to_int[c] for c in orig_labels]
    axes[0, 1].scatter(attr_2d[:, 0], attr_2d[:, 1], c=orig_ints, cmap='tab10', s=8, alpha=0.6)
    axes[0, 1].set_title("Original Labels (tileset)")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

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

    plt.suptitle("Texture Attribute Space — PixelTileGAN v9 (64x64)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attribute_space.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"属性空间图: {OUTPUT_DIR / 'attribute_space.png'}")

    # 保存属性数据
    attr_data = {
        "names": all_names,
        "attributes": {key: attr_matrix[:, i].tolist() for i, key in enumerate(attr_keys)},
        "attr_labels": attr_labels.tolist(),
        "cluster_profiles": cluster_profiles,
    }
    with open(OUTPUT_DIR / "attributes.json", "w") as f:
        json.dump(attr_data, f, indent=2)

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": attr_keys,
    }
    with open(OUTPUT_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"属性数据: {OUTPUT_DIR / 'attributes.json'}")

    # 结构类型聚类 (k=4)
    print("\n" + "=" * 60)
    print("结构类型聚类 (k=4)")
    print("=" * 60)
    kmeans_struct = KMeans(n_clusters=4, random_state=42, n_init=10)
    struct_labels = kmeans_struct.fit_predict(attr_scaled)

    struct_profiles_list = []
    for c in range(4):
        mask = struct_labels == c
        profile = {}
        for j, key in enumerate(attr_keys):
            profile[key] = float(attr_matrix[mask, j].mean())
        profile["size"] = int(mask.sum())
        struct_profiles_list.append(profile)

    type_names = name_structure_types(struct_profiles_list)

    struct_profiles = {}
    struct_names = {}
    for c in range(4):
        stype = type_names[c]
        struct_names[c] = stype
        struct_profiles_list[c]["type"] = stype
        struct_profiles[stype] = struct_profiles_list[c]
        p = struct_profiles_list[c]
        print(f"  C{c} → {stype:12s} (n={p['size']}, struct={p['structure']:.1f}, bright={p['brightness']:.1f}, entropy={p['entropy']:.1f})")

    structure_labels = {}
    for i, name in enumerate(all_names):
        structure_labels[name] = struct_names[struct_labels[i]]

    color_attrs = {}
    hue_idx = attr_keys.index("hue_dominant")
    sat_idx = attr_keys.index("saturation_mean")
    for i, name in enumerate(all_names):
        color_attrs[name] = {
            "hue_mean": float(attr_matrix[i, attr_keys.index("hue_mean")]),
            "hue_dominant": float(attr_matrix[i, hue_idx]),
            "saturation_mean": float(attr_matrix[i, sat_idx]),
            "green_ratio": float(attr_matrix[i, attr_keys.index("green_ratio")]),
            "warm_ratio": float(attr_matrix[i, attr_keys.index("warm_ratio")]),
        }

    labels_data = {
        "labels": structure_labels,
        "cluster_profiles": struct_profiles,
        "structure_types": list(struct_profiles.keys()),
        "color_attributes": color_attrs,
    }
    labels_path = OUTPUT_DIR / "structure_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
    print(f"\n结构标签: {labels_path}")

    # 结构类型可视化
    print("生成结构类型可视化...")
    n_types = len(struct_profiles)
    fig, axes = plt.subplots(n_types, 9, figsize=(18, n_types * 2.2))
    if n_types == 1:
        axes = axes.reshape(1, -1)

    img_cache = {}
    for img_path in image_files:
        img_cache[img_path.name] = np.array(Image.open(img_path).convert("RGBA"))

    for row, (stype, profile) in enumerate(struct_profiles.items()):
        type_indices = [i for i, n in enumerate(all_names) if structure_labels[n] == stype]
        type_attrs = attr_scaled[type_indices]
        center = type_attrs.mean(axis=0)
        dists = np.linalg.norm(type_attrs - center, axis=1)
        sorted_local = np.argsort(dists)[:8]

        axes[row, 0].set_ylabel(stype, fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center')
        for col in range(9):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < 8:
                idx = type_indices[sorted_local[col]]
                img = img_cache[all_names[idx]]
                ax.imshow(img, interpolation='nearest')
                if col == 0:
                    ax.set_title(f"n={profile['size']}", fontsize=9)
            else:
                info = f"struct:{profile['structure']:.1f}\nentropy:{profile['entropy']:.1f}\nbright:{profile['brightness']:.1f}\nperiod:{profile['periodicity']:.1f}"
                ax.text(0.5, 0.5, info, transform=ax.transAxes, fontsize=9,
                        ha='center', va='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))
                ax.set_facecolor('#1a1a1a')

    plt.suptitle("Structure Types — PixelTileGAN v9 (64x64)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "structure_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"结构类型图: {OUTPUT_DIR / 'structure_overview.png'}")

    # 结构×颜色 散点图
    print("生成 结构×颜色 2D 散点图...")
    struct_score = attr_matrix[:, attr_keys.index("structure")] + attr_matrix[:, attr_keys.index("periodicity")]
    color_score = attr_matrix[:, hue_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for stype, profile in struct_profiles.items():
        type_mask = np.array([structure_labels[n] == stype for n in all_names])
        ax.scatter(struct_score[type_mask], color_score[type_mask],
                   label=f"{stype} (n={profile['size']})", s=20, alpha=0.6)
    ax.set_xlabel("Structure Score (structure + periodicity)")
    ax.set_ylabel("Hue Dominant (color)")
    ax.set_title("Structure × Color — PixelTileGAN v9 (64x64)")
    ax.legend()
    plt.savefig(OUTPUT_DIR / "structure_color_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"散点图: {OUTPUT_DIR / 'structure_color_scatter.png'}")

    # 聚类纯度
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

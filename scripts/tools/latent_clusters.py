"""
Latent Space 聚类分析

1. KMeans 聚类 (n=8) → 每个簇的纹理拼图
2. 最近邻可视化 → latent 相似但文件夹不同的样本
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "checkpoints" / "v7_cluster_analysis"


def load_data():
    latent_dir = project_root / "datasets" / "vqvae_latent_data_v7"
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))
    return latents, names


def load_images(names):
    """加载所有原图"""
    data_dir = project_root / "datasets" / "classified" / "pixel_32_quantized"
    images = {}
    for name in names:
        path = data_dir / name
        if path.exists():
            images[name] = Image.open(path).convert("RGBA")
    return images


def get_category(name):
    stem = Path(name).stem
    if "_from_" in stem:
        return stem.split("_from_")[0]
    return stem.split("_")[0]


def kmeans_analysis(latents, names, images, n_clusters=8):
    """KMeans 聚类 + 簇拼图"""
    print("\n" + "=" * 60)
    print(f"KMeans 聚类 (k={n_clusters})")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flat = latents.reshape(latents.shape[0], -1)

    # PCA 降到 50 维再聚类
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_pca)

    # 统计每个簇
    print("\n簇分布:")
    for c in range(n_clusters):
        mask = labels == c
        cats = [get_category(names[i]) for i in np.where(mask)[0]]
        cat_counts = {}
        for cat in cats:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]))
        print(f"  Cluster {c}: {mask.sum():4d} 张  ({cat_str})")

    # 每个簇选 16 张最靠近中心的样本，拼成 4×4 网格
    print("\n生成簇拼图...")
    grid_size = 4
    tile_size = 64
    gap = 4

    for c in range(n_clusters):
        mask = labels == c
        indices = np.where(mask)[0]

        # 计算到簇中心的距离
        center = kmeans.cluster_centers_[c]
        dists = np.linalg.norm(flat_pca[indices] - center, axis=1)
        sorted_idx = indices[np.argsort(dists)]

        # 拼图
        n_show = min(grid_size * grid_size, len(sorted_idx))
        canvas = Image.new("RGBA",
            (grid_size * tile_size + (grid_size - 1) * gap,
             grid_size * tile_size + (grid_size - 1) * gap),
            (40, 40, 40, 255))

        for i in range(n_show):
            idx = sorted_idx[i]
            name = names[idx]
            if name in images:
                img = images[name].resize((tile_size, tile_size), Image.Resampling.NEAREST)
                row, col = divmod(i, grid_size)
                x = col * (tile_size + gap)
                y = row * (tile_size + gap)
                canvas.paste(img, (x, y))

        # 保存
        canvas_large = canvas.resize(
            (canvas.width * 2, canvas.height * 2),
            Image.Resampling.NEAREST)
        save_path = OUTPUT_DIR / f"cluster_{c}.png"
        canvas_large.save(save_path)

    # 大拼图: 所有簇并排
    print("生成总览拼图...")
    overview_w = n_clusters * (grid_size * tile_size + (grid_size - 1) * gap) + (n_clusters - 1) * gap
    overview_h = grid_size * tile_size + (grid_size - 1) * gap + 30  # +30 for label
    overview = Image.new("RGBA", (overview_w, overview_h), (30, 30, 30, 255))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(overview)

    for c in range(n_clusters):
        cluster_img = Image.open(OUTPUT_DIR / f"cluster_{c}.png")
        # 缩放到目标大小
        target_w = grid_size * tile_size + (grid_size - 1) * gap
        target_h = grid_size * tile_size + (grid_size - 1) * gap
        cluster_img = cluster_img.resize((target_w, target_h), Image.Resampling.NEAREST)

        x_offset = c * (target_w + gap)
        overview.paste(cluster_img, (x_offset, 30))

        # 标签
        n_in_cluster = (labels == c).sum()
        draw.text((x_offset + 2, 5), f"Cluster {c} ({n_in_cluster})", fill=(200, 200, 200))

    overview_large = overview.resize((overview.width * 2, overview.height * 2), Image.Resampling.NEAREST)
    save_path = OUTPUT_DIR / "clusters_overview.png"
    overview_large.save(save_path)
    print(f"总览: {save_path}")

    return labels


def nearest_neighbor_analysis(latents, names, images, n_neighbors=10):
    """最近邻可视化"""
    print("\n" + "=" * 60)
    print("最近邻分析")
    print("=" * 60)

    flat = latents.reshape(latents.shape[0], -1)

    # 拟合最近邻模型
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(flat)

    # 选几个样本
    samples = [0, 100, 300, 500, 700, 900]
    samples = [s for s in samples if s < len(names)]

    tile_size = 64
    gap = 4

    for query_idx in samples:
        query_name = names[query_idx]
        query_cat = get_category(query_name)

        distances, indices = nn.kneighbors(flat[query_idx:query_idx+1])
        neighbor_indices = indices[0][1:]  # 排除自身
        neighbor_dists = distances[0][1:]

        # 拼图: query + neighbors
        n_total = 1 + n_neighbors
        canvas = Image.new("RGBA",
            (n_total * tile_size + (n_total - 1) * gap, tile_size + 25),
            (40, 40, 40, 255))

        draw = ImageDraw.Draw(canvas)

        # Query
        if query_name in images:
            img = images[query_name].resize((tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(img, (0, 0))
        draw.text((2, tile_size + 3), f"QUERY", fill=(255, 200, 100))

        # Neighbors
        for i, (nidx, dist) in enumerate(zip(neighbor_indices, neighbor_dists)):
            nname = names[nidx]
            ncat = get_category(nname)
            if nname in images:
                img = images[nname].resize((tile_size, tile_size), Image.Resampling.NEAREST)
                x = (i + 1) * (tile_size + gap)
                canvas.paste(img, (x, 0))

                # 标注: 绿色=同类别, 红色=不同类别
                same_cat = ncat == query_cat
                color = (100, 255, 100) if same_cat else (255, 100, 100)
                draw.text((x + 2, tile_size + 3), f"{ncat[:8]}", fill=color)
                draw.text((x + 2, tile_size + 14), f"d:{dist:.1f}", fill=(180, 180, 180))

        # 保存
        canvas_large = canvas.resize((canvas.width * 2, canvas.height * 2), Image.Resampling.NEAREST)
        save_path = OUTPUT_DIR / f"nearest_{query_idx}_{query_cat}.png"
        canvas_large.save(save_path)

        # 打印
        same = sum(1 for nidx in neighbor_indices if get_category(names[nidx]) == query_cat)
        print(f"  [{query_name}] 同类别: {same}/{n_neighbors}  最近距离: {neighbor_dists[0]:.2f}")

    print(f"\n最近邻图: {OUTPUT_DIR}")


from PIL import ImageDraw


def main():
    print("=" * 60)
    print("Latent Space 聚类分析")
    print("=" * 60)

    latents, names = load_data()
    print(f"加载 {len(names)} 个样本")

    images = load_images(names)
    print(f"加载 {len(images)} 张图片")

    # KMeans
    labels = kmeans_analysis(latents, names, images, n_clusters=8)

    # 最近邻
    nearest_neighbor_analysis(latents, names, images, n_neighbors=10)

    print("\n" + "=" * 60)
    print(f"分析完成! 输出: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

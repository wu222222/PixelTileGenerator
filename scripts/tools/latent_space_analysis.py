"""
V7 Latent Space 深度分析

Phase 7.1: UMAP 可视化 (类别聚类)
Phase 7.2: 跨类别插值
Phase 7.3: 类别向量实验 (snow_vector = mean(snow) - mean(grass))
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2

OUTPUT_DIR = project_root / "checkpoints" / "v7_latent_analysis"


def load_latent_data():
    """加载 v7 latent 数据"""
    latent_dir = project_root / "datasets" / "vqvae_latent_data_v7"
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))
    return latents, names


def extract_categories(names):
    """从文件名提取类别前缀"""
    categories = {}
    for name in names:
        stem = Path(name).stem
        # grass_from_gpt_0000 -> grass
        # tileset_from_gpt_0000 -> tileset
        # tileset3_from_gpt_0000 -> tileset3
        if "_from_" in stem:
            cat = stem.split("_from_")[0]
        elif "_" in stem:
            cat = stem.split("_")[0]
        else:
            cat = stem[:6]
        categories[name] = cat.lower()
    return categories


def phase71_umap(latents, names):
    """Phase 7.1: UMAP 可视化"""
    print("\n" + "=" * 60)
    print("Phase 7.1: UMAP 可视化")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PCA 降到 50 维
    flat = latents.reshape(latents.shape[0], -1)
    print(f"原始维度: {flat.shape[1]}")
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)
    print(f"PCA 50 维完成")

    # UMAP 降到 2 维
    print("UMAP 降维中...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(flat_pca)
    print(f"UMAP 完成: {embedding.shape}")

    # 提取类别
    categories = extract_categories(names)
    cat_list = [categories[n] for n in names]
    unique_cats = sorted(set(cat_list))

    # 颜色映射
    cmap = plt.cm.get_cmap('tab10', len(unique_cats))
    cat_colors = {cat: cmap(i) for i, cat in enumerate(unique_cats)}
    colors = [cat_colors[c] for c in cat_list]

    # 画图
    fig, ax = plt.subplots(figsize=(14, 10))
    for cat in unique_cats:
        mask = np.array([c == cat for c in cat_list])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cat_colors[cat]], label=f"{cat} ({mask.sum()})",
                   s=20, alpha=0.7, edgecolors='none')

    ax.set_title(f"V7 Latent Space UMAP (n={len(names)})", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "umap_categories.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"UMAP 图: {save_path}")

    # 统计各类别数量
    print("\n类别统计:")
    for cat in unique_cats:
        count = sum(1 for c in cat_list if c == cat)
        print(f"  {cat:15s}: {count:4d} 张")

    return categories, unique_cats, embedding


def phase72_cross_category_interpolation(latents, names, categories, unique_cats):
    """Phase 7.2: 跨类别插值"""
    print("\n" + "=" * 60)
    print("Phase 7.2: 跨类别插值")
    print("=" * 60)

    # 加载模型
    ckpt_path = project_root / "checkpoints/vqvae_v7/vqvae_v7_best.pth"
    device = "cuda"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
        latent_size=config.get("latent_size", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 找到各类别的代表样本 (最靠近类别中心的)
    cat_indices = defaultdict(list)
    for i, name in enumerate(names):
        cat = categories[name]
        cat_indices[cat].append(i)

    # 计算类别中心
    cat_centers = {}
    for cat, indices in cat_indices.items():
        cat_latents = latents[indices].reshape(len(indices), -1)
        cat_centers[cat] = cat_latents.mean(axis=0)

    # 选择要插值的类别对
    interp_pairs = [
        ("grass", "tileset"),
        ("tileset3", "tileset4"),
        ("tileset5", "tileset6"),
    ]

    # 找最接近中心的样本作为代表
    def find_representative(cat, n=1):
        indices = cat_indices[cat]
        center = cat_centers[cat]
        dists = [np.linalg.norm(latents[i].flatten() - center) for i in indices]
        sorted_idx = np.argsort(dists)
        return [indices[sorted_idx[j]] for j in range(min(n, len(sorted_idx)))]

    transform = transforms.Compose([transforms.ToTensor()])

    valid_pairs = []
    for cat1, cat2 in interp_pairs:
        if cat1 in cat_indices and cat2 in cat_indices:
            valid_pairs.append((cat1, cat2))

    if not valid_pairs:
        print("  没有可用的类别对")
        return

    fig, axes = plt.subplots(len(valid_pairs), 11, figsize=(22, len(valid_pairs) * 2))
    if len(valid_pairs) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, (cat1, cat2) in enumerate(valid_pairs):
            idx1 = find_representative(cat1)[0]
            idx2 = find_representative(cat2)[0]

            z1 = torch.FloatTensor(latents[idx1]).unsqueeze(0).to(device)
            z2 = torch.FloatTensor(latents[idx2]).unsqueeze(0).to(device)

            for col, alpha in enumerate(np.linspace(0, 1, 11)):
                z_interp = z1 * (1 - alpha) + z2 * alpha
                recon = model.decode(z_interp)
                arr = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]

                axes[row, col].imshow(arr)
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(f"α={alpha:.1f}", fontsize=8)

            axes[row, 0].set_ylabel(f"{cat1}\n→\n{cat2}", fontsize=9, rotation=0, labelpad=50)

    plt.suptitle("Cross-Category Interpolation (v7)", fontsize=14)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "cross_category_interpolation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"跨类别插值图: {save_path}")


def phase73_category_vectors(latents, names, categories, unique_cats):
    """Phase 7.3: 类别向量实验"""
    print("\n" + "=" * 60)
    print("Phase 7.3: 类别向量实验")
    print("=" * 60)

    # 加载模型
    ckpt_path = project_root / "checkpoints/vqvae_v7/vqvae_v7_best.pth"
    device = "cuda"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
        latent_size=config.get("latent_size", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 计算类别中心
    cat_indices = defaultdict(list)
    for i, name in enumerate(names):
        cat_indices[categories[name]].append(i)

    cat_centers = {}
    for cat, indices in cat_indices.items():
        cat_latents = latents[indices].reshape(len(indices), -1)
        cat_centers[cat] = cat_latents.mean(axis=0)

    # 类别向量: tileset_vector = mean(tileset) - mean(grass)
    vector_pairs = [
        ("tileset", "grass", "tileset-grass"),
        ("tileset3", "grass", "tileset3-grass"),
        ("tileset4", "grass", "tileset4-grass"),
        ("tileset5", "grass", "tileset5-grass"),
        ("tileset6", "grass", "tileset6-grass"),
    ]

    # 对 grass 样本应用类别向量
    base_cat = "grass"
    if base_cat not in cat_indices:
        # fallback: 用第一个类别
        base_cat = list(cat_indices.keys())[0]
        print(f"  grass 不存在，使用 {base_cat} 作为基础类别")

    # 选几个样本
    base_indices = cat_indices[base_cat][:4]

    transform = transforms.Compose([transforms.ToTensor()])

    valid_vectors = []
    for cat1, cat2, label in vector_pairs:
        if cat1 in cat_centers and cat2 in cat_centers:
            valid_vectors.append((cat1, cat2, label))

    if not valid_vectors:
        print("  没有可用的类别向量")
        return

    fig, axes = plt.subplots(len(base_indices), len(valid_vectors) + 1, figsize=((len(valid_vectors) + 1) * 3, len(base_indices) * 3))

    with torch.no_grad():
        for row, base_idx in enumerate(base_indices):
            z_base = torch.FloatTensor(latents[base_idx]).unsqueeze(0).to(device)
            recon_base = model.decode(z_base)
            arr_base = recon_base.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]

            axes[row, 0].imshow(arr_base)
            axes[row, 0].set_title(f"Original\n({names[base_idx][:15]})", fontsize=7)
            axes[row, 0].axis('off')

            for col, (cat1, cat2, label) in enumerate(valid_vectors):
                # 类别向量
                direction = cat_centers[cat1] - cat_centers[cat2]
                # 归一化到与 latent 相同的尺度
                direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
                # 用不同的强度
                strength = 0.5 * np.linalg.norm(cat_centers[base_cat] - cat_centers.get(cat1, cat_centers[base_cat]))
                direction_tensor = torch.FloatTensor(direction_norm.reshape(latents.shape[1:])).unsqueeze(0).to(device)

                z_new = z_base + direction_tensor * strength
                recon_new = model.decode(z_new)
                arr_new = recon_new.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]

                axes[row, col + 1].imshow(arr_new)
                axes[row, col + 1].set_title(f"+ {label}", fontsize=7)
                axes[row, col + 1].axis('off')

    plt.suptitle("Category Vector Arithmetic (v7)\ngrass + (cat1 - cat2)", fontsize=12)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "category_vector_arithmetic.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"类别向量图: {save_path}")

    # 打印类别间距离
    print("\n类别间欧氏距离 (latent space):")
    sorted_cats = sorted(cat_centers.keys())
    for i, cat1 in enumerate(sorted_cats):
        for j, cat2 in enumerate(sorted_cats):
            if i < j:
                dist = np.linalg.norm(cat_centers[cat1] - cat_centers[cat2])
                print(f"  {cat1:10s} ↔ {cat2:10s}: {dist:.2f}")


def main():
    print("=" * 60)
    print("V7 Latent Space 深度分析")
    print("=" * 60)

    latents, names = load_latent_data()
    print(f"加载 {len(names)} 个样本, latent shape: {latents.shape}")

    # Phase 7.1
    categories, unique_cats, embedding = phase71_umap(latents, names)

    # Phase 7.2
    phase72_cross_category_interpolation(latents, names, categories, unique_cats)

    # Phase 7.3
    phase73_category_vectors(latents, names, categories, unique_cats)

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

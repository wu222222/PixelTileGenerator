"""
Manifold Quality Map

分析每个簇的 latent 质量:
1. Cohesion (紧密度): 样本到中心的平均距离
2. Structure Score (结构性): 边缘能量 / 频率方差
3. Interpolation Stability (插值稳定性): 样本对插值的 MSE 变化
4. Codebook Diversity (码本多样性): 簇内唯一 token 数
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2

OUTPUT_DIR = project_root / "checkpoints" / "v7_manifold_quality"


def get_category(name):
    stem = Path(name).stem
    if "_from_" in stem:
        return stem.split("_from_")[0]
    return stem.split("_")[0]


def compute_edge_energy(img_array):
    """计算图像的边缘能量 (高频结构指标)"""
    # img_array: (H, W, C) uint8
    gray = img_array[:, :, :3].mean(axis=2).astype(float)
    # Sobel-like gradient
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    return float(dx.mean() + dy.mean())


def compute_frequency_energy(img_array):
    """计算频率域能量 (纹理复杂度)"""
    gray = img_array[:, :, :3].mean(axis=2).astype(float)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    # 高频能量 (外圈)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask = ((Y - cy) ** 2 + (X - cx) ** 2) > (min(h, w) // 4) ** 2
    high_freq = magnitude[mask].mean()
    total = magnitude.mean()
    return float(high_freq / (total + 1e-8))


def main():
    print("=" * 60)
    print("Manifold Quality Map")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    latent_dir = project_root / "datasets" / "vqvae_latent_data_v7"
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))
    indices = np.load(latent_dir / "indices.npy")
    print(f"加载 {len(names)} 个样本")

    # 加载图片
    data_dir = project_root / "datasets" / "classified" / "pixel_32_quantized"
    images = {}
    for name in names:
        path = data_dir / name
        if path.exists():
            images[name] = np.array(Image.open(path).convert("RGBA"))

    # KMeans
    flat = latents.reshape(latents.shape[0], -1)
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_pca)

    # 簇中心
    n_clusters = 8
    cluster_centers = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_centers[c] = latents[mask].mean(axis=0)

    # 加载模型 (用于插值稳定性测试)
    device = "cuda"
    ckpt_path = project_root / "checkpoints/vqvae_v7/vqvae_v7_best.pth"
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

    # 分析每个簇
    results = []

    for c in range(n_clusters):
        mask = labels == c
        indices_c = np.where(mask)[0]
        n = len(indices_c)

        cats = defaultdict(int)
        for i in indices_c:
            cats[get_category(names[i])] += 1
        top_cats = sorted(cats.items(), key=lambda x: -x[1])
        primary = top_cats[0][0]
        cat_str = ", ".join(f"{k}({v})" for k, v in top_cats[:3])

        # 1. Cohesion: 样本到中心的平均距离
        center_flat = cluster_centers[c].flatten()
        dists = np.linalg.norm(latents[indices_c].reshape(n, -1) - center_flat, axis=1)
        cohesion = float(dists.mean())
        cohesion_std = float(dists.std())

        # 2. Structure Score: 边缘能量
        edge_energies = []
        freq_energies = []
        for i in indices_c:
            name = names[i]
            if name in images:
                edge_energies.append(compute_edge_energy(images[name]))
                freq_energies.append(compute_frequency_energy(images[name]))
        edge_score = float(np.mean(edge_energies)) if edge_energies else 0
        freq_score = float(np.mean(freq_energies)) if freq_energies else 0

        # 3. Interpolation Stability: 随机样本对插值的 MSE 变化
        if n >= 4:
            rng = np.random.RandomState(42)
            pair_count = min(20, n * (n - 1) // 2)
            interp_losses = []
            for _ in range(pair_count):
                i1, i2 = rng.choice(n, 2, replace=False)
                z1 = torch.FloatTensor(latents[indices_c[i1]]).unsqueeze(0).to(device)
                z2 = torch.FloatTensor(latents[indices_c[i2]]).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon1 = model.decode(z1)
                    recon2 = model.decode(z2)
                    z_mid = (z1 + z2) / 2
                    recon_mid = model.decode(z_mid)
                    expected = (recon1 + recon2) / 2
                    mse = ((recon_mid - expected) ** 2).mean().item()
                    interp_losses.append(mse)
            stability = float(np.mean(interp_losses))
            stability_std = float(np.std(interp_losses))
        else:
            stability = 0
            stability_std = 0

        # 4. Codebook Diversity
        cluster_indices = indices[indices_c].flatten()
        unique_tokens = len(np.unique(cluster_indices))
        total_tokens = len(cluster_indices)
        token_entropy = 0
        if unique_tokens > 0:
            _, counts = np.unique(cluster_indices, return_counts=True)
            probs = counts / counts.sum()
            token_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        results.append({
            "cluster": c,
            "primary": primary,
            "categories": cat_str,
            "size": n,
            "cohesion": cohesion,
            "cohesion_std": cohesion_std,
            "edge_score": edge_score,
            "freq_score": freq_score,
            "stability": stability,
            "stability_std": stability_std,
            "unique_tokens": unique_tokens,
            "token_entropy": token_entropy,
        })

    # 打印报告
    print("\n" + "=" * 80)
    print(f"{'C':>2} {'Primary':>10} {'Size':>5} {'Cohesion':>10} {'Edge':>8} {'Freq':>8} {'Stability':>12} {'Tokens':>7} {'Entropy':>8}")
    print("=" * 80)

    for r in results:
        print(f"{r['cluster']:2d} {r['primary']:>10} {r['size']:5d} "
              f"{r['cohesion']:10.2f} {r['edge_score']:8.2f} {r['freq_score']:8.4f} "
              f"{r['stability']:12.6f} {r['unique_tokens']:7d} {r['token_entropy']:8.2f}")

    # 质量评分 (归一化后加权)
    def normalize(vals):
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-8:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    cohesions = [r["cohesion"] for r in results]
    edges = [r["edge_score"] for r in results]
    freqs = [r["freq_score"] for r in results]
    stabilities = [r["stability"] for r in results]
    entropies = [r["token_entropy"] for r in results]

    # cohesion: 越小越好 (越紧密)
    n_cohesion = [1 - v for v in normalize(cohesions)]
    # edge: 越高越好 (结构越强)
    n_edge = normalize(edges)
    # freq: 适度最好 (太高=纯噪声, 太低=纯色)
    n_freq = normalize(freqs)
    n_freq = [1 - abs(v - 0.5) * 2 for v in n_freq]  # 0.5 最好
    # stability: 越小越好 (越稳定)
    n_stability = [1 - v for v in normalize(stabilities)]
    # entropy: 越高越好 (码本多样性)
    n_entropy = normalize(entropies)

    weights = {"cohesion": 0.25, "edge": 0.25, "stability": 0.25, "entropy": 0.25}
    quality_scores = []
    for i in range(len(results)):
        score = (weights["cohesion"] * n_cohesion[i] +
                 weights["edge"] * n_edge[i] +
                 weights["stability"] * n_stability[i] +
                 weights["entropy"] * n_entropy[i])
        quality_scores.append(score)

    print("\n" + "=" * 60)
    print("质量评分 (0~1, 越高越好)")
    print("=" * 60)
    for i, r in enumerate(results):
        stars = "★" * int(quality_scores[i] * 5 + 0.5) + "☆" * (5 - int(quality_scores[i] * 5 + 0.5))
        print(f"  C{r['cluster']} {r['primary']:>10}: {quality_scores[i]:.3f} {stars}")

    # 生成可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cluster_labels = [f"C{r['cluster']}\n{r['primary']}" for r in results]

    # 1. Cohesion
    axes[0, 0].bar(range(8), cohesions, yerr=[r["cohesion_std"] for r in results],
                   color=['#4CAF50' if c < np.median(cohesions) else '#FF9800' for c in cohesions])
    axes[0, 0].set_title("Cohesion (lower = tighter)")
    axes[0, 0].set_xticks(range(8))
    axes[0, 0].set_xticklabels(cluster_labels, fontsize=8)

    # 2. Edge Score
    axes[0, 1].bar(range(8), edges, color='#2196F3')
    axes[0, 1].set_title("Edge Energy (higher = more structure)")
    axes[0, 1].set_xticks(range(8))
    axes[0, 1].set_xticklabels(cluster_labels, fontsize=8)

    # 3. Frequency
    axes[0, 2].bar(range(8), freqs, color='#9C27B0')
    axes[0, 2].set_title("High-Freq Ratio (texture complexity)")
    axes[0, 2].set_xticks(range(8))
    axes[0, 2].set_xticklabels(cluster_labels, fontsize=8)

    # 4. Stability
    axes[1, 0].bar(range(8), stabilities, yerr=[r["stability_std"] for r in results],
                   color=['#4CAF50' if s < np.median(stabilities) else '#f44336' for s in stabilities])
    axes[1, 0].set_title("Interpolation Instability (lower = better)")
    axes[1, 0].set_xticks(range(8))
    axes[1, 0].set_xticklabels(cluster_labels, fontsize=8)

    # 5. Token Entropy
    axes[1, 1].bar(range(8), entropies, color='#FF5722')
    axes[1, 1].set_title("Token Entropy (higher = more diverse)")
    axes[1, 1].set_xticks(range(8))
    axes[1, 1].set_xticklabels(cluster_labels, fontsize=8)

    # 6. Quality Score
    colors = ['#4CAF50' if q > 0.6 else '#FF9800' if q > 0.4 else '#f44336' for q in quality_scores]
    axes[1, 2].barh(range(8), quality_scores, color=colors)
    axes[1, 2].set_yticks(range(8))
    axes[1, 2].set_yticklabels(cluster_labels, fontsize=9)
    axes[1, 2].set_title("Quality Score (weighted)")
    axes[1, 2].set_xlim(0, 1)
    for i, v in enumerate(quality_scores):
        axes[1, 2].text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=9)

    plt.suptitle("Manifold Quality Map — PixelTileGAN v7", fontsize=14)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "quality_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n质量地图: {save_path}")

    # 保存 JSON
    with open(OUTPUT_DIR / "quality_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    # 结论
    best = results[np.argmax(quality_scores)]
    worst = results[np.argmin(quality_scores)]
    print(f"\n最高质量簇: C{best['cluster']} ({best['primary']}) = {max(quality_scores):.3f}")
    print(f"最低质量簇: C{worst['cluster']} ({worst['primary']}) = {min(quality_scores):.3f}")

    print("\n" + "=" * 60)
    print("结论:")
    print("  高质量簇 → 适合语义编辑、插值、导航")
    print("  低质量簇 → 只适合采样,不适合编辑")
    print("=" * 60)


if __name__ == "__main__":
    main()

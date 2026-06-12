"""
Cluster Navigation: 簇间连续语义插值

测试 latent 空间是否具有连续语义过渡:
- Grass(7) → Desert(5): 绿色→沙色
- Grass(7) → Stone(4): 随机→规则
- Swamp(6) → Desert(5): 湿润→干燥
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2

OUTPUT_DIR = project_root / "checkpoints" / "v7_cluster_navigation"


def get_category(name):
    stem = Path(name).stem
    if "_from_" in stem:
        return stem.split("_from_")[0]
    return stem.split("_")[0]


def main():
    print("=" * 60)
    print("Cluster Navigation")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    latent_dir = project_root / "datasets" / "vqvae_latent_data_v7"
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))

    flat = latents.reshape(latents.shape[0], -1)

    # PCA + KMeans
    print("PCA + KMeans...")
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_pca)

    # 加载模型
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

    # 簇中心 (在原始 latent 空间)
    cluster_centers = {}
    for c in range(8):
        mask = labels == c
        cluster_centers[c] = latents[mask].mean(axis=0)

    # 找每个簇中最靠近中心的样本 (用于解码验证)
    def find_representative(cluster_id, n=1):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        center = cluster_centers[cluster_id].flatten()
        dists = np.linalg.norm(latents[indices].reshape(len(indices), -1) - center, axis=1)
        sorted_idx = indices[np.argsort(dists)]
        return sorted_idx[:n].tolist()

    # 定义导航路径
    nav_paths = [
        (7, 5, "Grass → Desert"),
        (7, 4, "Grass → Stone"),
        (6, 5, "Swamp → Desert"),
    ]

    n_steps = 11

    with torch.no_grad():
        for c_from, c_to, label in nav_paths:
            print(f"\n{label} (Cluster {c_from} → Cluster {c_to})")

            # 簇中心插值
            z_from = torch.FloatTensor(cluster_centers[c_from]).unsqueeze(0).to(device)
            z_to = torch.FloatTensor(cluster_centers[c_to]).unsqueeze(0).to(device)

            fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2, 4))

            for col, alpha in enumerate(np.linspace(0, 1, n_steps)):
                # 簇中心插值
                z_interp = z_from * (1 - alpha) + z_to * alpha
                recon = model.decode(z_interp)
                arr = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]
                axes[0, col].imshow(arr)
                axes[0, col].set_title(f"α={alpha:.1f}", fontsize=8)
                axes[0, col].axis('off')

                # 真实样本插值 (用最靠近中心的样本)
                rep_from = find_representative(c_from, 1)[0]
                rep_to = find_representative(c_to, 1)[0]
                z1 = torch.FloatTensor(latents[rep_from]).unsqueeze(0).to(device)
                z2 = torch.FloatTensor(latents[rep_to]).unsqueeze(0).to(device)
                z_real_interp = z1 * (1 - alpha) + z2 * alpha
                recon_real = model.decode(z_real_interp)
                arr_real = recon_real.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]
                axes[1, col].imshow(arr_real)
                axes[1, col].axis('off')
                if col == 0:
                    axes[1, col].set_ylabel("Real\nSamples", fontsize=7, rotation=0, labelpad=40)

            axes[0, 0].set_ylabel("Cluster\nCenter", fontsize=7, rotation=0, labelpad=40)

            plt.suptitle(f"{label}  (Cluster {c_from} → Cluster {c_to})", fontsize=12)
            plt.tight_layout()
            save_path = OUTPUT_DIR / f"nav_{c_from}_to_{c_to}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  保存: {save_path}")

    # 额外: 多样本插值 (每个簇选3个不同样本)
    print("\n多样本插值...")
    with torch.no_grad():
        for c_from, c_to, label in nav_paths:
            reps_from = find_representative(c_from, 3)
            reps_to = find_representative(c_to, 3)

            fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 2, 6))

            for row in range(3):
                z1 = torch.FloatTensor(latents[reps_from[row]]).unsqueeze(0).to(device)
                z2 = torch.FloatTensor(latents[reps_to[row]]).unsqueeze(0).to(device)

                for col, alpha in enumerate(np.linspace(0, 1, n_steps)):
                    z_interp = z1 * (1 - alpha) + z2 * alpha
                    recon = model.decode(z_interp)
                    arr = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]
                    axes[row, col].imshow(arr)
                    axes[row, col].axis('off')
                    if row == 0:
                        axes[row, col].set_title(f"α={alpha:.1f}", fontsize=8)

            plt.suptitle(f"{label} — Multi-sample", fontsize=12)
            plt.tight_layout()
            save_path = OUTPUT_DIR / f"nav_{c_from}_to_{c_to}_multi.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  保存: {save_path}")

    # 簇间距离矩阵
    print("\n簇间距离矩阵:")
    n_clusters = len(cluster_centers)
    dist_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            dist_matrix[i, j] = np.linalg.norm(
                cluster_centers[i].flatten() - cluster_centers[j].flatten())

    header = "       " + "  ".join(f"C{i:5d}" for i in range(n_clusters))
    print(header)
    for i in range(n_clusters):
        row = f"C{i:2d}  " + "  ".join(f"{dist_matrix[i,j]:5.1f}" for j in range(n_clusters))
        print(row)

    # 保存距离矩阵图
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap='viridis')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels([f"C{i}" for i in range(n_clusters)])
    ax.set_yticklabels([f"C{i}" for i in range(n_clusters)])
    for i in range(n_clusters):
        for j in range(n_clusters):
            ax.text(j, i, f"{dist_matrix[i,j]:.1f}", ha="center", va="center",
                    color="white" if dist_matrix[i,j] > dist_matrix.max()/2 else "black",
                    fontsize=9)
    ax.set_title("Cluster Distance Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_matrix.png", dpi=150)
    plt.close()
    print(f"\n距离矩阵图: {OUTPUT_DIR / 'distance_matrix.png'}")

    print("\n" + "=" * 60)
    print(f"完成! 输出: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

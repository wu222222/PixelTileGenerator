"""
验证 VQ-VAE v6 (20×20) 是否真正提升了有效信息

检查:
1. Token 分布是否塌缩
2. v5 vs v6 重建对比
3. PCA 95% 方差需要多少维
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


def check_token_distribution(indices_path, name):
    """检查 token 分布是否塌缩"""
    indices = np.load(indices_path)
    flat = indices.flatten()

    unique, counts = np.unique(flat, return_counts=True)
    total = len(flat)

    print(f"\n{'='*50}")
    print(f"[{name}] Token 分布分析")
    print(f"{'='*50}")
    print(f"  总 token 数: {total:,}")
    print(f"  唯一 token 数: {len(unique)}")
    print(f"  使用率: {len(unique)}/码本大小")

    # Top-10 最常用 token
    top_idx = np.argsort(counts)[::-1][:10]
    print(f"\n  Top-10 最常用 token:")
    for rank, idx in enumerate(top_idx):
        pct = counts[idx] / total * 100
        print(f"    #{rank+1}: token {unique[idx]:3d}  出现 {counts[idx]:6,} 次 ({pct:5.1f}%)")

    # 基尼系数 (衡量分布不均匀程度, 0=均匀, 1=完全集中)
    sorted_counts = np.sort(counts)[::-1].astype(float)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    gini = 1 - 2 * np.sum(cumulative / cumulative[-1]) / n + 1 / n
    print(f"\n  基尼系数: {gini:.3f} (0=均匀, 1=完全集中)")

    # Top-3 占比
    top3_pct = sorted_counts[:3].sum() / total * 100
    print(f"  Top-3 token 占比: {top3_pct:.1f}%")

    return {
        "total_tokens": total,
        "unique_tokens": len(unique),
        "gini": float(gini),
        "top3_pct": float(top3_pct),
        "top10": [(int(unique[idx]), int(counts[idx]), float(counts[idx]/total*100)) for idx in top_idx],
    }


def check_pca_dimensionality(latents_path, name):
    """检查 PCA 95% 方差需要多少维"""
    from sklearn.decomposition import PCA

    latents = np.load(latents_path)
    flat = latents.reshape(latents.shape[0], -1)

    print(f"\n{'='*50}")
    print(f"[{name}] PCA 维度分析")
    print(f"{'='*50}")
    print(f"  样本数: {flat.shape[0]}")
    print(f"  原始维度: {flat.shape[1]}")

    # 拟合 PCA
    pca = PCA()
    pca.fit(flat)

    # 累积方差
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    for threshold in [0.90, 0.95, 0.99]:
        dims = np.searchsorted(cumvar, threshold) + 1
        print(f"  {threshold*100:.0f}% 方差: {dims} 维")

    return {
        "original_dim": flat.shape[1],
        "dims_90": int(np.searchsorted(cumvar, 0.90) + 1),
        "dims_95": int(np.searchsorted(cumvar, 0.95) + 1),
        "dims_99": int(np.searchsorted(cumvar, 0.99) + 1),
    }


def compare_reconstruction(v5_ckpt, v6_ckpt, data_dir, device='cuda'):
    """并排对比 v5 和 v6 重建质量"""
    output_dir = project_root / "checkpoints" / "v6_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 v5
    ckpt5 = torch.load(project_root / v5_ckpt, map_location=device)
    cfg5 = ckpt5.get("config", {})
    model5 = VQVAEv2(
        in_channels=cfg5.get("in_channels", 4),
        hidden_channels=cfg5.get("hidden_channels", 256),
        embedding_dim=cfg5.get("embedding_dim", 64),
        num_embeddings=cfg5.get("num_embeddings", 256),
        latent_size=cfg5.get("latent_size", 16),
    ).to(device)
    model5.load_state_dict(ckpt5["model_state_dict"])
    model5.eval()

    # 加载 v6
    ckpt6 = torch.load(project_root / v6_ckpt, map_location=device)
    cfg6 = ckpt6.get("config", {})
    model6 = VQVAEv2(
        in_channels=cfg6.get("in_channels", 4),
        hidden_channels=cfg6.get("hidden_channels", 256),
        embedding_dim=cfg6.get("embedding_dim", 64),
        num_embeddings=cfg6.get("num_embeddings", 512),
        latent_size=cfg6.get("latent_size", 20),
    ).to(device)
    model6.load_state_dict(ckpt6["model_state_dict"])
    model6.eval()

    # 加载测试图片
    data_path = project_root / data_dir
    image_files = sorted([f for f in data_path.iterdir() if f.suffix == ".png"])[:8]
    transform = transforms.Compose([transforms.ToTensor()])

    fig, axes = plt.subplots(3, 8, figsize=(24, 9))

    mse5_total, mse6_total = 0, 0

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGBA")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # v5 重建
            recon5, _, _ = model5(img_tensor)
            mse5 = ((recon5 - img_tensor) ** 2).mean().item()
            mse5_total += mse5

            # v6 重建
            recon6, _, _ = model6(img_tensor)
            mse6 = ((recon6 - img_tensor) ** 2).mean().item()
            mse6_total += mse6

            # 显示
            def to_display(t):
                arr = t.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                return arr[:, :, :3]  # 只取 RGB

            axes[0, i].imshow(to_display(img_tensor))
            axes[0, i].set_title(f"Original", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(to_display(recon5))
            axes[1, i].set_title(f"v5 (MSE:{mse5:.4f})", fontsize=8)
            axes[1, i].axis('off')

            axes[2, i].imshow(to_display(recon6))
            axes[2, i].set_title(f"v6 (MSE:{mse6:.4f})", fontsize=8)
            axes[2, i].axis('off')

    avg5 = mse5_total / len(image_files)
    avg6 = mse6_total / len(image_files)

    fig.suptitle(f"v5 avg MSE: {avg5:.4f}  |  v6 avg MSE: {avg6:.4f}  |  差值: {(avg6-avg5):.4f}", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "v5_vs_v6_reconstruction.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n{'='*50}")
    print(f"重建质量对比")
    print(f"{'='*50}")
    print(f"  v5 平均 MSE: {avg5:.4f}")
    print(f"  v6 平均 MSE: {avg6:.4f}")
    print(f"  差值: {(avg6-avg5):.4f} ({'v6更好' if avg6 < avg5 else 'v5更好'})")
    print(f"  对比图: {save_path}")

    return {"v5_mse": avg5, "v6_mse": avg6}


def main():
    print("=" * 60)
    print("VQ-VAE v6 验证")
    print("=" * 60)

    # 1. Token 分布
    v5_indices = project_root / "datasets/vqvae_latent_data/indices.npy"
    v6_indices = project_root / "datasets/vqvae_latent_data_v6/indices.npy"

    if v5_indices.exists():
        check_token_distribution(v5_indices, "v5 (16×16)")
    else:
        print(f"[跳过] v5 indices 不存在: {v5_indices}")

    if v6_indices.exists():
        check_token_distribution(v6_indices, "v6 (20×20)")
    else:
        print(f"[跳过] v6 indices 不存在: {v6_indices}")

    # 2. PCA 维度
    v5_latents = project_root / "datasets/vqvae_latent_data/latents.npy"
    v6_latents = project_root / "datasets/vqvae_latent_data_v6/latents.npy"

    if v5_latents.exists():
        check_pca_dimensionality(v5_latents, "v5 (16×16)")
    else:
        print(f"[跳过] v5 latents 不存在: {v5_latents}")

    if v6_latents.exists():
        check_pca_dimensionality(v6_latents, "v6 (20×20)")
    else:
        print(f"[跳过] v6 latents 不存在: {v6_latents}")

    # 3. 重建对比
    v5_ckpt = "checkpoints/vqvae_v5/vqvae_v5_best.pth"
    v6_ckpt = "checkpoints/vqvae_v6/vqvae_v6_best.pth"

    if (project_root / v5_ckpt).exists() and (project_root / v6_ckpt).exists():
        compare_reconstruction(v5_ckpt, v6_ckpt, "datasets/classified/pixel_32_quantized")
    else:
        print(f"[跳过] checkpoint 不存在")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

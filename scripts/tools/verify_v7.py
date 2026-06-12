"""
VQ-VAE v7 验证 (数据规模实验)

检查:
1. 码本利用率 + Token 熵
2. PCA 95% 方差维度
3. 插值实验 (跨类别过渡)
4. v5 vs v7 重建对比
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

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
    """Token 分布 + 熵"""
    indices = np.load(indices_path)
    flat = indices.flatten()
    total = len(flat)
    unique, counts = np.unique(flat, return_counts=True)

    # Token 熵
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(unique)) if len(unique) > 0 else 0

    print(f"\n{'='*50}")
    print(f"[{name}] Token 分析")
    print(f"{'='*50}")
    print(f"  总 token 数: {total:,}")
    print(f"  唯一 token 数: {len(unique)}")
    print(f"  使用率: {len(unique)}/码本大小")
    print(f"  Token 熵: {entropy:.2f} bits (最大: {max_entropy:.2f})")
    print(f"  熵效率: {entropy/max_entropy*100:.1f}%" if max_entropy > 0 else "")

    # Top-10
    top_idx = np.argsort(counts)[::-1][:10]
    print(f"\n  Top-10:")
    for rank, idx in enumerate(top_idx):
        pct = counts[idx] / total * 100
        print(f"    #{rank+1}: token {unique[idx]:3d}  {counts[idx]:6,} 次 ({pct:5.1f}%)")

    # Top-3 占比
    sorted_counts = np.sort(counts)[::-1].astype(float)
    top3_pct = sorted_counts[:3].sum() / total * 100
    print(f"\n  Top-3 占比: {top3_pct:.1f}%")

    return {
        "total_tokens": total,
        "unique_tokens": int(len(unique)),
        "usage_rate": float(len(unique)),
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "top3_pct": float(top3_pct),
    }


def check_pca_dimensionality(latents_path, name):
    """PCA 维度分析"""
    from sklearn.decomposition import PCA

    latents = np.load(latents_path)
    flat = latents.reshape(latents.shape[0], -1)

    print(f"\n{'='*50}")
    print(f"[{name}] PCA 分析")
    print(f"{'='*50}")
    print(f"  样本数: {flat.shape[0]}")
    print(f"  原始维度: {flat.shape[1]}")

    pca = PCA()
    pca.fit(flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    results = {}
    for threshold in [0.90, 0.95, 0.99]:
        dims = int(np.searchsorted(cumvar, threshold) + 1)
        results[f"dims_{int(threshold*100)}"] = dims
        print(f"  {threshold*100:.0f}% 方差: {dims} 维")

    return results


def interpolation_experiment(v7_ckpt, data_dir, output_dir, device='cuda'):
    """插值实验: 在两个不同类别样本间插值"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(project_root / v7_ckpt, map_location=device)
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

    data_path = project_root / data_dir
    image_files = sorted([f for f in data_path.iterdir() if f.suffix == ".png"])
    transform = transforms.Compose([transforms.ToTensor()])

    # 选几个不同类别的样本进行插值
    # 用文件名前缀猜测类别
    categories = {}
    for f in image_files:
        prefix = f.stem.rsplit('_', 1)[0] if '_' in f.stem else f.stem
        if prefix not in categories:
            categories[prefix] = f
    cat_list = list(categories.items())
    print(f"\n  发现 {len(cat_list)} 个类别前缀")

    # 选前4个类别做插值
    pairs = []
    for i in range(min(4, len(cat_list))):
        for j in range(i + 1, min(4, len(cat_list))):
            pairs.append((cat_list[i], cat_list[j]))

    pairs = pairs[:4]  # 最多4对

    if not pairs:
        print("  [跳过] 样本不足")
        return

    fig, axes = plt.subplots(len(pairs), 11, figsize=(22, len(pairs) * 2))

    with torch.no_grad():
        for row, ((name1, f1), (name2, f2)) in enumerate(pairs):
            img1 = transform(Image.open(f1).convert("RGBA")).unsqueeze(0).to(device)
            img2 = transform(Image.open(f2).convert("RGBA")).unsqueeze(0).to(device)

            z1, _ = model.encode(img1)
            z2, _ = model.encode(img2)

            for col, alpha in enumerate(np.linspace(0, 1, 11)):
                z_interp = z1 * (1 - alpha) + z2 * alpha
                recon = model.decode(z_interp)
                arr = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]

                axes[row, col].imshow(arr)
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(f"{alpha:.1f}", fontsize=8)

            axes[row, 0].set_ylabel(f"{name1[:12]}\n→\n{name2[:12]}", fontsize=7, rotation=0, labelpad=60)

    plt.suptitle("Latent Interpolation (v7)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "interpolation_v7.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  插值图: {save_path}")


def compare_reconstruction(v5_ckpt, v7_ckpt, data_dir, output_dir, device='cuda'):
    """v5 vs v7 重建对比"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    def load_model(ckpt_path):
        ckpt = torch.load(project_root / ckpt_path, map_location=device)
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
        return model

    model5 = load_model(v5_ckpt)
    model7 = load_model(v7_ckpt)

    data_path = project_root / data_dir
    image_files = sorted([f for f in data_path.iterdir() if f.suffix == ".png"])
    # 用 v5 训练集外的图片做测试
    test_files = image_files[:8]
    transform = transforms.Compose([transforms.ToTensor()])

    fig, axes = plt.subplots(3, 8, figsize=(24, 9))
    mse5_total, mse7_total = 0, 0

    with torch.no_grad():
        for i, img_path in enumerate(test_files):
            img = Image.open(img_path).convert("RGBA")
            img_tensor = transform(img).unsqueeze(0).to(device)

            recon5, _, _ = model5(img_tensor)
            mse5 = ((recon5 - img_tensor) ** 2).mean().item()
            mse5_total += mse5

            recon7, _, _ = model7(img_tensor)
            mse7 = ((recon7 - img_tensor) ** 2).mean().item()
            mse7_total += mse7

            def to_display(t):
                return t.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()[:, :, :3]

            axes[0, i].imshow(to_display(img_tensor))
            axes[0, i].set_title("Original", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(to_display(recon5))
            axes[1, i].set_title(f"v5 ({mse5:.4f})", fontsize=8)
            axes[1, i].axis('off')

            axes[2, i].imshow(to_display(recon7))
            axes[2, i].set_title(f"v7 ({mse7:.4f})", fontsize=8)
            axes[2, i].axis('off')

    avg5 = mse5_total / len(test_files)
    avg7 = mse7_total / len(test_files)

    fig.suptitle(f"v5 MSE: {avg5:.4f}  |  v7 MSE: {avg7:.4f}", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "v5_vs_v7_reconstruction.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n{'='*50}")
    print(f"重建对比")
    print(f"{'='*50}")
    print(f"  v5 MSE: {avg5:.4f}")
    print(f"  v7 MSE: {avg7:.4f}")
    print(f"  对比图: {save_path}")

    return {"v5_mse": avg5, "v7_mse": avg7}


def main():
    print("=" * 60)
    print("VQ-VAE v7 验证 (数据规模实验)")
    print("=" * 60)

    v5_indices = project_root / "datasets/vqvae_latent_data/indices.npy"
    v7_indices = project_root / "datasets/vqvae_latent_data_v7/indices.npy"
    v5_latents = project_root / "datasets/vqvae_latent_data/latents.npy"
    v7_latents = project_root / "datasets/vqvae_latent_data_v7/latents.npy"
    output_dir = project_root / "checkpoints" / "v7_verification"

    # 1. Token 分布 + 熵
    if v5_indices.exists():
        check_token_distribution(v5_indices, "v5 (314张)")
    if v7_indices.exists():
        check_token_distribution(v7_indices, "v7 (1778张)")

    # 2. PCA 维度
    if v5_latents.exists():
        check_pca_dimensionality(v5_latents, "v5 (314张)")
    if v7_latents.exists():
        check_pca_dimensionality(v7_latents, "v7 (1778张)")

    # 3. 重建对比
    v5_ckpt = "checkpoints/vqvae_v5/vqvae_v5_best.pth"
    v7_ckpt = "checkpoints/vqvae_v7/vqvae_v7_best.pth"
    if (project_root / v5_ckpt).exists() and (project_root / v7_ckpt).exists():
        compare_reconstruction(v5_ckpt, v7_ckpt,
                               "datasets/classified/pixel_32_quantized",
                               output_dir)

    # 4. 插值实验
    if (project_root / v7_ckpt).exists():
        interpolation_experiment(v7_ckpt,
                                "datasets/classified/pixel_32_quantized",
                                output_dir)

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

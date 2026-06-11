"""
Latent空间PCA分析

功能:
- 加载所有latent codes
- 执行PCA分析
- 统计累计解释方差
- 决定最佳降维维度
"""

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 配置
CONFIG = {
    "latent_data": "datasets/vqvae_latent_data",
    "output_dir": "checkpoints/vqvae_v5/pca_analysis",
}


def main():
    print("=" * 60)
    print("Latent空间PCA分析")
    print("=" * 60)

    # 路径
    latent_data_path = project_root / CONFIG["latent_data"]
    output_dir = project_root / CONFIG["output_dir"]

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载latent数据
    latents = np.load(latent_data_path / "latents.npy")
    print(f"Latent形状: {latents.shape}")

    # 展平latent: [N, 64, 16, 16] -> [N, 16384]
    N = latents.shape[0]
    latents_flat = latents.reshape(N, -1)
    print(f"展平后: {latents_flat.shape}")

    # 执行PCA
    print("\n执行PCA...")
    pca = PCA()
    pca.fit(latents_flat)

    # 计算累计解释方差
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 找到不同阈值所需的维度
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    print("\n累计解释方差:")
    for threshold in thresholds:
        n_dims = np.argmax(cumulative_variance >= threshold) + 1
        print(f"  {threshold*100:.0f}% 方差: {n_dims} 维")

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: 累计解释方差曲线
    axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
    axes[0].axhline(y=0.9, color='r', linestyle='--', label='90%')
    axes[0].axhline(y=0.95, color='g', linestyle='--', label='95%')
    axes[0].set_xlabel('Principal Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title('PCA Cumulative Explained Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 500)  # 只显示前500维

    # 图2: 前100维的解释方差
    n_show = min(100, len(pca.explained_variance_ratio_))
    axes[1].bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show], alpha=0.7)
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title('Top 100 Components Explained Variance')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_dir / "pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPCA分析图已保存: {output_dir / 'pca_analysis.png'}")

    # 保存统计结果
    stats = {
        "total_dimensions": int(latents_flat.shape[1]),
        "total_samples": int(N),
        "cumulative_variance": {
            f"{int(t*100)}%": int(np.argmax(cumulative_variance >= t) + 1)
            for t in thresholds
        },
        "top_10_components_variance": pca.explained_variance_ratio_[:10].tolist(),
    }

    with open(output_dir / "pca_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("PCA分析完成!")
    print("=" * 60)
    print(f"总维度: {latents_flat.shape[1]}")
    print(f"样本数: {N}")
    print("\n关键发现:")

    # 找到90%和95%方差所需的维度
    dim_90 = np.argmax(cumulative_variance >= 0.9) + 1
    dim_95 = np.argmax(cumulative_variance >= 0.95) + 1

    print(f"  90% 方差: {dim_90} 维 (压缩率: {latents_flat.shape[1]/dim_90:.1f}x)")
    print(f"  95% 方差: {dim_95} 维 (压缩率: {latents_flat.shape[1]/dim_95:.1f}x)")

    print("\n建议:")
    if dim_90 < 256:
        print(f"  → GAN可以降到 {dim_90} 维 (90%方差)")
        print(f"  → 难度从 {latents_flat.shape[1]} 维降到 {dim_90} 维")
    elif dim_95 < 512:
        print(f"  → GAN可以降到 {dim_95} 维 (95%方差)")
        print(f"  → 难度从 {latents_flat.shape[1]} 维降到 {dim_95} 维")
    else:
        print(f"  → 有效维度仍然很高，建议用Transformer路线")

    print("=" * 60)


if __name__ == "__main__":
    main()

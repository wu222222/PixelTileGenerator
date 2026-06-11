"""
极端扰动测试

功能:
- 测试更大的sigma值
- 计算latent变化量
- 验证Decoder是否过于平滑
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v5/vqvae_v5_best.pth",
    "latent_data": "datasets/vqvae_latent_data",
    "pca_model": "checkpoints/vqvae_v5/pca_reconstruction/pca_model.pkl",
    "output_dir": "checkpoints/vqvae_v5/extreme_perturbation",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 极端扰动强度
    "sigma_values": [0.1, 0.5, 1.0, 2.0, 5.0],

    # 选择几个真实样本
    "num_base_samples": 3,
}


def quantize_image(img: Image.Image, colors: int = 32) -> Image.Image:
    """对图片进行颜色量化"""
    if img.mode == "RGBA":
        quantized = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    else:
        quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGBA")


def main():
    print("=" * 60)
    print("极端扰动测试")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    latent_data_path = project_root / CONFIG["latent_data"]
    pca_model_path = project_root / CONFIG["pca_model"]
    output_dir = project_root / CONFIG["output_dir"]

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载VQ-VAE
    print(f"加载VQ-VAE: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})

    vqvae = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
    ).to(CONFIG["device"])

    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    # 加载PCA模型
    print(f"加载PCA模型: {pca_model_path}")
    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    # 加载latent数据
    latents = np.load(latent_data_path / "latents.npy")
    names = json.load(open(latent_data_path / "names.json"))
    print(f"Latent形状: {latents.shape}")

    # 展平latent
    N = latents.shape[0]
    latents_flat = latents.reshape(N, -1)

    # PCA压缩
    latents_pca = pca.transform(latents_flat)

    # 计算PCA空间的统计信息
    pca_std = np.std(latents_pca, axis=0)
    print(f"\nPCA空间标准差统计:")
    print(f"  平均标准差: {pca_std.mean():.4f}")
    print(f"  最大标准差: {pca_std.max():.4f}")
    print(f"  最小标准差: {pca_std.min():.4f}")

    # 选择基础样本
    base_indices = np.random.choice(N, CONFIG["num_base_samples"], replace=False)

    print(f"\n选择 {CONFIG['num_base_samples']} 个基础样本")
    print(f"扰动强度: {CONFIG['sigma_values']}")

    # 存储结果用于分析
    results = []

    for base_idx in base_indices:
        base_name = names[base_idx]
        z_pca_original = latents_pca[base_idx]

        print(f"\n{'='*40}")
        print(f"基础样本: {base_name}")
        print(f"{'='*40}")

        for sigma in CONFIG["sigma_values"]:
            # 添加扰动
            noise = np.random.randn(*z_pca_original.shape) * sigma
            z_pca_new = z_pca_original + noise

            # 计算PCA空间中的距离
            pca_distance = np.linalg.norm(noise)

            # 逆PCA
            z_new = pca.inverse_transform(z_pca_new.reshape(1, -1))
            z_new = z_new.reshape(latents[0].shape)

            # 计算原始latent空间中的距离
            z_original = latents[base_idx]
            latent_distance = np.linalg.norm(z_new - z_original)

            # 用Decoder生成图片
            z_tensor = torch.FloatTensor(z_new).unsqueeze(0).to(CONFIG["device"])
            with torch.no_grad():
                img = vqvae.decode(z_tensor)

            # 量化并保存
            img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())
            img_quantized = quantize_image(img_pil, colors=32)

            filename = f"base{base_idx:02d}_sigma{sigma:.1f}.png"
            img_quantized.save(output_dir / filename)

            # 记录结果
            results.append({
                "base_idx": int(base_idx),
                "base_name": base_name,
                "sigma": sigma,
                "pca_distance": float(pca_distance),
                "latent_distance": float(latent_distance),
                "filename": filename,
            })

            print(f"  σ={sigma:.1f}: PCA距离={pca_distance:.4f}, Latent距离={latent_distance:.4f}")

    # 创建对比图
    print("\n创建对比图...")
    create_comparison_grid(output_dir, results, CONFIG)

    # 保存结果
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 分析结果
    print("\n" + "=" * 60)
    print("极端扰动测试完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")

    # 分析距离和sigma的关系
    print("\n距离分析:")
    for sigma in CONFIG["sigma_values"]:
        sigma_results = [r for r in results if r["sigma"] == sigma]
        avg_pca_dist = np.mean([r["pca_distance"] for r in sigma_results])
        avg_latent_dist = np.mean([r["latent_distance"] for r in sigma_results])
        print(f"  σ={sigma:.1f}: 平均PCA距离={avg_pca_dist:.4f}, 平均Latent距离={avg_latent_dist:.4f}")

    print("\n请查看生成的图片:")
    print("  - 如果σ=5.0仍然没有明显变化 → Decoder过于平滑")
    print("  - 如果σ=1.0-2.0开始有变化 → 最佳区域在此范围")
    print("=" * 60)


def create_comparison_grid(output_dir, results, config):
    """创建对比图"""
    # 按基础样本组织
    organized = {}
    for r in results:
        base_idx = r["base_idx"]
        if base_idx not in organized:
            organized[base_idx] = []
        organized[base_idx].append(r)

    # 计算布局
    num_base = len(organized)
    num_sigma = len(config["sigma_values"])
    cell_size = 64
    padding = 5

    total_width = num_sigma * (cell_size + padding) + padding
    total_height = num_base * (cell_size + padding) + padding + 40

    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), "Extreme Perturbation Test", fill="white")

    # 绘制图片
    for base_idx, base_results in organized.items():
        for sigma_idx, result in enumerate(base_results):
            x = padding + sigma_idx * (cell_size + padding)
            y = 40 + base_idx * (cell_size + padding)

            # 加载并粘贴图片
            img_path = output_dir / result["filename"]
            if img_path.exists():
                img = Image.open(img_path)
                img_small = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                grid.paste(img_small, (x, y))

                # 绘制sigma标签
                draw.text((x, y + cell_size + 2), f"σ={result['sigma']:.1f}", fill="gray")

    # 保存
    grid.save(output_dir / "extreme_perturbation_grid.png")
    print(f"对比图已保存: {output_dir / 'extreme_perturbation_grid.png'}")


if __name__ == "__main__":
    main()

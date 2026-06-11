"""
PCA邻域采样实验

功能:
- 选择真实latent
- 在PCA空间中添加扰动
- 生成新纹理
- 测试不同扰动强度
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
    "output_dir": "checkpoints/vqvae_v5/neighbor_sampling",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 扰动强度
    "sigma_values": [0.01, 0.03, 0.05, 0.1, 0.2],

    # 每个sigma生成的数量
    "samples_per_sigma": 4,

    # 选择几个真实样本进行扰动
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
    print("PCA邻域采样实验")
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

    # 选择基础样本
    base_indices = np.random.choice(N, CONFIG["num_base_samples"], replace=False)

    print(f"\n选择 {CONFIG['num_base_samples']} 个基础样本进行扰动")
    print(f"扰动强度: {CONFIG['sigma_values']}")

    # 生成所有样本
    all_results = []

    for base_idx in base_indices:
        base_name = names[base_idx]
        z_pca_original = latents_pca[base_idx]

        print(f"\n基础样本: {base_name}")

        for sigma in CONFIG["sigma_values"]:
            print(f"  σ={sigma}:")

            for sample_idx in range(CONFIG["samples_per_sigma"]):
                # 添加扰动
                noise = np.random.randn(*z_pca_original.shape) * sigma
                z_pca_new = z_pca_original + noise

                # 逆PCA
                z_new = pca.inverse_transform(z_pca_new.reshape(1, -1))

                # reshape回原始形状
                z_new = z_new.reshape(latents[0].shape)

                # 用Decoder生成图片
                z_tensor = torch.FloatTensor(z_new).unsqueeze(0).to(CONFIG["device"])
                with torch.no_grad():
                    img = vqvae.decode(z_tensor)

                # 量化并保存
                img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())
                img_quantized = quantize_image(img_pil, colors=32)

                # 保存
                filename = f"base{base_idx:02d}_sigma{sigma:.2f}_sample{sample_idx:02d}.png"
                img_quantized.save(output_dir / filename)

                all_results.append({
                    "base_idx": int(base_idx),
                    "base_name": base_name,
                    "sigma": sigma,
                    "sample_idx": sample_idx,
                    "filename": filename,
                })

                print(f"    样本{sample_idx}: {filename}")

    # 创建汇总网格图
    print("\n创建汇总网格图...")
    create_summary_grid(output_dir, all_results, CONFIG)

    # 保存结果
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("邻域采样实验完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"生成图片数: {len(all_results)}")
    print("\n请查看生成的图片:")
    print("  - σ=0.01: 几乎原图，细节略有变化")
    print("  - σ=0.03-0.05: 新纹理，仍然合理")
    print("  - σ=0.1: 可能是最佳区域")
    print("  - σ=0.2: 可能开始崩坏")
    print("=" * 60)


def create_summary_grid(output_dir, results, config):
    """创建汇总网格图"""
    # 按基础样本和sigma组织
    organized = {}
    for r in results:
        key = (r["base_idx"], r["sigma"])
        if key not in organized:
            organized[key] = []
        organized[key].append(r)

    # 计算布局
    num_base = config["num_base_samples"]
    num_sigma = len(config["sigma_values"])
    samples_per_sigma = config["samples_per_sigma"]

    # 每个基础样本一行，每个sigma一组
    cell_size = 64
    padding = 3
    group_padding = 10

    # 总尺寸
    row_height = cell_size + padding
    total_width = num_sigma * (samples_per_sigma * (cell_size + padding) + group_padding) + padding
    total_height = num_base * row_height + padding + 40

    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), "Neighbor Sampling Results", fill="white")

    # 绘制图片
    for base_idx in range(num_base):
        for sigma_idx, sigma in enumerate(config["sigma_values"]):
            key = (base_idx, sigma)

            if key not in organized:
                continue

            samples = organized[key]

            for sample_idx, sample in enumerate(samples[:samples_per_sigma]):
                # 计算位置
                x = padding + sigma_idx * (samples_per_sigma * (cell_size + padding) + group_padding) + sample_idx * (cell_size + padding)
                y = 40 + base_idx * row_height

                # 加载并粘贴图片
                img_path = output_dir / sample["filename"]
                if img_path.exists():
                    img = Image.open(img_path)
                    img_small = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                    grid.paste(img_small, (x, y))

            # 绘制sigma标签
            x_label = padding + sigma_idx * (samples_per_sigma * (cell_size + padding) + group_padding)
            draw.text((x_label, 40 + base_idx * row_height + cell_size + 2), f"σ={sigma}", fill="gray")

    # 保存
    grid.save(output_dir / "summary_grid.png")
    print(f"汇总网格图已保存: {output_dir / 'summary_grid.png'}")


if __name__ == "__main__":
    main()

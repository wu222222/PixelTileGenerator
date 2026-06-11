"""
PCA重建验证

功能:
- 对latent进行PCA压缩
- 逆PCA重建
- 用Decoder生成图片
- 与原图对比
"""

import sys
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v5/vqvae_v5_best.pth",
    "data_dir": "datasets/classified/pixel_32_quantized",
    "latent_data": "datasets/vqvae_latent_data",
    "output_dir": "checkpoints/vqvae_v5/pca_reconstruction",
    "n_components": 186,  # 95%方差
    "num_samples": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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
    print("PCA重建验证")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    latent_data_path = project_root / CONFIG["latent_data"]
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

    # 加载latent数据
    print(f"加载latent数据: {latent_data_path}")
    latents = np.load(latent_data_path / "latents.npy")
    print(f"Latent形状: {latents.shape}")

    # 展平latent
    N = latents.shape[0]
    latents_flat = latents.reshape(N, -1)
    print(f"展平后: {latents_flat.shape}")

    # 训练PCA
    print(f"\n训练PCA (n_components={CONFIG['n_components']})...")
    pca = PCA(n_components=CONFIG["n_components"])
    latents_pca = pca.fit_transform(latents_flat)

    print(f"PCA后形状: {latents_pca.shape}")
    print(f"解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    # 选择样本进行重建验证
    sample_indices = np.random.choice(N, CONFIG["num_samples"], replace=False)

    # 创建对比图
    print(f"\n生成 {CONFIG['num_samples']} 个重建样本...")
    cell_size = 128
    padding = 5
    num_samples = CONFIG["num_samples"]

    # 总宽度: 原图 + PCA重建图 + 差异图
    total_width = cell_size * 3 + padding * 4
    total_height = num_samples * (cell_size + padding) + padding + 40

    comparison = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(comparison)

    # 标题
    draw.text((padding, 10), "Original", fill="white")
    draw.text((cell_size + padding * 2, 10), f"PCA Reconstruction ({CONFIG['n_components']}D)", fill="white")
    draw.text((cell_size * 2 + padding * 3, 10), "Difference", fill="white")

    for i, idx in enumerate(sample_indices):
        # 获取原始latent
        z_original = latents[idx]
        z_flat = latents_flat[idx:idx+1]

        # PCA压缩和重建
        z_pca = pca.transform(z_flat)
        z_reconstructed = pca.inverse_transform(z_pca)

        # reshape回原始形状
        z_recon = z_reconstructed.reshape(z_original.shape)

        # 转换为tensor
        z_orig_tensor = torch.FloatTensor(z_original).unsqueeze(0).to(CONFIG["device"])
        z_recon_tensor = torch.FloatTensor(z_recon).unsqueeze(0).to(CONFIG["device"])

        # 用Decoder生成图片
        with torch.no_grad():
            img_orig = vqvae.decode(z_orig_tensor)
            img_recon = vqvae.decode(z_recon_tensor)

        # 转换为PIL图片
        img_orig_pil = transforms.ToPILImage()(img_orig.squeeze(0).cpu())
        img_recon_pil = transforms.ToPILImage()(img_recon.squeeze(0).cpu())

        # 量化
        img_orig_quantized = quantize_image(img_orig_pil, colors=32)
        img_recon_quantized = quantize_image(img_recon_pil, colors=32)

        # 创建差异图
        orig_arr = np.array(img_orig_quantized).astype(float)
        recon_arr = np.array(img_recon_quantized).astype(float)
        diff_arr = np.abs(orig_arr - recon_arr).astype(np.uint8)
        diff_img = Image.fromarray(diff_arr)

        # 放大
        img_orig_large = img_orig_quantized.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        img_recon_large = img_recon_quantized.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        diff_large = diff_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        # 粘贴到对比图
        y = 40 + i * (cell_size + padding)
        comparison.paste(img_orig_large, (padding, y))
        comparison.paste(img_recon_large, (cell_size + padding * 2, y))
        comparison.paste(diff_large, (cell_size * 2 + padding * 3, y))

    # 保存对比图
    comparison.save(output_dir / "pca_reconstruction_comparison.png")
    print(f"\n对比图已保存: {output_dir / 'pca_reconstruction_comparison.png'}")

    # 保存单张图片
    for i, idx in enumerate(sample_indices):
        z_flat = latents_flat[idx:idx+1]
        z_pca = pca.transform(z_flat)
        z_reconstructed = pca.inverse_transform(z_pca)
        z_recon = z_reconstructed.reshape(latents[idx].shape)

        z_recon_tensor = torch.FloatTensor(z_recon).unsqueeze(0).to(CONFIG["device"])
        with torch.no_grad():
            img_recon = vqvae.decode(z_recon_tensor)

        img_pil = transforms.ToPILImage()(img_recon.squeeze(0).cpu())
        img_quantized = quantize_image(img_pil, colors=32)
        img_quantized.save(output_dir / f"pca_recon_{i:02d}.png")

    # 保存PCA模型
    import pickle
    with open(output_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    # 打印总结
    print("\n" + "=" * 60)
    print("PCA重建验证完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n请查看对比图:")
    print("  - 如果重建图和原图几乎一样 → PCA压缩成功")
    print("  - 如果重建图明显变糊 → 需要增加n_components")
    print("=" * 60)


if __name__ == "__main__":
    main()

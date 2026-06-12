"""
提取VQ-VAE v6 latent codes

功能:
- 加载训练好的VQ-VAE v6 (20×20 latent)
- 提取所有图片的latent codes
- 保存为训练数据
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2


# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v6/vqvae_v6_best.pth",
    "data_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "datasets/vqvae_latent_data_v6",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    print("=" * 60)
    print("提取VQ-VAE v6 Latent Codes")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    data_dir = project_root / CONFIG["data_dir"]
    output_dir = project_root / CONFIG["output_dir"]

    if not checkpoint_path.exists():
        print(f"[错误] VQ-VAE模型不存在: {checkpoint_path}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载VQ-VAE
    print(f"加载VQ-VAE: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})
    latent_size = config.get("latent_size", 20)

    print(f"Latent大小: {latent_size}×{latent_size}")

    vqvae = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 512),
        latent_size=latent_size,
    ).to(CONFIG["device"])

    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    print(f"模型加载成功")

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"找到 {len(image_files)} 张图片")

    # 数据变换
    transform = transforms.Compose([transforms.ToTensor()])

    # 提取latent codes
    print("\n提取latent codes...")
    all_indices = []
    all_latents = []
    all_names = []

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                # 加载图片
                img = Image.open(img_path).convert("RGBA")
                img_tensor = transform(img).unsqueeze(0).to(CONFIG["device"])

                # 提取latent
                z_q, indices = vqvae.encode(img_tensor)

                # reshape indices: [batch * h * w] -> [batch, h, w]
                indices_reshaped = indices.view(1, latent_size, latent_size)

                # 保存indices (离散索引)
                all_indices.append(indices_reshaped.cpu().numpy())

                # 保存z_q (量化后的向量)
                all_latents.append(z_q.cpu().numpy())

                # 保存文件名
                all_names.append(img_path.name)

                if (i + 1) % 50 == 0:
                    print(f"  进度: {i+1}/{len(image_files)}")

            except Exception as e:
                print(f"  [错误] {img_path.name}: {e}")

    # 合并所有数据
    all_indices = np.concatenate(all_indices, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    print(f"\nIndices形状: {all_indices.shape}")
    print(f"Latent codes形状: {all_latents.shape}")

    # 保存
    np.save(output_dir / "indices.npy", all_indices)
    np.save(output_dir / "latents.npy", all_latents)

    with open(output_dir / "names.json", "w") as f:
        json.dump(all_names, f)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "vqvae_checkpoint": str(checkpoint_path),
            "num_samples": len(all_indices),
            "indices_shape": list(all_indices.shape[1:]),
            "latent_shape": list(all_latents.shape[1:]),
            "latent_size": latent_size,
            "source": str(data_dir),
        }, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("提取完成!")
    print("=" * 60)
    print(f"Latent codes形状: {all_latents.shape}")
    print(f"输出目录: {output_dir}")
    print(f"文件:")
    print(f"  - latents.npy (latent codes)")
    print(f"  - names.json (文件名)")
    print(f"  - config.json (配置)")
    print("=" * 60)


if __name__ == "__main__":
    main()

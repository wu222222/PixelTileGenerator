"""
提取 VQ-VAE v9 latent codes

功能:
- 加载训练好的 VQ-VAEv4 (v9 checkpoint)
- 提取所有 64x64 图片的离散 code indices
- 保存为 VQ Code GAN 训练数据
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v4 import VQVAEv4


CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v9/vqvae_v9_best.pth",
    "data_dir": "datasets/classified/pixel_64_quantized",
    "output_dir": "datasets/vqvae_v9_latent_data",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    print("=" * 60)
    print("提取 VQ-VAE v9 Latent Codes")
    print("=" * 60)

    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    data_dir = project_root / CONFIG["data_dir"]
    output_dir = project_root / CONFIG["output_dir"]

    if not checkpoint_path.exists():
        print(f"[错误] VQ-VAE模型不存在: {checkpoint_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 VQ-VAE
    print(f"加载 VQ-VAE: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"], weights_only=False)
    config = checkpoint.get("config", {})

    vqvae = VQVAEv4(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 128),
        num_embeddings=config.get("num_embeddings", 1024),
        latent_size=config.get("latent_size", 16),
    ).to(CONFIG["device"])

    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    print(f"模型加载成功 (epoch {checkpoint.get('epoch', '?')})")

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = sorted([f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    print(f"找到 {len(image_files)} 张图片")

    transform = transforms.Compose([transforms.ToTensor()])

    # 提取 indices
    print("\n提取 latent codes...")
    all_indices = []
    all_names = []

    latent_size = config.get("latent_size", 16)

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path).convert("RGBA")
                img_tensor = transform(img).unsqueeze(0).to(CONFIG["device"])

                _, indices = vqvae.encode(img_tensor)

                # [batch * h * w] -> [batch, h, w]
                indices_reshaped = indices.view(1, latent_size, latent_size)
                all_indices.append(indices_reshaped.cpu().numpy())
                all_names.append(img_path.name)

                if (i + 1) % 100 == 0:
                    print(f"  进度: {i+1}/{len(image_files)}")

            except Exception as e:
                print(f"  [错误] {img_path.name}: {e}")

    all_indices = np.concatenate(all_indices, axis=0)
    print(f"\nIndices形状: {all_indices.shape}")

    # 统计码本使用情况
    unique_codes = np.unique(all_indices)
    num_embeddings = config.get("num_embeddings", 1024)
    print(f"使用的码字数: {len(unique_codes)} / {num_embeddings} ({len(unique_codes)/num_embeddings:.1%})")

    # 保存
    np.save(output_dir / "indices.npy", all_indices)

    with open(output_dir / "names.json", "w") as f:
        json.dump(all_names, f)

    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "vqvae_checkpoint": str(checkpoint_path),
            "num_samples": len(all_indices),
            "indices_shape": list(all_indices.shape[1:]),
            "num_embeddings": num_embeddings,
            "latent_size": latent_size,
            "source": str(data_dir),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("提取完成!")
    print("=" * 60)
    print(f"Indices形状: {all_indices.shape}")
    print(f"码本使用率: {len(unique_codes)}/{num_embeddings} ({len(unique_codes)/num_embeddings:.1%})")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

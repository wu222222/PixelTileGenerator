"""
提取latent vectors脚本

功能:
- 加载已训练的AutoEncoder
- 提取所有图片的latent z
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

from models.autoencoder_resnet import ResNetAutoEncoder


# 配置
CONFIG = {
    "checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",
    "data_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "datasets/latent_data",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    print("=" * 60)
    print("提取Latent Vectors")
    print("=" * 60)

    # 路径
    checkpoint_path = project_root / CONFIG["checkpoint"]
    data_dir = project_root / CONFIG["data_dir"]
    output_dir = project_root / CONFIG["output_dir"]

    if not checkpoint_path.exists():
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载AutoEncoder
    print(f"加载AutoEncoder: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})
    latent_dim = config.get("latent_dim", 128)

    autoencoder = ResNetAutoEncoder(latent_dim=latent_dim).to(CONFIG["device"])
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    print(f"Latent维度: {latent_dim}")

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions and f.name != "quantize_status.json"]

    print(f"找到 {len(image_files)} 张图片")

    # 数据变换
    transform = transforms.Compose([transforms.ToTensor()])

    # 提取latent vectors
    print("\n提取latent vectors...")
    latents = []
    names = []

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                # 加载图片
                img = Image.open(img_path).convert("RGBA")
                img_tensor = transform(img).unsqueeze(0).to(CONFIG["device"])

                # 提取latent
                z = autoencoder.encode(img_tensor)
                latents.append(z.cpu().numpy().flatten())
                names.append(img_path.name)

                if (i + 1) % 50 == 0:
                    print(f"  进度: {i+1}/{len(image_files)}")

            except Exception as e:
                print(f"  [错误] {img_path.name}: {e}")

    # 转换为numpy数组
    latents = np.array(latents)
    print(f"\nLatent vectors形状: {latents.shape}")

    # 保存
    np.save(output_dir / "latents.npy", latents)

    with open(output_dir / "names.json", "w") as f:
        json.dump(names, f)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "latent_dim": latent_dim,
            "num_samples": len(latents),
            "source": str(data_dir),
        }, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("提取完成!")
    print("=" * 60)
    print(f"Latent vectors: {latents.shape}")
    print(f"输出目录: {output_dir}")
    print(f"文件:")
    print(f"  - latents.npy (latent vectors)")
    print(f"  - names.json (图片名称)")
    print(f"  - config.json (配置)")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
模型对比脚本

功能:
- 加载不同版本的模型
- 生成相同条件下的图片
- 创建对比图
"""

import sys
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.conditional_gan import ConditionalGenerator as CGAN_Generator
from models.conditional_wgan_gp import ConditionalGenerator as WGAN_Generator
from models.autoencoder_resnet import ResNetAutoEncoder


# 配置
CONFIG = {
    "labels_file": "datasets/classified/pixel_32_filtered/class_labels.json",
    "autoencoder_checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",
    "output_dir": "checkpoints/comparison",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_cgan_model(checkpoint_path, num_classes, device):
    """加载Conditional GAN模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # 加载AutoEncoder获取decoder
    ae_checkpoint = torch.load(
        project_root / config.get("autoencoder_checkpoint", CONFIG["autoencoder_checkpoint"]),
        map_location=device
    )
    ae_config = ae_checkpoint.get("config", {})
    autoencoder = ResNetAutoEncoder(latent_dim=ae_config.get("latent_dim", 128)).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])

    # 创建生成器
    generator = CGAN_Generator(
        latent_dim=config.get("latent_dim", 128),
        num_classes=num_classes,
        embed_dim=config.get("embed_dim", 32),
        pretrained_decoder=autoencoder.decoder,
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    return generator


def load_wgan_model(checkpoint_path, num_classes, device):
    """加载WGAN-GP模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # 加载AutoEncoder获取decoder
    ae_checkpoint = torch.load(
        project_root / config.get("autoencoder_checkpoint", CONFIG["autoencoder_checkpoint"]),
        map_location=device
    )
    ae_config = ae_checkpoint.get("config", {})
    autoencoder = ResNetAutoEncoder(latent_dim=ae_config.get("latent_dim", 128)).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])

    # 创建生成器
    generator = WGAN_Generator(
        latent_dim=config.get("latent_dim", 128),
        num_classes=num_classes,
        embed_dim=config.get("embed_dim", 32),
        pretrained_decoder=autoencoder.decoder,
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    return generator


def load_autoencoder_model(checkpoint_path, device):
    """加载AutoEncoder模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = ResNetAutoEncoder(latent_dim=config.get("latent_dim", 128)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_samples(generator, z, labels, device):
    """生成样本"""
    with torch.no_grad():
        z = z.to(device)
        labels = labels.to(device)
        return generator(z, labels)


def create_comparison(images_dict, category_names, save_path, scale=4):
    """创建对比图"""
    # 获取图片尺寸
    sample_img = list(images_dict.values())[0][0]
    img_size = sample_img.size(2)  # 假设是 [C, H, W]

    # 计算布局
    num_models = len(images_dict)
    num_categories = min(len(category_names), 8)
    num_samples = 2  # 每个类别显示2个样本

    # 计算对比图尺寸
    cell_width = img_size * scale
    cell_height = img_size * scale
    padding = 10
    label_height = 30

    total_width = num_models * (num_samples * cell_width + padding) + padding
    total_height = num_categories * (cell_height + label_height + padding) + padding

    # 创建对比图
    comparison = Image.new("RGB", (total_width, total_height), (40, 40, 40))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)

    # 绘制每个模型的结果
    for model_idx, (model_name, model_images) in enumerate(images_dict.items()):
        x_offset = padding + model_idx * (num_samples * cell_width + padding)

        # 绘制模型名称
        draw.text((x_offset + 10, 5), model_name, fill="white")

        # 绘制每个类别的样本
        for cat_idx in range(num_categories):
            y_offset = padding + cat_idx * (cell_height + label_height + padding) + label_height

            # 绘制类别名称
            cat_name = category_names[cat_idx] if cat_idx < len(category_names) else f"class_{cat_idx}"
            draw.text((x_offset + 10, y_offset - label_height + 5), cat_name, fill="white")

            # 绘制样本
            for sample_idx in range(num_samples):
                img_tensor = model_images[cat_idx * num_samples + sample_idx]
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                img_large = img_pil.resize((cell_width, cell_height), Image.Resampling.NEAREST)

                x = x_offset + sample_idx * cell_width
                comparison.paste(img_large, (x, y_offset))

    # 保存
    comparison.save(save_path)
    print(f"对比图已保存: {save_path}")


def main():
    print("=" * 60)
    print("模型对比")
    print("=" * 60)

    # 加载标签
    labels_path = project_root / CONFIG["labels_file"]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    category_names = sorted(labels_data["category_counts"].keys())
    num_classes = len(category_names)

    print(f"类别数: {num_classes}")
    print(f"设备: {CONFIG['device']}")

    # 创建输出目录
    output_dir = project_root / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成固定的噪声和标签
    torch.manual_seed(42)
    z = torch.randn(num_classes * 2, 128)  # 每个类别2个样本
    labels = torch.arange(num_classes).repeat(2)

    # 加载模型
    models = {}

    # AutoEncoder (重建)
    ae_path = project_root / CONFIG["autoencoder_checkpoint"]
    if ae_path.exists():
        print(f"\n加载AutoEncoder...")
        models["AutoEncoder"] = load_autoencoder_model(ae_path, CONFIG["device"])

    # Conditional GAN
    cgan_path = project_root / "checkpoints/conditional_gan/conditional_gan_best.pth"
    if cgan_path.exists():
        print(f"加载Conditional GAN...")
        models["CGAN"] = load_cgan_model(cgan_path, num_classes, CONFIG["device"])

    # WGAN-GP
    wgan_path = project_root / "checkpoints/conditional_wgan_gp/wgan_gp_best.pth"
    if wgan_path.exists():
        print(f"加载WGAN-GP...")
        models["WGAN-GP"] = load_wgan_model(wgan_path, num_classes, CONFIG["device"])

    if not models:
        print("[错误] 没有找到模型文件")
        return

    # 生成样本
    print(f"\n生成样本...")
    images_dict = {}

    for model_name, model in models.items():
        print(f"  {model_name}...")

        if model_name == "AutoEncoder":
            # AutoEncoder需要真实图片作为输入
            # 这里简化处理，只显示生成的图片
            continue
        else:
            # GAN模型
            samples = generate_samples(model, z, labels, CONFIG["device"])
            images_dict[model_name] = [samples[i] for i in range(len(samples))]

    # 创建对比图
    if images_dict:
        save_path = output_dir / "model_comparison.png"
        create_comparison(images_dict, category_names, save_path)

    print("\n" + "=" * 60)
    print("对比完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()

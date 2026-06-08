"""
自动分类脚本

功能:
- 用文件名提取已有类别
- 对unknown图片用AutoEncoder latent space做K-means聚类
- 用轮廓系数选择最佳聚类数目
- 生成类别标签文件
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.autoencoder_resnet import ResNetAutoEncoder


# 配置
CONFIG = {
    "data_dir": "datasets/classified/pixel_32_quantized",
    "checkpoint": "checkpoints/autoencoder_resnet/autoencoder_best.pth",
    "output_file": "datasets/classified/pixel_32_quantized/class_labels.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 聚类参数
    "min_clusters": 3,
    "max_clusters": 20,
    "unknown_cluster_count": 5,  # unknown图片分成5类
}


def extract_class_from_filename(filename):
    """从文件名提取类别"""
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        category = parts[0].lower()
        # 如果类别太短或太长，合并前两个部分
        if len(category) < 3 and len(parts) > 2:
            category = "_".join(parts[:2]).lower()
        return category
    return "unknown"


def load_model(checkpoint_path, device):
    """加载AutoEncoder模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    latent_dim = config.get("latent_dim", 128)

    model = ResNetAutoEncoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, latent_dim


def extract_latent_vectors(model, image_files, device):
    """提取所有图片的latent vectors"""
    transform = transforms.Compose([transforms.ToTensor()])
    latents = []

    print(f"提取 {len(image_files)} 张图片的latent vectors...")

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path).convert("RGBA")
                img_tensor = transform(img).unsqueeze(0).to(device)
                z = model.encode(img_tensor)
                latents.append(z.cpu().numpy().flatten())

                if (i + 1) % 500 == 0:
                    print(f"  进度: {i+1}/{len(image_files)}")

            except Exception as e:
                print(f"  [错误] {img_path.name}: {e}")
                latents.append(np.zeros(model.latent_dim))

    return np.array(latents)


def find_best_clusters(latents, min_k=3, max_k=20):
    """用轮廓系数找最佳聚类数目"""
    print(f"\n寻找最佳聚类数目 (k={min_k}~{max_k})...")

    silhouette_scores = {}

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latents)
        score = silhouette_score(latents, labels)
        silhouette_scores[k] = score
        print(f"  k={k}: 轮廓系数={score:.4f}")

    # 找最佳k
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]

    print(f"\n最佳聚类数目: k={best_k} (轮廓系数={best_score:.4f})")

    return best_k, silhouette_scores


def cluster_unknown_images(latents, n_clusters):
    """对unknown图片做K-means聚类"""
    print(f"\n对unknown图片做K-means聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latents)
    return labels, kmeans


def main():
    print("=" * 60)
    print("自动分类脚本")
    print("=" * 60)

    # 路径
    data_dir = project_root / CONFIG["data_dir"]
    checkpoint_path = project_root / CONFIG["checkpoint"]
    output_path = project_root / CONFIG["output_file"]

    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        return

    if not checkpoint_path.exists():
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        return

    # 获取所有图片
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"找到 {len(image_files)} 张图片")

    # 步骤1: 从文件名提取类别
    print("\n" + "=" * 60)
    print("步骤1: 从文件名提取类别")
    print("=" * 60)

    filename_categories = {}
    category_counts = defaultdict(int)
    unknown_files = []

    for img_path in image_files:
        category = extract_class_from_filename(img_path.name)
        filename_categories[img_path.name] = category
        category_counts[category] += 1

        if category == "unknown":
            unknown_files.append(img_path)

    print(f"\n文件名类别统计:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {cat}: {count}")

    print(f"\nunknown图片: {len(unknown_files)} 张")

    # 步骤2: 对unknown图片做聚类
    final_categories = dict(filename_categories)

    if len(unknown_files) > 0:
        print("\n" + "=" * 60)
        print("步骤2: 对unknown图片做聚类")
        print("=" * 60)

        # 加载模型
        model, latent_dim = load_model(checkpoint_path, CONFIG["device"])
        print(f"模型加载成功 (latent_dim={latent_dim})")

        # 提取unknown图片的latent vectors
        unknown_latents = extract_latent_vectors(model, unknown_files, CONFIG["device"])

        # 用轮廓系数找最佳聚类数目
        best_k, silhouette_scores = find_best_clusters(
            unknown_latents,
            CONFIG["min_clusters"],
            min(CONFIG["max_clusters"], len(unknown_files) - 1)
        )

        # 对unknown做聚类
        cluster_labels, kmeans = cluster_unknown_images(unknown_latents, CONFIG["unknown_cluster_count"])

        # 更新类别
        for i, img_path in enumerate(unknown_files):
            final_categories[img_path.name] = f"unknown_cluster_{cluster_labels[i]}"

    # 步骤3: 保存结果
    print("\n" + "=" * 60)
    print("步骤3: 保存结果")
    print("=" * 60)

    # 统计最终类别
    final_category_counts = defaultdict(int)
    for cat in final_categories.values():
        final_category_counts[cat] += 1

    print(f"\n最终类别统计:")
    for cat, count in sorted(final_category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    # 保存到JSON
    output_data = {
        "categories": final_categories,
        "category_counts": dict(final_category_counts),
        "total_images": len(image_files),
        "total_categories": len(final_category_counts),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n类别标签已保存: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

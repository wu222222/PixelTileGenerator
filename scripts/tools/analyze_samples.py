"""分析AutoEncoder样本"""

from PIL import Image
import numpy as np
from pathlib import Path

# 检查样本图片
samples_dir = Path("checkpoints/autoencoder_v4/samples")
sample_files = sorted(list(samples_dir.glob("epoch_200_sample_*.png")))

print("分析epoch_200样本:")
for img_path in sample_files:
    img = Image.open(img_path)
    arr = np.array(img)

    # 计算统计信息
    unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    mean_val = arr.mean()
    std_val = arr.std()

    print(f"  {img_path.name}:")
    print(f"    尺寸: {img.size}")
    print(f"    唯一颜色数: {unique_colors}")
    print(f"    均值: {mean_val:.2f}")
    print(f"    标准差: {std_val:.2f}")

# 对比原始数据
print("\n对比原始数据:")
data_dir = Path("datasets/classified/pixel_32_quantized")
original_files = sorted(list(data_dir.glob("*.png")))[:8]

for img_path in original_files:
    img = Image.open(img_path)
    arr = np.array(img)

    unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    mean_val = arr.mean()
    std_val = arr.std()

    print(f"  {img_path.name}:")
    print(f"    唯一颜色数: {unique_colors}")
    print(f"    均值: {mean_val:.2f}")
    print(f"    标准差: {std_val:.2f}")

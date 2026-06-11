"""分析indices数据分布"""

import numpy as np
from collections import Counter

# 加载indices
indices = np.load('datasets/vqvae_latent_data/indices.npy')

print("=== Indices分析 ===")
print(f"形状: {indices.shape}")
print(f"数据类型: {indices.dtype}")

# 展平所有indices
flat_indices = indices.flatten()

# 统计每个code的使用频率
counter = Counter(flat_indices)
total = len(flat_indices)

print(f"\n总tokens数: {total}")
print(f"唯一codes数: {len(counter)}")

# 显示前20个最常用的codes
print(f"\n前20个最常用的codes:")
for code, count in counter.most_common(20):
    percentage = count / total * 100
    print(f"  Code {code:3d}: {count:6d} ({percentage:.2f}%)")

# 检查是否有完全未使用的codes
used_codes = set(counter.keys())
all_codes = set(range(256))
unused_codes = all_codes - used_codes
print(f"\n未使用的codes: {len(unused_codes)}")
if unused_codes:
    print(f"  {sorted(unused_codes)[:20]}...")

# 检查每张图片的indices统计
print(f"\n每张图片的indices统计:")
print(f"  最小值: {indices.min()}")
print(f"  最大值: {indices.max()}")
print(f"  平均值: {indices.mean():.2f}")
print(f"  标准差: {indices.std():.2f}")

# 检查每张图片的唯一codes数
unique_per_image = [len(np.unique(img)) for img in indices]
print(f"\n每张图片的唯一codes数:")
print(f"  最小: {min(unique_per_image)}")
print(f"  最大: {max(unique_per_image)}")
print(f"  平均: {sum(unique_per_image) / len(unique_per_image):.1f}")

"""检查数据集"""

from PIL import Image
import numpy as np
from pathlib import Path

# 检查几张图片的颜色分布
data_dir = Path("datasets/classified/pixel_32_quantized")
image_files = sorted(list(data_dir.glob("*.png")))[:10]

print("检查数据集图片:")
for img_path in image_files:
    img = Image.open(img_path)
    arr = np.array(img)
    unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    print(f"  {img_path.name}: 模式={img.mode}, 尺寸={img.size}, 唯一颜色数={unique_colors}")

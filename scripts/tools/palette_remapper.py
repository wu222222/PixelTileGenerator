"""
调色板重映射工具

将已量化的纹理从一个调色板映射到另一个调色板。
基于亮度(luminance)进行排序匹配，保留结构明暗关系。
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
PALETTES_DIR = PROJECT_ROOT / "palettes"


def hex_to_rgba(hex_color: str) -> tuple:
    """#RRGGBB → (R, G, B, 255)"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, 255)


def luminance(color: tuple) -> float:
    """计算颜色亮度 (ITU-R BT.601)"""
    if len(color) >= 4 and color[3] == 0:
        return -1  # 完全透明
    r, g, b = color[0], color[1], color[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def load_palette(name: str) -> dict:
    """加载调色板 JSON 文件"""
    path = PALETTES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"调色板不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_palettes() -> list:
    """列出所有可用调色板"""
    palettes = []
    for f in sorted(PALETTES_DIR.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        palettes.append({
            "id": f.stem,
            "name": data.get("name", f.stem),
            "description": data.get("description", ""),
        })
    return palettes


def remap_palette(
    img: Image.Image,
    target_palette_name: str,
    source_palette_name: str = None,
) -> Image.Image:
    """
    将量化图像的颜色映射到目标调色板。

    算法:
    1. 提取源图像的唯一颜色，按亮度排序
    2. 加载目标调色板，按亮度排序
    3. 亮度排名一一映射

    Args:
        img: 已量化的 RGBA 图像
        target_palette_name: 目标调色板名称
        source_palette_name: 源调色板名称（可选，默认自动提取）

    Returns:
        重映射后的 RGBA 图像
    """
    img = img.convert("RGBA")
    pixels = np.array(img)
    h, w, c = pixels.shape

    # 提取源图像的唯一颜色
    flat = pixels.reshape(-1, 4)
    unique_colors = np.unique(flat, axis=0)
    unique_colors_list = [tuple(c) for c in unique_colors]

    # 按亮度排序源颜色
    source_sorted = sorted(unique_colors_list, key=luminance)

    # 加载目标调色板
    target_data = load_palette(target_palette_name)
    target_colors = [hex_to_rgba(c) for c in target_data["colors"]]
    target_sorted = sorted(target_colors, key=luminance)

    # 亮度排名映射: 源排名 → 目标颜色
    n_source = len(source_sorted)
    n_target = len(target_sorted)

    # 创建映射表: source_color → target_color
    color_map = {}
    for i, src_color in enumerate(source_sorted):
        # 映射到目标调色板的对应排名（线性插值）
        if n_source == 1:
            target_idx = 0
        else:
            target_idx = int(i * (n_target - 1) / (n_source - 1))
        target_idx = min(target_idx, n_target - 1)
        color_map[src_color] = target_sorted[target_idx]

    # 应用映射
    result = np.zeros_like(pixels)
    for y in range(h):
        for x in range(w):
            src = tuple(pixels[y, x])
            if src in color_map:
                result[y, x] = color_map[src]
            else:
                result[y, x] = src  # 未找到映射，保持原色

    return Image.fromarray(result, "RGBA")


def _rgb_to_hsl(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """RGB [0,255] → H [0,360], S [0,1], L [0,1] (向量化)"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Lightness
    l = (max_c + min_c) / 2.0

    # Saturation
    s = np.where(delta == 0, 0.0,
                 delta / (1.0 - np.abs(2.0 * l - 1.0) + 1e-10))
    s = np.clip(s, 0, 1)

    # Hue
    h = np.zeros_like(r)
    mask = delta > 0
    idx = mask & (max_c == r)
    h[idx] = 60.0 * (((g[idx] - b[idx]) / delta[idx]) % 6)
    idx = mask & (max_c == g)
    h[idx] = 60.0 * ((b[idx] - r[idx]) / delta[idx] + 2)
    idx = mask & (max_c == b)
    h[idx] = 60.0 * ((r[idx] - g[idx]) / delta[idx] + 4)

    return h, s, l


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray):
    """H [0,360], S [0,1], L [0,1] → RGB [0,255] (向量化)"""
    h = h % 360
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - np.abs((h / 60.0) % 2 - 1.0))
    m = l - c / 2.0

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    idx = (h < 60)
    r[idx], g[idx], b[idx] = c[idx], x[idx], 0
    idx = (h >= 60) & (h < 120)
    r[idx], g[idx], b[idx] = x[idx], c[idx], 0
    idx = (h >= 120) & (h < 180)
    r[idx], g[idx], b[idx] = 0, c[idx], x[idx]
    idx = (h >= 180) & (h < 240)
    r[idx], g[idx], b[idx] = 0, x[idx], c[idx]
    idx = (h >= 240) & (h < 300)
    r[idx], g[idx], b[idx] = x[idx], 0, c[idx]
    idx = (h >= 300)
    r[idx], g[idx], b[idx] = c[idx], 0, x[idx]

    r = np.clip(((r + m) * 255).round(), 0, 255).astype(np.uint8)
    g = np.clip(((g + m) * 255).round(), 0, 255).astype(np.uint8)
    b = np.clip(((b + m) * 255).round(), 0, 255).astype(np.uint8)

    return r, g, b


def remap_palette_fast(
    img: Image.Image,
    target_palette_name: str,
    intensity: float = 1.0,
    saturation: float = 1.0,
) -> Image.Image:
    """
    快速版本：使用 numpy 向量化操作。

    Args:
        img: RGBA 图像
        target_palette_name: 目标调色板名称
        intensity: 调色板强度 0.0(原始)~1.0(完全替换)
        saturation: 饱和度倍率 0.0(灰度)~1.0(不变)~2.0(增强)
    """
    img = img.convert("RGBA")
    pixels = np.array(img, dtype=np.float64)
    h, w, c = pixels.shape

    # 保存原始像素
    original = pixels.copy()

    # 提取唯一颜色 (用于 LUT 映射)
    flat_uint8 = pixels.astype(np.uint8).reshape(-1, 4)
    unique_colors, inverse_indices = np.unique(flat_uint8, axis=0, return_inverse=True)
    unique_list = [tuple(c) for c in unique_colors]

    # 按亮度排序
    source_sorted = sorted(unique_list, key=luminance)
    source_rank = {color: i for i, color in enumerate(source_sorted)}

    # 加载目标调色板
    target_data = load_palette(target_palette_name)
    target_colors = [hex_to_rgba(c) for c in target_data["colors"]]
    target_sorted = sorted(target_colors, key=luminance)

    n_source = len(source_sorted)
    n_target = len(target_sorted)

    # 构建查找表 (LUT)
    lut = np.zeros((n_source, 4), dtype=np.uint8)
    for i, src_color in enumerate(source_sorted):
        if n_source == 1:
            target_idx = 0
        else:
            target_idx = int(i * (n_target - 1) / (n_source - 1))
        target_idx = min(target_idx, n_target - 1)
        rank = source_rank[src_color]
        lut[rank] = target_sorted[target_idx]

    # 用 LUT 映射
    mapped_flat = lut[inverse_indices]
    mapped = mapped_flat.reshape(h, w, 4).astype(np.float64)

    # 强度混合: intensity=0 → original, intensity=1 → mapped
    blended = original * (1.0 - intensity) + mapped * intensity
    blended = np.clip(blended, 0, 255)

    # 饱和度调整
    if saturation != 1.0:
        r, g, b = blended[:, :, 0], blended[:, :, 1], blended[:, :, 2]
        hsl_h, s, l = _rgb_to_hsl(r, g, b)
        s = np.clip(s * saturation, 0, 1)
        r2, g2, b2 = _hsl_to_rgb(hsl_h, s, l)
        blended[:, :, 0] = r2
        blended[:, :, 1] = g2
        blended[:, :, 2] = b2

    result = blended.astype(np.uint8)
    result[:, :, 3] = pixels[:, :, 3].astype(np.uint8)  # 保留原始 alpha

    return Image.fromarray(result, "RGBA")


# 便捷函数
def apply_palette(
    img: Image.Image,
    palette_name: str,
    intensity: float = 1.0,
    saturation: float = 1.0,
) -> Image.Image:
    """应用调色板到图像"""
    return remap_palette_fast(img, palette_name, intensity, saturation)


if __name__ == "__main__":
    # 测试
    print("可用调色板:")
    for p in list_palettes():
        print(f"  {p['id']:12s} - {p['name']:12s} - {p['description']}")

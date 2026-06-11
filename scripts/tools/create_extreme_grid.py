"""创建极端扰动对比图"""

from pathlib import Path
from PIL import Image, ImageDraw

# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "checkpoints/vqvae_v5/extreme_perturbation",
    "sigma_values": [0.1, 0.5, 1.0, 2.0, 5.0],
    "cell_size": 64,
}


def main():
    print("创建极端扰动对比图...")

    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]

    # 获取所有图片文件
    image_files = sorted(list(input_dir.glob("base*_sigma*.png")))

    if not image_files:
        print("[错误] 没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    # 解析文件名，按基础样本组织
    organized = {}
    for img_path in image_files:
        # 解析文件名: base227_sigma0.1.png
        parts = img_path.stem.split("_")
        base_idx = int(parts[0].replace("base", ""))
        sigma = float(parts[1].replace("sigma", ""))

        if base_idx not in organized:
            organized[base_idx] = {}
        organized[base_idx][sigma] = img_path

    # 获取唯一的base_idx
    base_indices = sorted(organized.keys())
    sigma_values = CONFIG["sigma_values"]

    print(f"基础样本数: {len(base_indices)}")
    print(f"扰动强度: {sigma_values}")

    # 计算布局
    cell_size = CONFIG["cell_size"]
    padding = 5

    num_base = len(base_indices)
    num_sigma = len(sigma_values)

    # 总尺寸
    total_width = num_sigma * (cell_size + padding) + padding
    total_height = num_base * (cell_size + padding) + padding + 40

    # 创建图片
    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), "Extreme Perturbation Test", fill="white")

    # 绘制图片
    for base_idx_idx, base_idx in enumerate(base_indices):
        for sigma_idx, sigma in enumerate(sigma_values):
            x = padding + sigma_idx * (cell_size + padding)
            y = 40 + base_idx_idx * (cell_size + padding)

            # 加载并粘贴图片
            if sigma in organized[base_idx]:
                img_path = organized[base_idx][sigma]
                try:
                    img = Image.open(img_path)
                    img_small = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                    grid.paste(img_small, (x, y))
                except Exception as e:
                    print(f"  [错误] 加载 {img_path.name}: {e}")

            # 绘制sigma标签
            draw.text((x, y + cell_size + 2), f"σ={sigma}", fill="gray")

    # 保存
    output_path = input_dir / "extreme_perturbation_grid.png"
    grid.save(output_path)
    print(f"对比图已保存: {output_path}")


if __name__ == "__main__":
    main()

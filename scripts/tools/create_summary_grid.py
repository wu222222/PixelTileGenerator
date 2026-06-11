"""创建邻域采样汇总网格图"""

from pathlib import Path
from PIL import Image, ImageDraw

# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "checkpoints/vqvae_v5/neighbor_sampling",
    "output_dir": "checkpoints/vqvae_v5/neighbor_sampling",
    "sigma_values": [0.01, 0.03, 0.05, 0.1, 0.2],
    "samples_per_sigma": 4,
    "cell_size": 64,
}


def main():
    print("创建汇总网格图...")

    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    output_dir = base_dir / CONFIG["output_dir"]

    # 获取所有图片文件
    image_files = sorted(list(input_dir.glob("base*_sigma*_sample*.png")))

    if not image_files:
        print("[错误] 没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    # 解析文件名，按基础样本和sigma组织
    organized = {}
    for img_path in image_files:
        # 解析文件名: base122_sigma0.01_sample00.png
        parts = img_path.stem.split("_")
        base_idx = int(parts[0].replace("base", ""))
        sigma = float(parts[1].replace("sigma", ""))
        sample_idx = int(parts[2].replace("sample", ""))

        key = (base_idx, sigma)
        if key not in organized:
            organized[key] = []
        organized[key].append({
            "path": img_path,
            "sample_idx": sample_idx,
        })

    # 获取唯一的base_idx和sigma值
    base_indices = sorted(set(k[0] for k in organized.keys()))
    sigma_values = CONFIG["sigma_values"]

    print(f"基础样本数: {len(base_indices)}")
    print(f"扰动强度: {sigma_values}")

    # 计算布局
    cell_size = CONFIG["cell_size"]
    padding = 3
    group_padding = 10
    samples_per_sigma = CONFIG["samples_per_sigma"]

    num_base = len(base_indices)
    num_sigma = len(sigma_values)

    # 总尺寸
    row_height = cell_size + padding + 15  # 15 for label
    total_width = num_sigma * (samples_per_sigma * (cell_size + padding) + group_padding) + padding
    total_height = num_base * row_height + padding + 40

    # 创建图片
    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), "Neighbor Sampling Results", fill="white")

    # 绘制图片
    for base_idx_idx, base_idx in enumerate(base_indices):
        for sigma_idx, sigma in enumerate(sigma_values):
            key = (base_idx, sigma)

            if key not in organized:
                continue

            samples = organized[key]

            for sample_idx, sample in enumerate(samples[:samples_per_sigma]):
                # 计算位置
                x = padding + sigma_idx * (samples_per_sigma * (cell_size + padding) + group_padding) + sample_idx * (cell_size + padding)
                y = 40 + base_idx_idx * row_height

                # 加载并粘贴图片
                try:
                    img = Image.open(sample["path"])
                    img_small = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                    grid.paste(img_small, (x, y))
                except Exception as e:
                    print(f"  [错误] 加载 {sample['path'].name}: {e}")

            # 绘制sigma标签
            x_label = padding + sigma_idx * (samples_per_sigma * (cell_size + padding) + group_padding)
            draw.text((x_label, 40 + base_idx_idx * row_height + cell_size + 2), f"σ={sigma}", fill="gray")

    # 保存
    output_path = output_dir / "summary_grid.png"
    grid.save(output_path)
    print(f"汇总网格图已保存: {output_path}")


if __name__ == "__main__":
    main()

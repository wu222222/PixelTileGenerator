"""
训练可视化脚本

功能:
- 生成checkpoint对比图（按时间轴）
- 生成训练曲线图（Loss, Usage等）
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "checkpoint_dir": "checkpoints/vqvae_v5",
    "output_dir": "checkpoints/vqvae_v5/visualization",
}


def create_timeline_grid(samples_dir, output_path, max_epochs=20):
    """创建时间轴对比图"""
    # 获取所有样本文件
    sample_files = sorted(list(samples_dir.glob("*_comparison_*.png")))

    if not sample_files:
        print("[警告] 没有找到样本文件")
        return

    # 提取epoch信息
    epochs = set()
    for f in sample_files:
        parts = f.stem.split("_")
        for i, part in enumerate(parts):
            if part == "epoch" and i + 1 < len(parts):
                try:
                    epochs.add(int(parts[i + 1]))
                except ValueError:
                    pass

    epochs = sorted(epochs)

    # 选择要显示的epoch
    if len(epochs) > max_epochs:
        step = len(epochs) // max_epochs
        selected_epochs = epochs[::step]
        if epochs[-1] not in selected_epochs:
            selected_epochs.append(epochs[-1])
    else:
        selected_epochs = epochs

    print(f"选择的epoch: {selected_epochs}")

    # 收集每个epoch的对比图（使用第一个样本）
    images = []
    for epoch in selected_epochs:
        pattern = f"epoch_{epoch:03d}_comparison_0.png"
        matches = list(samples_dir.glob(pattern))
        if matches:
            img = Image.open(matches[0])
            images.append((epoch, img))

    if not images:
        print("[警告] 没有找到对比图")
        return

    # 计算布局
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols

    # 图片尺寸
    img_width, img_height = images[0][1].size
    padding = 10
    label_height = 30

    # 总尺寸
    total_width = cols * (img_width + padding) + padding
    total_height = rows * (img_height + label_height + padding) + padding + 50

    # 创建图片
    timeline = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(timeline)

    # 标题
    draw.text((padding, 10), "Training Timeline (Original → Reconstruction)", fill="white")

    # 绘制图片
    for i, (epoch, img) in enumerate(images):
        row = i // cols
        col = i % cols

        x = padding + col * (img_width + padding)
        y = 50 + row * (img_height + label_height + padding)

        # 粘贴图片
        timeline.paste(img, (x, y))

        # 绘制epoch标签
        draw.text((x, y + img_height + 5), f"Epoch {epoch}", fill="white")

    # 保存
    timeline.save(output_path)
    print(f"时间轴对比图已保存: {output_path}")


def create_training_curves(history_path, output_dir):
    """创建训练曲线图"""
    # 加载训练历史
    with open(history_path, "r") as f:
        history = json.load(f)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("VQ-VAE Training Curves", fontsize=16)

    epochs = range(1, len(history["total_loss"]) + 1)

    # 总损失
    axes[0, 0].plot(epochs, history["total_loss"], label="Total Loss", color="blue")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 重建损失
    axes[0, 1].plot(epochs, history["recon_loss"], label="Recon Loss", color="green")
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # VQ损失
    axes[1, 0].plot(epochs, history["vq_loss"], label="VQ Loss", color="red")
    axes[1, 0].set_title("VQ Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 码本使用率
    if "codebook_usage" in history:
        axes[1, 1].plot(epochs, history["codebook_usage"], label="Codebook Usage", color="purple")
        axes[1, 1].set_title("Codebook Usage")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Usage (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No usage data", ha="center", va="center", fontsize=12)
        axes[1, 1].set_title("Codebook Usage")

    plt.tight_layout()

    # 保存
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线图已保存: {output_path}")


def create_comparison_grid(samples_dir, output_path, epoch=-1):
    """创建某个epoch的对比网格（所有样本）"""
    # 获取指定epoch的所有对比图
    if epoch == -1:
        # 使用最后一个epoch
        all_files = sorted(list(samples_dir.glob("*_comparison_*.png")))
        if not all_files:
            print("[警告] 没有找到对比图")
            return

        # 提取最大epoch
        max_epoch = 0
        for f in all_files:
            parts = f.stem.split("_")
            for i, part in enumerate(parts):
                if part == "epoch" and i + 1 < len(parts):
                    try:
                        max_epoch = max(max_epoch, int(parts[i + 1]))
                    except ValueError:
                        pass
        epoch = max_epoch

    # 获取该epoch的所有样本
    pattern = f"epoch_{epoch:03d}_comparison_*.png"
    sample_files = sorted(list(samples_dir.glob(pattern)))

    if not sample_files:
        print(f"[警告] 没有找到epoch {epoch}的对比图")
        return

    print(f"找到 {len(sample_files)} 个样本")

    # 加载图片
    images = []
    for f in sample_files:
        img = Image.open(f)
        images.append(img)

    # 计算布局
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    # 图片尺寸
    img_width, img_height = images[0].size
    padding = 5

    # 总尺寸
    total_width = cols * (img_width + padding) + padding
    total_height = rows * (img_height + padding) + padding + 40

    # 创建图片
    grid = Image.new("RGB", (total_width, total_height), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # 标题
    draw.text((padding, 10), f"Epoch {epoch} - All Samples (Original → Reconstruction)", fill="white")

    # 绘制图片
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = padding + col * (img_width + padding)
        y = 40 + row * (img_height + padding)

        grid.paste(img, (x, y))

    # 保存
    grid.save(output_path)
    print(f"对比网格图已保存: {output_path}")


def main():
    print("=" * 60)
    print("训练可视化")
    print("=" * 60)

    # 路径
    base_dir = CONFIG["project_root"]
    checkpoint_dir = base_dir / CONFIG["checkpoint_dir"]
    output_dir = base_dir / CONFIG["output_dir"]
    samples_dir = checkpoint_dir / "samples"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查目录
    if not checkpoint_dir.exists():
        print(f"[错误] 检查点目录不存在: {checkpoint_dir}")
        return

    # 1. 创建时间轴对比图
    print("\n1. 创建时间轴对比图...")
    create_timeline_grid(samples_dir, output_dir / "timeline.png")

    # 2. 创建训练曲线图
    print("\n2. 创建训练曲线图...")
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        create_training_curves(history_path, output_dir)
    else:
        print("[警告] 没有找到训练历史文件")

    # 3. 创建最终epoch的对比网格
    print("\n3. 创建最终epoch对比网格...")
    create_comparison_grid(samples_dir, output_dir / "final_comparison.png")

    # 打印总结
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n生成的文件:")
    print(f"  - timeline.png        (训练时间轴)")
    print(f"  - training_curves.png (训练曲线)")
    print(f"  - final_comparison.png (最终对比)")
    print("=" * 60)


if __name__ == "__main__":
    main()

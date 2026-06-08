"""
筛除pixel_32目录中非32x32的瓦片

功能:
- 扫描pixel_32目录
- 检查每张图片的尺寸
- 移动非32x32的图片到rejected目录
- 生成统计报告
"""

import time
import shutil
import json
from pathlib import Path
from PIL import Image


# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent,
    "input_dir": "datasets/classified/pixel_32",
    "rejected_dir": "datasets/classified/rejected",
    "target_size": (32, 32),
    "max_retries": 3,
    "retry_delay": 1,  # 秒
}


def filter_images():
    """筛选非32x32的图片"""
    base_dir = CONFIG["project_root"]
    input_dir = base_dir / CONFIG["input_dir"]
    rejected_dir = base_dir / CONFIG["rejected_dir"]

    if not input_dir.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    # 创建rejected目录
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": 0,
        "valid": 0,
        "rejected": 0,
        "errors": 0,
    }
    rejected_details = []
    error_files = []

    # 图片扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    # 扫描所有图片
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    stats["total"] = len(image_files)

    print(f"扫描完成: 找到 {len(image_files)} 个图片")
    print(f"目标尺寸: {CONFIG['target_size'][0]}×{CONFIG['target_size'][1]}")
    print("=" * 60)

    for img_path in image_files:
        success = False

        for attempt in range(CONFIG["max_retries"]):
            try:
                # 使用with语句确保文件正确关闭
                with Image.open(img_path) as img:
                    width, height = img.size
                    # 复制图片数据，避免文件句柄问题
                    img_copy = img.copy()

                if (width, height) == CONFIG["target_size"]:
                    stats["valid"] += 1
                else:
                    # 移动到rejected目录
                    dest = rejected_dir / img_path.name
                    # 如果目标已存在，添加后缀
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = rejected_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    shutil.move(str(img_path), str(dest))
                    stats["rejected"] += 1
                    rejected_details.append({
                        "name": img_path.name,
                        "size": f"{width}×{height}",
                    })
                    print(f"  [移除] {img_path.name} ({width}×{height})")

                success = True
                break

            except PermissionError as e:
                if attempt < CONFIG["max_retries"] - 1:
                    print(f"  [重试] {img_path.name} (尝试 {attempt + 1}/{CONFIG['max_retries']})")
                    time.sleep(CONFIG["retry_delay"])
                else:
                    stats["errors"] += 1
                    error_files.append(img_path.name)
                    print(f"  [错误] {img_path.name}: 文件被占用")

            except Exception as e:
                stats["errors"] += 1
                error_files.append(img_path.name)
                print(f"  [错误] {img_path.name}: {e}")
                break

    # 打印报告
    print("\n" + "=" * 60)
    print("筛选完成!")
    print("=" * 60)
    print(f"总计: {stats['total']}")
    print(f"有效 (32×32): {stats['valid']}")
    print(f"已移除: {stats['rejected']}")
    print(f"错误: {stats['errors']}")
    print("=" * 60)

    if error_files:
        print("\n错误文件列表:")
        for f in error_files[:20]:
            print(f"  - {f}")
        if len(error_files) > 20:
            print(f"  ... 还有 {len(error_files) - 20} 个文件")

    # 保存报告
    report_path = base_dir / CONFIG["input_dir"] / "filter_report.json"
    report = {
        "stats": stats,
        "rejected": rejected_details[:100],
        "errors": error_files[:100],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")


def main():
    print("=" * 60)
    print("筛除pixel_32目录中非32x32的瓦片")
    print("=" * 60)

    print("\n提示: 请确保没有其他程序正在使用这些文件")
    print("      (如slicer_tool服务器、图片查看器等)\n")

    filter_images()


if __name__ == "__main__":
    main()

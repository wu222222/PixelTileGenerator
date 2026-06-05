"""
瓦片分类脚本

功能:
- 扫描reviewed/good目录
- 按尺寸和颜色数分类
- 输出到不同目录
- 生成分类报告
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
from typing import Optional
from collections import defaultdict


# 配置
CONFIG = {
    # 项目根目录
    "project_root": Path(__file__).parent.parent,

    # 输入目录(审核通过的数据)
    "input_dir": "datasets/reviewed/good",

    # 输出目录
    "output_dir": "datasets/classified",

    # 分类规则
    "target_size": 32,           # 目标尺寸
    "max_colors_pixel": 64,      # 像素风最大颜色数
    "max_colors_highres": 128,   # 高清阈值

    # 噪声关键词(文件名包含则归为noise)
    "noise_keywords": ["normal", "specular", "height", "roughness", "metallic", "ao"],

    # 图片扩展名
    "image_extensions": {".png", ".jpg", ".jpeg", ".gif", ".bmp"},
}


class TileClassifier:
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = config["project_root"]

        # 输入目录
        self.input_dir = self.base_dir / config["input_dir"]

        # 输出目录
        self.output_dir = self.base_dir / config["output_dir"]
        self.categories = {
            "pixel_32": self.output_dir / "pixel_32",
            "spritesheet": self.output_dir / "spritesheet",
            "irregular": self.output_dir / "irregular",
            "highres": self.output_dir / "highres",
            "noise": self.output_dir / "noise",
        }

        # 统计
        self.stats = defaultdict(int)
        self.details = defaultdict(list)

    def create_dirs(self):
        """创建输出目录"""
        for dir_path in self.categories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_image_info(self, img_path: Path) -> Optional[dict]:
        """获取图片信息"""
        try:
            # 文件大小
            file_size = img_path.stat().st_size / 1024  # KB

            # 打开图片
            img = Image.open(img_path)

            # 尺寸
            width, height = img.size

            # 颜色数
            try:
                colors = len(img.getcolors(maxcolors=256))
            except TypeError:
                colors = 257

            return {
                "path": img_path,
                "width": width,
                "height": height,
                "file_size_kb": file_size,
                "colors": colors,
                "is_square": width == height,
                "is_32_multiple": width % 32 == 0 and height % 32 == 0,
                "is_exact_32": width == 32 and height == 32,
            }

        except Exception as e:
            print(f"  [错误] 无法读取 {img_path.name}: {e}")
            return None

    def is_noise(self, img_path: Path) -> bool:
        """检查是否是噪声文件"""
        filename_lower = img_path.stem.lower()
        for keyword in self.config["noise_keywords"]:
            if keyword in filename_lower:
                return True
        return False

    def classify_image(self, img_info: dict) -> str:
        """对图片进行分类"""
        path = img_info["path"]
        width = img_info["width"]
        height = img_info["height"]
        colors = img_info["colors"]
        is_square = img_info["is_square"]
        is_32_multiple = img_info["is_32_multiple"]
        is_exact_32 = img_info["is_exact_32"]

        # 检查是否是噪声
        if self.is_noise(path):
            return "noise"

        # 检查是否是32×32像素风
        if is_exact_32 and colors <= self.config["max_colors_pixel"]:
            return "pixel_32"

        # 检查是否是32倍数的大图(可能是spritesheet)
        if is_32_multiple and not is_exact_32:
            return "spritesheet"

        # 检查颜色数是否过多(高清)
        if colors > self.config["max_colors_highres"]:
            return "highres"

        # 其他归为不规则
        return "irregular"

    def process_single_image(self, img_path: Path, resource_name: str, category: str):
        """处理单个图片"""
        info = self.get_image_info(img_path)
        if not info:
            self.stats["error"] += 1
            return

        # 分类
        classification = self.classify_image(info)

        # 生成保存路径
        prefix = f"{category}_{resource_name}"
        save_path = self.categories[classification] / f"{prefix}_{img_path.name}"

        # 复制文件
        shutil.copy2(img_path, save_path)

        # 记录统计
        self.stats[classification] += 1
        self.details[classification].append({
            "file": img_path.name,
            "size": f"{info['width']}×{info['height']}",
            "colors": info["colors"],
            "resource": resource_name,
            "category": category,
        })

    def run(self):
        """主运行函数"""
        print("=" * 60)
        print("瓦片分类脚本")
        print("=" * 60)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

        # 创建目录
        self.create_dirs()

        # 检查输入目录
        if not self.input_dir.exists():
            print(f"[错误] 输入目录不存在: {self.input_dir}")
            return

        # 统计文件数量
        total_files = 0
        for f in self.input_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.config["image_extensions"]:
                total_files += 1

        print(f"找到 {total_files} 个图片文件")

        # 处理每个文件
        processed = 0
        for img_path in self.input_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in self.config["image_extensions"]:
                # 从路径提取资源名和分类
                # 路径格式: reviewed/good/category_resourname_filename.png
                relative = img_path.relative_to(self.input_dir)
                parts = relative.stem.split("_", 2)

                if len(parts) >= 2:
                    category = parts[0]
                    resource_name = parts[1]
                else:
                    category = "unknown"
                    resource_name = relative.stem

                self.process_single_image(img_path, resource_name, category)
                processed += 1

                if processed % 100 == 0:
                    print(f"  已处理: {processed}/{total_files}")

        # 打印报告
        self.print_report()

        # 保存统计到JSON
        self.save_report()

    def print_report(self):
        """打印统计报告"""
        print("\n" + "=" * 60)
        print("分类完成!")
        print("=" * 60)

        total = sum(self.stats.values())
        print(f"总计: {total} 个文件")
        print()
        print("分类统计:")
        for category in ["pixel_32", "spritesheet", "irregular", "highres", "noise"]:
            count = self.stats.get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")
        print(f"  error: {self.stats.get('error', 0)}")
        print("=" * 60)

    def save_report(self):
        """保存统计报告到JSON"""
        report_path = self.output_dir / "classify_report.json"
        report = {
            "stats": dict(self.stats),
            "details": {k: v[:100] for k, v in self.details.items()},  # 只保存前100条详情
            "config": {
                "target_size": self.config["target_size"],
                "max_colors_pixel": self.config["max_colors_pixel"],
                "max_colors_highres": self.config["max_colors_highres"],
                "noise_keywords": self.config["noise_keywords"],
            }
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"报告已保存: {report_path}")


def main():
    classifier = TileClassifier(CONFIG)
    classifier.run()


if __name__ == "__main__":
    main()

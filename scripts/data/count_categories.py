"""
统计每类样本数
"""

import json
from collections import Counter
from pathlib import Path

# 加载标签
labels_path = Path("datasets/classified/pixel_32_filtered/class_labels.json")

with open(labels_path) as f:
    data = json.load(f)

counts = Counter(data["categories"].values())
total = sum(counts.values())

print("类别样本数统计:")
print("=" * 50)
for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    percentage = count / total * 100
    bar = "█" * int(percentage / 2)
    print(f"{cat:25s} {count:4d} ({percentage:5.1f}%) {bar}")

print("=" * 50)
print(f"总计: {total} 张, {len(counts)} 类")
print(f"平均每类: {total/len(counts):.0f} 张")
print(f"最少: {min(counts.values())} 张 ({min(counts, key=counts.get)})")
print(f"最多: {max(counts.values())} 张 ({max(counts, key=counts.get)})")

# 统计小类别
small_categories = {cat: count for cat, count in counts.items() if count < 100}
if small_categories:
    print(f"\n样本数 < 100 的类别 ({len(small_categories)} 个):")
    for cat, count in sorted(small_categories.items(), key=lambda x: x[1]):
        print(f"  {cat}: {count}")

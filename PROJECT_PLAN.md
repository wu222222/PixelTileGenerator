# PixelTileGAN

## 项目目标

构建一个能够生成高质量 32×32 像素风格无缝循环地形瓦片（Tile）的深度学习系统，并与现有的 Autotile 生成器结合，实现完整的 AI 地形生产流水线。

最终目标：

```text
Biome Prompt
      ↓
AI Generator
      ↓
32×32 Seamless Tile
      ↓
Autotile Generator
      ↓
16-Tile Set
      ↓
Godot / Unity
```

---

# 项目背景

目前已经实现：

```text
单瓦片
    ↓
自动生成16瓦片集
```

因此项目重点不再是：

* Wang Tile
* 自动拼接算法
* Autotile系统

而是：

```text
如何生成高质量的32×32无缝像素纹理
```

例如：

* Grass
* Dirt
* Stone
* Sand
* Snow
* Mud
* Crystal
* Lava Rock

---

# MVP（第一阶段）

目标：

训练一个能够生成单类别瓦片的模型。

例如：

```text
Grass
↓
生成
↓
32×32 Grass Tile
```

暂时不考虑：

* 多类别
* Prompt
* 文本输入

---

## 数据集

### 来源

Terraria

Core Keeper

Stardew Valley

OpenGameArt

Kenney Assets

itch.io 免费资源

---

### 数据格式

```text
dataset/

grass/
    grass_001.png
    grass_002.png
    ...

stone/
    stone_001.png
    stone_002.png

sand/
    sand_001.png
    ...
```

---

### 要求

尺寸统一：

```text
32×32
```

格式：

```text
PNG
```

颜色：

```text
RGB
```

目标数量：

```text
每类1000~3000张
```

总计：

```text
5000~10000张
```

---

# 第一阶段模型

## CNN AutoEncoder

结构：

```text
32×32
↓
Encoder
↓
Latent
↓
Decoder
↓
32×32
```

作用：

学习纹理空间。

---

## 验证指标

重建效果：

```text
Input Tile
↓
AutoEncoder
↓
Reconstructed Tile
```

要求：

* 保持像素风格
* 无明显模糊

---

# 第二阶段

## Conditional GAN

目标：

根据类别生成不同地形。

输入：

```text
grass
```

输出：

```text
Grass Tile
```

输入：

```text
stone
```

输出：

```text
Stone Tile
```

---

## 模型结构

```text
Class Label
      ↓
Embedding
      ↓
Generator
      ↓
32×32 Tile

Real Tile
      ↓
Discriminator
```

---

## 输出效果

```text
Grass
↓
随机生成多个不同草地
```

```text
Stone
↓
随机生成多个不同石头
```

---

# 第三阶段

## Seamless Tile

核心阶段。

---

### 问题

普通GAN生成：

```text
Tile A
```

单独看很好。

平铺：

```text
A A
A A
```

出现接缝。

---

## 方案1

### Circular Padding

PyTorch：

```python
padding_mode="circular"
```

效果：

```text
左边连接右边
上边连接下边
```

网络天然学习循环纹理。

---

## 方案2

### Edge Loss

增加边缘损失：

```python
left_edge
right_edge

top_edge
bottom_edge
```

计算：

```python
MSE(left,right)
+
MSE(top,bottom)
```

目标：

```text
左边缘≈右边缘
上边缘≈下边缘
```

---

## 方案3

Tile Preview Loss

训练时：

```text
Tile
↓
2×2拼接
↓
计算连续性
```

让模型直接学习：

```text
无缝平铺
```

---

# 第四阶段

## Pixel Art Optimization

目标：

减少AI常见问题。

---

### 问题1

模糊

解决：

```text
Nearest Neighbor Upsample
```

避免：

```text
Bilinear
```

---

### 问题2

颜色过多

限制：

```text
16色
32色
64色
```

调色板。

---

### 问题3

出现半透明颜色

训练前：

```text
Quantization
```

统一调色板。

---

# 第五阶段

## 自动流水线

流程：

```text
Biome
↓
AI Generator
↓
32×32 Tile
↓
Autotile Generator
↓
16 Tiles
↓
PNG Export
```

---

# 项目目录

```text
pixel-tile-gan/

datasets/

models/

checkpoints/

outputs/

scripts/

train_autoencoder.py

train_gan.py

train_seamless_gan.py

generate_tile.py

autotile_generator.py

evaluate.py
```

---

# 技术栈

## Deep Learning

PyTorch

TorchVision

NumPy

Pillow

OpenCV

---

## 数据处理

Aseprite

Python

---

## 游戏引擎

Godot 4

---

# 训练路线

## Week 1

数据集构建

目标：

```text
5000张以上
```

---

## Week 2

AutoEncoder

验证纹理学习能力

---

## Week 3

Conditional GAN

实现类别控制

---

## Week 4

Seamless Loss

实现无缝平铺

---

## Week 5

流水线整合

实现：

```text
Grass
↓
AI
↓
32×32
↓
16 Tiles
```

一键生成。

---

# 长期扩展

## Text To Tile

输入：

```text
lush green grass
```

输出：

```text
Grass Tile
```

---

## Tile Diffusion

研究：

```text
Stable Diffusion for Pixel Art
```

---

## Tile VAE

学习：

```text
Grass ↔ Dirt ↔ Stone
```

连续潜空间。

---

## Biome Generator

输入：

```text
Forest
```

自动生成：

```text
Grass
Tree
Bush
Flower
Rock
```

整套素材。

---

# 最终成果

实现：

```text
Prompt
↓
AI
↓
32×32 Seamless Tile
↓
Autotile
↓
16 Tile Set
↓
Godot Import
```

形成完整的 AI 像素地形生产工具链。

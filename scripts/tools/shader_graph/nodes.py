"""
节点类型定义

每个节点:
- inputs:  输入端口定义 {name: dtype}
- outputs: 输出端口定义 {name: dtype}
- params:  参数定义 {name: {type, default, min, max, options, desc}}
- execute(context): 执行逻辑, 从 context 读输入, 写输出到 context
"""

import numpy as np
from PIL import Image
from abc import ABC, abstractmethod

# 数据类型
DTYPE_LATENT = "latent"
DTYPE_IMAGE = "image"
DTYPE_SCALAR = "scalar"

# 全局节点注册表
NODE_REGISTRY = {}


def register_node(cls):
    """装饰器: 注册节点类型"""
    NODE_REGISTRY[cls.NODE_TYPE] = cls
    return cls


def create_node(node_type, node_id, params=None):
    """创建节点实例"""
    if node_type not in NODE_REGISTRY:
        raise ValueError(f"未知节点类型: {node_type}")
    return NODE_REGISTRY[node_type](node_id, params or {})


class NodeBase(ABC):
    """节点基类"""
    NODE_TYPE = "base"
    DISPLAY_NAME = "Base"
    DESCRIPTION = "基础节点"
    CATEGORY = "general"
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}

    def __init__(self, node_id, params=None):
        self.id = node_id
        self.params = {}
        for k, v in self.PARAMS.items():
            self.params[k] = v.get("default")
        if params:
            self.params.update(params)

    def get_schema(self):
        return {
            "type": self.NODE_TYPE,
            "name": self.DISPLAY_NAME,
            "description": self.DESCRIPTION,
            "category": self.CATEGORY,
            "inputs": dict(self.INPUTS),
            "outputs": dict(self.OUTPUTS),
            "params": {k: {kk: vv for kk, vv in v.items()} for k, v in self.PARAMS.items()},
        }

    @abstractmethod
    def execute(self, context):
        pass


# ============================================================
# Generator Nodes
# ============================================================

@register_node
class LatentSampleNode(NodeBase):
    NODE_TYPE = "latent_sample"
    DISPLAY_NAME = "Latent Sample"
    DESCRIPTION = "按属性值采样 latent\n拖滑杆改变纹理特征"
    CATEGORY = "generator"
    INPUTS = {}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {
        "structure":    {"type": "slider", "default": 20.0, "min": 5.0,  "max": 45.0,  "label": "Structure",    "desc": "结构强度: 高=砖块, 低=平坦"},
        "entropy":      {"type": "slider", "default": 3.0,  "min": 1.0,  "max": 4.5,   "label": "Entropy",      "desc": "随机性: 高=噪声纹理, 低=规则"},
        "brightness":   {"type": "slider", "default": 100.0,"min": 40.0, "max": 220.0, "label": "Brightness",   "desc": "亮度: 高=雪地/沙地, 低=沼泽/暗色"},
        "periodicity":  {"type": "slider", "default": 6.0,  "min": 3.0,  "max": 10.0,  "label": "Periodicity",  "desc": "周期性: 高=重复图案, 低=随机"},
        "color_variance":{"type": "slider","default": 800.0,"min": 50.0, "max": 3000.0,"label": "Color Var",    "desc": "颜色方差: 高=多彩, 低=单色"},
        "edge_density": {"type": "slider", "default": 0.14, "min": 0.05, "max": 0.20,  "label": "Edge Density", "desc": "边缘密度: 高=复杂边缘, 低=平滑"},
        "local_contrast":{"type": "slider","default": 20.0, "min": 5.0,  "max": 55.0,  "label": "Contrast",     "desc": "局部对比度: 高=强烈明暗, 低=平坦"},
        "color_count":  {"type": "slider", "default": 25.0, "min": 4.0,  "max": 32.0,  "label": "Color Count",  "desc": "使用颜色数: 高=丰富, 低=简约"},
    }

    def execute(self, context):
        from scripts.tools.attribute_generator import generate_latent_from_attrs
        attrs = {k: self.params[k] for k in self.PARAMS}
        latent = generate_latent_from_attrs(attrs)
        context[self.id] = {"latent": latent}


@register_node
class LatentInterpNode(NodeBase):
    NODE_TYPE = "latent_interp"
    DISPLAY_NAME = "Latent Interp"
    DESCRIPTION = "两个 latent 线性插值\nα=0 用 A, α=1 用 B"
    CATEGORY = "generator"
    INPUTS = {"latent_a": DTYPE_LATENT, "latent_b": DTYPE_LATENT}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {
        "alpha": {"type": "slider", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Alpha", "desc": "混合比例: 0=A, 1=B"},
    }

    def execute(self, context):
        z_a = context[self.inputs_connected.get("latent_a", "")]["latent"]
        z_b = context[self.inputs_connected.get("latent_b", "")]["latent"]
        alpha = self.params["alpha"]
        context[self.id] = {"latent": z_a * (1 - alpha) + z_b * alpha}


@register_node
class LatentPerturbNode(NodeBase):
    NODE_TYPE = "latent_perturb"
    DISPLAY_NAME = "Latent Perturb"
    DESCRIPTION = "在 latent 上加随机噪声\nσ 越大变化越强"
    CATEGORY = "generator"
    INPUTS = {"latent": DTYPE_LATENT}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {
        "sigma": {"type": "slider", "default": 0.5, "min": 0.0, "max": 3.0, "label": "Sigma", "desc": "扰动强度: 0=不变, 3=剧烈"},
        "seed":  {"type": "number", "default": -1, "label": "Seed", "desc": "随机种子: -1=每次不同", "integer": True},
    }

    def execute(self, context):
        z_in = context[self.inputs_connected.get("latent", "")]["latent"]
        sigma = self.params["sigma"]
        seed = self.params["seed"]
        if seed >= 0:
            np.random.seed(int(seed))
        context[self.id] = {"latent": z_in + np.random.randn(*z_in.shape) * sigma}


@register_node
class ClusterSampleNode(NodeBase):
    NODE_TYPE = "cluster_sample"
    DISPLAY_NAME = "Cluster Sample"
    DESCRIPTION = "从指定簇中取代表样本\n选择簇编号和样本序号"
    CATEGORY = "generator"
    INPUTS = {}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {
        "cluster":      {"type": "select", "default": 0, "options": list(range(8)), "label": "Cluster", "desc": "簇编号: 0~7"},
        "sample_index": {"type": "number", "default": 0, "min": 0, "max": 4, "label": "Sample", "desc": "簇内样本序号: 0=最靠近中心", "integer": True},
    }

    def execute(self, context):
        cluster_reps = context["_cluster_reps"]
        c = int(self.params["cluster"])
        idx = int(self.params["sample_index"])
        rep_idx = cluster_reps[c][min(idx, len(cluster_reps[c]) - 1)]
        context[self.id] = {"latent": context["_latents"][rep_idx]}


@register_node
class BaseTileNode(NodeBase):
    """从数据集加载已有瓦片"""
    NODE_TYPE = "base_tile"
    DISPLAY_NAME = "Base Tile"
    DESCRIPTION = "加载数据集中的已有瓦片\n可按类别筛选"
    CATEGORY = "generator"
    INPUTS = {}
    OUTPUTS = {"image": DTYPE_IMAGE, "latent": DTYPE_LATENT}
    PARAMS = {
        "category": {"type": "select", "default": "all", "options": [], "label": "Category", "desc": "瓦片类别筛选"},
        "index":    {"type": "number", "default": 0, "min": 0, "max": 9999, "label": "Index", "desc": "该类别中的序号", "integer": True},
    }

    def execute(self, context):
        names = context["_names"]
        latents_data = context["_latents"]
        images_cache = context["_images_cache"]
        category = self.params.get("category", "all")
        index = int(self.params.get("index", 0))

        # 按类别筛选
        def get_cat(n):
            stem = n.rsplit('.', 1)[0]
            if "_from_" in stem:
                return stem.split("_from_")[0]
            return stem.split("_")[0]

        if category and category != "all":
            filtered = [(i, n) for i, n in enumerate(names) if get_cat(n) == category]
        else:
            filtered = list(enumerate(names))

        if not filtered:
            filtered = list(enumerate(names))

        idx = min(index, len(filtered) - 1)
        real_idx, name = filtered[idx]

        context[self.id] = {
            "image": images_cache.get(name),
            "latent": latents_data[real_idx],
        }


@register_node
class EncodeNode(NodeBase):
    """将图片编码为 latent"""
    NODE_TYPE = "encode"
    DISPLAY_NAME = "Encode"
    DESCRIPTION = "图片 → VQ-VAE 编码 → latent\n用于对已有瓦片做修改"
    CATEGORY = "process"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {}

    def execute(self, context):
        import torch
        from torchvision import transforms
        img = context[self.inputs_connected.get("image", "")]["image"]
        device = context["_device"]
        model = context["_model"]
        img_tensor = transforms.ToTensor()(img.convert("RGBA")).unsqueeze(0).to(device)
        with torch.no_grad():
            z_q, _ = model.encode(img_tensor)
        context[self.id] = {"latent": z_q.cpu().numpy().squeeze(0)}


# ============================================================
# Process Nodes
# ============================================================

@register_node
class DecodeNode(NodeBase):
    NODE_TYPE = "decode"
    DISPLAY_NAME = "Decode"
    DESCRIPTION = "latent → VQ-VAE 解码 → 图片"
    CATEGORY = "process"
    INPUTS = {"latent": DTYPE_LATENT}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {}

    def execute(self, context):
        import torch
        from torchvision import transforms
        z = context[self.inputs_connected.get("latent", "")]["latent"]
        device = context["_device"]
        model = context["_model"]
        z_tensor = torch.FloatTensor(z).unsqueeze(0).to(device)
        with torch.no_grad():
            img = model.decode(z_tensor)
        context[self.id] = {"image": transforms.ToPILImage()(img.squeeze(0).cpu())}


@register_node
class QuantizeNode(NodeBase):
    NODE_TYPE = "quantize"
    DISPLAY_NAME = "Quantize"
    DESCRIPTION = "将图片量化到指定颜色数\n像素画风格必备"
    CATEGORY = "process"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {
        "colors": {"type": "number", "default": 32, "min": 4, "max": 64, "label": "Colors", "desc": "颜色数量: 4=极简, 32=标准像素画", "integer": True},
    }

    def execute(self, context):
        img = context[self.inputs_connected.get("image", "")]["image"]
        colors = int(self.params["colors"])
        if img.mode == "RGBA":
            q = img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
        else:
            q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        context[self.id] = {"image": q.convert("RGBA")}


@register_node
class PaletteNode(NodeBase):
    NODE_TYPE = "palette"
    DISPLAY_NAME = "Palette"
    DESCRIPTION = "替换调色板\n保留结构, 换颜色风格"
    CATEGORY = "process"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {
        "name":       {"type": "select", "default": "original", "options": [], "label": "Palette", "desc": "目标调色板"},
        "intensity":  {"type": "slider", "default": 1.0, "min": 0.0, "max": 1.0, "label": "Intensity", "desc": "替换强度: 0=原色, 1=完全替换"},
        "saturation": {"type": "slider", "default": 1.0, "min": 0.0, "max": 2.0, "label": "Saturation", "desc": "饱和度: 0=灰度, 2=增强"},
    }

    def execute(self, context):
        from scripts.tools.palette_remapper import apply_palette
        img = context[self.inputs_connected.get("image", "")]["image"]
        name = self.params["name"]
        if name and name != "original":
            img = apply_palette(img, name, self.params["intensity"], self.params["saturation"])
        context[self.id] = {"image": img}


@register_node
class BlendNode(NodeBase):
    """混合两张图片"""
    NODE_TYPE = "blend"
    DISPLAY_NAME = "Blend"
    DESCRIPTION = "混合两张图片\nα=0 用 A, α=1 用 B"
    CATEGORY = "process"
    INPUTS = {"image_a": DTYPE_IMAGE, "image_b": DTYPE_IMAGE}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {
        "alpha": {"type": "slider", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Alpha", "desc": "混合比例: 0=A, 1=B"},
        "mode":  {"type": "select", "default": "mix", "options": ["mix", "overlay", "multiply"], "label": "Mode", "desc": "混合模式"},
    }

    def execute(self, context):
        img_a = context[self.inputs_connected.get("image_a", "")]["image"]
        img_b = context[self.inputs_connected.get("image_b", "")]["image"]
        alpha = self.params["alpha"]
        mode = self.params.get("mode", "mix")

        # 确保尺寸一致
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size, Image.Resampling.NEAREST)

        a = np.array(img_a.convert("RGBA")).astype(float)
        b = np.array(img_b.convert("RGBA")).astype(float)

        if mode == "mix":
            result = a * (1 - alpha) + b * alpha
        elif mode == "overlay":
            # Overlay: 暗区用 multiply, 亮区用 screen
            mask = a < 128
            multiply = 2 * a * b / 255.0
            screen = 255 - 2 * (255 - a) * (255 - b) / 255.0
            result = np.where(mask, multiply, screen) * alpha + a * (1 - alpha)
        elif mode == "multiply":
            result = (a * b / 255.0) * alpha + a * (1 - alpha)
        else:
            result = a * (1 - alpha) + b * alpha

        context[self.id] = {"image": Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGBA")}


@register_node
class ResizeNode(NodeBase):
    NODE_TYPE = "resize"
    DISPLAY_NAME = "Resize"
    DESCRIPTION = "最近邻缩放\n保持像素画锐利"
    CATEGORY = "process"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {
        "scale": {"type": "number", "default": 4, "min": 1, "max": 16, "label": "Scale", "desc": "缩放倍数: 1=原始, 4=128px, 8=256px", "integer": True},
    }

    def execute(self, context):
        img = context[self.inputs_connected.get("image", "")]["image"]
        scale = int(self.params["scale"])
        context[self.id] = {"image": img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)}


# ============================================================
# Output Nodes
# ============================================================

@register_node
class OutputNode(NodeBase):
    NODE_TYPE = "output"
    DISPLAY_NAME = "Output"
    DESCRIPTION = "最终输出\n预览窗口显示此节点结果"
    CATEGORY = "output"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {}
    PARAMS = {
        "label": {"type": "text", "default": "output", "label": "Label", "desc": "输出标签名"},
    }

    def execute(self, context):
        img = context[self.inputs_connected.get("image", "")]["image"]
        context[self.id] = {"image": img, "_is_output": True}


# ============================================================
# 工具函数
# ============================================================

def get_all_node_schemas():
    schemas = {}
    for node_type, cls in NODE_REGISTRY.items():
        schemas[node_type] = {
            "type": cls.NODE_TYPE,
            "name": cls.DISPLAY_NAME,
            "description": cls.DESCRIPTION,
            "category": cls.CATEGORY,
            "inputs": dict(cls.INPUTS),
            "outputs": dict(cls.OUTPUTS),
            "params": {k: {kk: vv for kk, vv in v.items()} for k, v in cls.PARAMS.items()},
        }
    return schemas


def get_palette_options():
    from scripts.tools.palette_remapper import list_palettes
    return ["original"] + [p["id"] for p in list_palettes()]


def get_tile_categories():
    """获取数据集中的瓦片类别列表"""
    from pathlib import Path
    import json
    project_root = Path(__file__).parent.parent.parent.parent
    names_path = project_root / "datasets" / "vqvae_latent_data_v7" / "names.json"
    if not names_path.exists():
        return ["all"]
    names = json.load(open(names_path))
    cats = set()
    for n in names:
        stem = n.rsplit('.', 1)[0]
        if "_from_" in stem:
            cats.add(stem.split("_from_")[0])
        else:
            cats.add(stem.split("_")[0])
    return ["all"] + sorted(cats)

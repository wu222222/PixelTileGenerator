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
        from scripts.tools.attribute_generator_v9 import generate_from_attributes
        attrs = {k: self.params[k] for k in self.PARAMS}
        # attribute_generator_v9 直接返回图片，不返回 latent
        # 这里需要返回 latent，所以用 KNN 找最近邻的 z_q
        import json
        from pathlib import Path
        from sklearn.neighbors import NearestNeighbors

        project_root = Path(__file__).parent.parent.parent.parent
        attr_path = project_root / "checkpoints" / "v9_attribute_space" / "attributes.json"
        scaler_path = project_root / "checkpoints" / "v9_attribute_space" / "scaler_params.json"

        if not attr_path.exists():
            context[self.id] = {"latent": context["_latents"][0]}
            return

        attr_data = json.load(open(attr_path))
        scaler_data = json.load(open(scaler_path))
        attr_keys = list(attr_data["attributes"].keys())
        attr_matrix = np.array([attr_data["attributes"][k] for k in attr_keys]).T
        scaler_mean = np.array(scaler_data["mean"])
        scaler_scale = np.array(scaler_data["scale"])

        target_raw = np.array([attrs.get(k, 0) for k in attr_keys])
        target_scaled = (target_raw - scaler_mean) / scaler_scale
        attr_scaled = (attr_matrix - scaler_mean) / scaler_scale

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(attr_scaled)
        _, indices = nn.kneighbors(target_scaled.reshape(1, -1))

        context[self.id] = {"latent": context["_latents"][indices[0][0]]}


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
class StructureNode(NodeBase):
    """语义结构类型选择"""
    NODE_TYPE = "structure"
    DISPLAY_NAME = "Structure"
    DESCRIPTION = "选择地形结构类型\n输出该类型的代表 latent"
    CATEGORY = "structure"
    INPUTS = {}
    OUTPUTS = {"latent": DTYPE_LATENT}
    PARAMS = {
        "type":    {"type": "select", "default": "flat", "options": ["flat", "grass", "ice", "rock"], "label": "Type", "desc": "结构类型"},
        "variant": {"type": "slider", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Variant", "desc": "类型内变化: 0=典型, 1=边缘"},
    }

    def execute(self, context):
        import json
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        labels_path = project_root / "checkpoints" / "v9_attribute_space" / "structure_labels.json"

        names = context["_names"]
        latents_data = context["_latents"]
        stype = self.params.get("type", "flat")
        variant = float(self.params.get("variant", 0.5))

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
            labels = labels_data.get("labels", {})
            type_indices = [i for i, n in enumerate(names) if labels.get(n) == stype]
        else:
            # fallback: 按文件名前缀匹配
            type_indices = [i for i, n in enumerate(names) if stype in n.lower()]

        if not type_indices:
            type_indices = list(range(len(names)))

        # 根据 variant 在该类型的 latent 中选择/插值
        # variant=0 → 最靠近中心的样本, variant=1 → 最远离中心的样本
        type_latents = latents_data[type_indices]
        center = type_latents.reshape(len(type_indices), -1).mean(axis=0)
        dists = np.linalg.norm(type_latents.reshape(len(type_indices), -1) - center, axis=1)
        sorted_idx = np.argsort(dists)

        # 在排序后的样本中按 variant 插值
        pos = variant * (len(sorted_idx) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_idx) - 1)
        alpha = pos - lo

        z_lo = latents_data[type_indices[sorted_idx[lo]]]
        z_hi = latents_data[type_indices[sorted_idx[hi]]]
        latent = z_lo * (1 - alpha) + z_hi * alpha

        context[self.id] = {"latent": latent}


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
        "biome":      {"type": "select", "default": "original", "options": ["original", "grass", "snow", "desert", "swamp", "lava"], "label": "Biome", "desc": "生物群系: 自动匹配调色板"},
        "name":       {"type": "select", "default": "original", "options": [], "label": "Palette", "desc": "手动选择调色板 (优先级高于 Biome)"},
        "intensity":  {"type": "slider", "default": 1.0, "min": 0.0, "max": 1.0, "label": "Intensity", "desc": "替换强度: 0=原色, 1=完全替换"},
        "saturation": {"type": "slider", "default": 1.0, "min": 0.0, "max": 2.0, "label": "Saturation", "desc": "饱和度: 0=灰度, 2=增强"},
        "hue_shift":  {"type": "slider", "default": 0.0, "min": -180.0, "max": 180.0, "label": "Hue Shift", "desc": "色相偏移: -180~+180度"},
    }

    def execute(self, context):
        from scripts.tools.palette_remapper import apply_palette
        img = context[self.inputs_connected.get("image", "")]["image"]
        name = self.params["name"]
        biome = self.params.get("biome", "original")

        # 如果手动选了 palette, 用手动的; 否则用 biome 自动匹配
        if name and name != "original":
            palette_name = name
        elif biome and biome != "original":
            palette_name = self._biome_to_palette(biome)
        else:
            palette_name = None

        if palette_name:
            img = apply_palette(img, palette_name, self.params["intensity"], self.params["saturation"])

        # 色相偏移
        hue_shift = self.params.get("hue_shift", 0)
        if abs(hue_shift) > 0.5:
            img = self._shift_hue(img, hue_shift)

        context[self.id] = {"image": img}

    @staticmethod
    def _biome_to_palette(biome):
        mapping = {"grass": "grass", "snow": "snow", "desert": "desert", "swamp": "grass", "lava": "lava"}
        return mapping.get(biome, "grass")

    @staticmethod
    def _shift_hue(img, degrees):
        arr = np.array(img.convert("RGBA")).astype(float)
        rgb = arr[:, :, :3] / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c + 1e-10
        h = np.zeros_like(r)
        mask_r = max_c == r
        mask_g = max_c == g
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
        mask_b = ~mask_r & ~mask_g
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
        h = h / 6.0
        s = np.where(max_c > 0, delta / (max_c + 1e-10), 0)
        v = max_c
        h = (h + degrees / 360.0) % 1.0
        i = (h * 6).astype(int) % 6
        f = h * 6 - np.floor(h * 6)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r_out = np.choose(i, [v, q, p, p, t, v])
        g_out = np.choose(i, [t, v, v, q, p, p])
        b_out = np.choose(i, [p, p, t, v, v, q])
        arr[:, :, :3] = np.stack([r_out, g_out, b_out], axis=2) * 255
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


@register_node
class TintNode(NodeBase):
    """自由颜色着色"""
    NODE_TYPE = "tint"
    DISPLAY_NAME = "Tint"
    DESCRIPTION = "用自定义颜色着色\n保留亮度结构, 替换色相"
    CATEGORY = "process"
    INPUTS = {"image": DTYPE_IMAGE}
    OUTPUTS = {"image": DTYPE_IMAGE}
    PARAMS = {
        "color":    {"type": "text", "default": "#4CAF50", "label": "Color", "desc": "十六进制颜色 (如 #ff6600)"},
        "strength": {"type": "slider", "default": 1.0, "min": 0.0, "max": 1.0, "label": "Strength", "desc": "着色强度: 0=原色, 1=完全着色"},
    }

    def execute(self, context):
        img = context[self.inputs_connected.get("image", "")]["image"]
        hex_color = self.params.get("color", "#4CAF50").lstrip("#")
        strength = float(self.params.get("strength", 1.0))

        tr = int(hex_color[0:2], 16) / 255.0
        tg = int(hex_color[2:4], 16) / 255.0
        tb = int(hex_color[4:6], 16) / 255.0

        arr = np.array(img.convert("RGBA")).astype(float)
        rgb = arr[:, :, :3] / 255.0
        gray = rgb.mean(axis=2, keepdims=True)
        tinted = gray * np.array([[[tr, tg, tb]]])
        result = rgb * (1 - strength) + tinted * strength
        arr[:, :, :3] = result * 255
        context[self.id] = {"image": Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")}


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
    names_path = project_root / "datasets" / "vqvae_v9_zq_data" / "names.json"
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

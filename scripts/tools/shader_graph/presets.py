"""
预设图 — 常用纹理生成管线
"""

PRESETS = {
    "brick": {
        "name": "砖块 (Brick)",
        "description": "高结构规则砖块纹理",
        "graph": {
            "nodes": [
                {"id": "gen1", "type": "latent_sample", "x": 50, "y": 100,
                 "params": {"structure": 35, "entropy": 3.0, "brightness": 100,
                            "periodicity": 6.5, "color_variance": 1500,
                            "edge_density": 0.15, "local_contrast": 35, "color_count": 32}},
                {"id": "dec1", "type": "decode", "x": 300, "y": 100},
                {"id": "qnt1", "type": "quantize", "x": 500, "y": 100, "params": {"colors": 32}},
                {"id": "out1", "type": "output", "x": 700, "y": 100, "params": {"label": "brick"}},
            ],
            "edges": [
                {"from": "gen1", "to": "dec1", "slot": "latent"},
                {"from": "dec1", "to": "qnt1", "slot": "image"},
                {"from": "qnt1", "to": "out1", "slot": "image"},
            ],
        },
    },

    "grass": {
        "name": "草地 (Grass)",
        "description": "中等结构随机草地",
        "graph": {
            "nodes": [
                {"id": "gen1", "type": "latent_sample", "x": 50, "y": 100,
                 "params": {"structure": 18, "entropy": 3.0, "brightness": 80,
                            "periodicity": 5.5, "color_variance": 600,
                            "edge_density": 0.13, "local_contrast": 20, "color_count": 30}},
                {"id": "dec1", "type": "decode", "x": 300, "y": 100},
                {"id": "qnt1", "type": "quantize", "x": 500, "y": 100, "params": {"colors": 32}},
                {"id": "out1", "type": "output", "x": 700, "y": 100, "params": {"label": "grass"}},
            ],
            "edges": [
                {"from": "gen1", "to": "dec1", "slot": "latent"},
                {"from": "dec1", "to": "qnt1", "slot": "image"},
                {"from": "qnt1", "to": "out1", "slot": "image"},
            ],
        },
    },

    "snow_grass": {
        "name": "草地→雪地 (Palette Swap)",
        "description": "草地结构 + 雪地调色板",
        "graph": {
            "nodes": [
                {"id": "gen1", "type": "latent_sample", "x": 50, "y": 100,
                 "params": {"structure": 18, "entropy": 3.0, "brightness": 80,
                            "periodicity": 5.5, "color_variance": 600,
                            "edge_density": 0.13, "local_contrast": 20, "color_count": 30}},
                {"id": "dec1", "type": "decode", "x": 300, "y": 100},
                {"id": "qnt1", "type": "quantize", "x": 500, "y": 100, "params": {"colors": 32}},
                {"id": "pal1", "type": "palette", "x": 700, "y": 100,
                 "params": {"name": "snow", "intensity": 0.85, "saturation": 0.7}},
                {"id": "out1", "type": "output", "x": 900, "y": 100, "params": {"label": "snow_grass"}},
            ],
            "edges": [
                {"from": "gen1", "to": "dec1", "slot": "latent"},
                {"from": "dec1", "to": "qnt1", "slot": "image"},
                {"from": "qnt1", "to": "pal1", "slot": "image"},
                {"from": "pal1", "to": "out1", "slot": "image"},
            ],
        },
    },

    "brick_snow": {
        "name": "砖块→雪砖 (Palette Swap)",
        "description": "砖块结构 + 雪地调色板",
        "graph": {
            "nodes": [
                {"id": "gen1", "type": "latent_sample", "x": 50, "y": 100,
                 "params": {"structure": 35, "entropy": 3.0, "brightness": 100,
                            "periodicity": 6.5, "color_variance": 1500,
                            "edge_density": 0.15, "local_contrast": 35, "color_count": 32}},
                {"id": "dec1", "type": "decode", "x": 300, "y": 100},
                {"id": "qnt1", "type": "quantize", "x": 500, "y": 100, "params": {"colors": 32}},
                {"id": "pal1", "type": "palette", "x": 700, "y": 100,
                 "params": {"name": "snow", "intensity": 0.9, "saturation": 0.6}},
                {"id": "out1", "type": "output", "x": 900, "y": 100, "params": {"label": "snow_brick"}},
            ],
            "edges": [
                {"from": "gen1", "to": "dec1", "slot": "latent"},
                {"from": "dec1", "to": "qnt1", "slot": "image"},
                {"from": "qnt1", "to": "pal1", "slot": "image"},
                {"from": "pal1", "to": "out1", "slot": "image"},
            ],
        },
    },

    "interp": {
        "name": "双源插值 + 换肤",
        "description": "两个 latent 插值 + 调色板",
        "graph": {
            "nodes": [
                {"id": "gen1", "type": "cluster_sample", "x": 50, "y": 50,
                 "params": {"cluster": 7, "sample_index": 0}},
                {"id": "gen2", "type": "cluster_sample", "x": 50, "y": 200,
                 "params": {"cluster": 5, "sample_index": 0}},
                {"id": "interp1", "type": "latent_interp", "x": 300, "y": 120,
                 "params": {"alpha": 0.4}},
                {"id": "dec1", "type": "decode", "x": 520, "y": 120},
                {"id": "qnt1", "type": "quantize", "x": 700, "y": 120, "params": {"colors": 32}},
                {"id": "pal1", "type": "palette", "x": 880, "y": 120,
                 "params": {"name": "autumn", "intensity": 0.7, "saturation": 1.2}},
                {"id": "out1", "type": "output", "x": 1080, "y": 120, "params": {"label": "interp"}},
            ],
            "edges": [
                {"from": "gen1", "to": "interp1", "slot": "latent_a"},
                {"from": "gen2", "to": "interp1", "slot": "latent_b"},
                {"from": "interp1", "to": "dec1", "slot": "latent"},
                {"from": "dec1", "to": "qnt1", "slot": "image"},
                {"from": "qnt1", "to": "pal1", "slot": "image"},
                {"from": "pal1", "to": "out1", "slot": "image"},
            ],
        },
    },
}


def get_preset(name):
    """获取预设图"""
    return PRESETS.get(name)


def list_presets():
    """列出所有预设"""
    return [{"id": k, "name": v["name"], "description": v["description"]} for k, v in PRESETS.items()]

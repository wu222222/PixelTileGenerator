"""
Attribute-Controlled Texture Generator

输入: 滑杆指定属性值 (structure, entropy, brightness, ...)
输出: 符合属性的纹理

原理: 在属性空间中找到最近邻 latent, 加权插值后解码
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file
from io import BytesIO
from sklearn.neighbors import NearestNeighbors

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2
from scripts.tools.palette_remapper import list_palettes, apply_palette

CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v7/vqvae_v7_best.pth",
    "latent_data": "datasets/vqvae_latent_data_v7",
    "attribute_data": "checkpoints/v7_attribute_space/attributes.json",
    "scaler_params": "checkpoints/v7_attribute_space/scaler_params.json",
    "output_dir": "generated_textures",
    "port": 5004,
    "n_neighbors": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

app = Flask(__name__)

# 全局变量
vqvae = None
latents = None
names = None
attr_matrix = None
attr_keys = None
scaler_mean = None
scaler_scale = None
nn_model = None
attr_ranges = None


def load_models():
    global vqvae, latents, names, attr_matrix, attr_keys
    global scaler_mean, scaler_scale, nn_model, attr_ranges

    device = CONFIG["device"]

    # 加载 VQ-VAE
    ckpt_path = project_root / CONFIG["vqvae_checkpoint"]
    print(f"加载 VQ-VAE: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    latent_size = config.get("latent_size", 16)

    vqvae = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
        latent_size=latent_size,
    ).to(device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    # 加载 latent
    latent_dir = project_root / CONFIG["latent_data"]
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))
    print(f"加载 {len(names)} 个 latent")

    # 加载属性
    attr_path = project_root / CONFIG["attribute_data"]
    attr_data = json.load(open(attr_path))
    attr_keys = list(attr_data["attributes"].keys())
    attr_matrix = np.array([attr_data["attributes"][k] for k in attr_keys]).T
    print(f"属性: {attr_keys}, shape: {attr_matrix.shape}")

    # 加载标准化参数
    scaler_path = project_root / CONFIG["scaler_params"]
    scaler_data = json.load(open(scaler_path))
    scaler_mean = np.array(scaler_data["mean"])
    scaler_scale = np.array(scaler_data["scale"])

    # 标准化属性矩阵
    attr_scaled = (attr_matrix - scaler_mean) / scaler_scale

    # 拟合最近邻模型
    nn_model = NearestNeighbors(n_neighbors=CONFIG["n_neighbors"], metric='euclidean')
    nn_model.fit(attr_scaled)

    # 计算属性范围 (用于滑杆)
    attr_ranges = {}
    for i, key in enumerate(attr_keys):
        vmin, vmax = float(attr_matrix[:, i].min()), float(attr_matrix[:, i].max())
        attr_ranges[key] = {"min": vmin, "max": vmax, "mean": float(attr_matrix[:, i].mean())}

    print(f"属性范围: {attr_ranges}")
    print("加载完成")


def generate_latent_from_attrs(target_attrs):
    """
    给定目标属性值, 返回 latent (numpy array)

    Args:
        target_attrs: dict {attr_name: value}

    Returns:
        numpy array: latent (64, 16, 16)
    """
    global nn_model, scaler_mean, scaler_scale, attr_keys, latents, attr_ranges

    # 确保模型已加载
    if nn_model is None:
        load_models()

    target_raw = np.array([target_attrs.get(k, attr_ranges[k]["mean"]) for k in attr_keys])
    target_scaled = (target_raw - scaler_mean) / scaler_scale

    distances, indices = nn_model.kneighbors(target_scaled.reshape(1, -1))
    distances = distances[0]
    indices = indices[0]

    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum()

    z_interp = np.zeros_like(latents[0])
    for w, idx in zip(weights, indices):
        z_interp += w * latents[idx]

    return z_interp


def generate_from_attributes(target_attrs):
    """
    给定目标属性值, 生成纹理

    Args:
        target_attrs: dict {attr_name: value}

    Returns:
        PIL.Image
    """
    # 构建目标向量
    target_raw = np.array([target_attrs.get(k, attr_ranges[k]["mean"]) for k in attr_keys])
    target_scaled = (target_raw - scaler_mean) / scaler_scale

    # 找最近邻
    distances, indices = nn_model.kneighbors(target_scaled.reshape(1, -1))
    distances = distances[0]
    indices = indices[0]

    # 加权插值 (距离越近权重越大)
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum()

    # latent 加权平均
    z_interp = np.zeros_like(latents[0])
    for w, idx in zip(weights, indices):
        z_interp += w * latents[idx]

    # 解码
    device = CONFIG["device"]
    z_tensor = torch.FloatTensor(z_interp).unsqueeze(0).to(device)
    with torch.no_grad():
        img = vqvae.decode(z_tensor)

    img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())

    if img_pil.mode == "RGBA":
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.FASTOCTREE)
    else:
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.MEDIANCUT)

    return img_quantized.convert("RGBA")


def img_to_bytes(img):
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attribute-Controlled Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
        h1 { margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
        .main { display: flex; gap: 20px; }
        .left { flex: 1; text-align: center; }
        .right { width: 320px; }
        .panel { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .panel h3 { margin-bottom: 10px; font-size: 14px; color: #aaa; }
        .preview-box { background: #222; padding: 15px; border-radius: 8px; display: inline-block; }
        .preview-box img { width: 256px; height: 256px; image-rendering: pixelated; }
        .batch-grid { display: flex; gap: 8px; justify-content: center; margin-top: 15px; flex-wrap: wrap; }
        .batch-grid img { width: 64px; height: 64px; image-rendering: pixelated; border: 1px solid #444; cursor: pointer; }
        .batch-grid img:hover { border-color: #4CAF50; }
        label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; font-size: 13px; }
        label .val { font-family: monospace; color: #4CAF50; min-width: 50px; text-align: right; }
        input[type=range] { width: 100%; margin-bottom: 12px; }
        button { padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 5px; margin-bottom: 5px; }
        button:hover { background: #45a049; }
        button.secondary { background: #555; }
        button.secondary:hover { background: #666; }
        .preset { font-size: 11px; padding: 4px 8px; }
        select { width: 100%; padding: 6px; margin-bottom: 10px; }
        .info { background: #333; padding: 8px; border-radius: 4px; font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Attribute-Controlled Generator</h1>
    <p class="subtitle">指定纹理属性值 → 生成匹配纹理</p>

    <div class="main">
        <div class="left">
            <div class="preview-box">
                <img id="preview-img" src="/generate" alt="Preview">
            </div>
            <div class="info" id="info-text">调整滑杆生成纹理</div>

            <div class="panel" style="text-align:left;">
                <h3>批量生成 (4张变体)</h3>
                <button onclick="generateBatch()">生成变体</button>
                <div class="batch-grid" id="batch-grid"></div>
            </div>
        </div>

        <div class="right">
            <div class="panel">
                <h3>属性控制</h3>
                {% for key, r in ranges.items() %}
                <label>{{key}} <span class="val" id="val-{{key}}">{{"%.1f"|format(r.mean)}}</span></label>
                <input type="range" id="attr-{{key}}" min="{{r.min}}" max="{{r.max}}"
                    step="{{(r.max - r.min) / 100}}" value="{{r.mean}}"
                    oninput="updateAttr('{{key}}'); generate()">
                {% endfor %}
            </div>

            <div class="panel">
                <h3>预设</h3>
                <button class="preset" onclick="setPreset('brick')">砖块</button>
                <button class="preset" onclick="setPreset('grass')">草地</button>
                <button class="preset" onclick="setPreset('sand')">沙漠</button>
                <button class="preset" onclick="setPreset('snow')">雪地</button>
                <button class="preset" onclick="setPreset('flat')">平坦</button>
                <button class="preset" onclick="setPreset('complex')">复杂</button>
                <button class="preset" onclick="setPreset('bright')">明亮</button>
                <button class="preset" onclick="setPreset('dark')">暗色</button>
            </div>

            <div class="panel">
                <h3>Palette 换肤</h3>
                <select id="palette" onchange="generate()">
                    <option value="original">原始调色板</option>
                    {% for p in palettes %}
                    <option value="{{p.id}}">{{p.name}}</option>
                    {% endfor %}
                </select>
                <label>强度 <span class="val" id="val-intensity">1.0</span></label>
                <input type="range" id="intensity" min="0" max="1" step="0.05" value="1.0"
                    oninput="document.getElementById('val-intensity').textContent=this.value; generate()">
                <label>饱和度 <span class="val" id="val-saturation">1.0</span></label>
                <input type="range" id="saturation" min="0" max="2" step="0.05" value="1.0"
                    oninput="document.getElementById('val-saturation').textContent=this.value; generate()">
            </div>

            <div class="panel">
                <h3>操作</h3>
                <button onclick="randomize()">随机属性</button>
                <button class="secondary" onclick="resetAll()">重置</button>
                <button onclick="saveCurrent()">保存当前</button>
            </div>
        </div>
    </div>

    <script>
        const attrKeys = {{ attr_keys | tojson }};
        const attrRanges = {{ ranges | tojson }};

        function getAttrUrl() {
            const params = new URLSearchParams();
            attrKeys.forEach(k => {
                params.set(k, document.getElementById('attr-' + k).value);
            });
            return params.toString();
        }

        function updateAttr(key) {
            document.getElementById('val-' + key).textContent =
                parseFloat(document.getElementById('attr-' + key).value).toFixed(1);
        }

        function generate() {
            const url = '/generate?' + getAttrUrl();
            document.getElementById('preview-img').src = url;

            const palette = document.getElementById('palette').value;
            const intensity = document.getElementById('intensity').value;
            const saturation = document.getElementById('saturation').value;
            if (palette !== 'original') {
                document.getElementById('preview-img').src =
                    url + '&palette=' + palette + '&intensity=' + intensity + '&saturation=' + saturation;
            }

            // 更新info
            const vals = attrKeys.map(k => k + ':' + parseFloat(document.getElementById('attr-' + k).value).toFixed(1));
            document.getElementById('info-text').textContent = vals.join(' | ');
        }

        function generateBatch() {
            const grid = document.getElementById('batch-grid');
            grid.innerHTML = '';
            for (let i = 0; i < 4; i++) {
                const img = document.createElement('img');
                const url = '/generate?' + getAttrUrl() + '&seed=' + Math.floor(Math.random() * 999999);
                img.src = url;
                img.onclick = function() { document.getElementById('preview-img').src = this.src; };
                grid.appendChild(img);
            }
        }

        function setPreset(name) {
            const presets = {
                brick:    {structure: 35, entropy: 3.0, color_variance: 1500, brightness: 100, periodicity: 6.5, edge_density: 0.15, local_contrast: 35, color_count: 32},
                grass:    {structure: 18, entropy: 3.0, color_variance: 600, brightness: 80, periodicity: 5.5, edge_density: 0.13, local_contrast: 20, color_count: 30},
                sand:     {structure: 12, entropy: 2.0, color_variance: 200, brightness: 140, periodicity: 7.0, edge_density: 0.10, local_contrast: 10, color_count: 15},
                snow:     {structure: 10, entropy: 1.5, color_variance: 100, brightness: 200, periodicity: 8.0, edge_density: 0.08, local_contrast: 8, color_count: 12},
                flat:     {structure: 6, entropy: 1.4, color_variance: 96, brightness: 122, periodicity: 8.9, edge_density: 0.14, local_contrast: 8, color_count: 12},
                complex:  {structure: 39, entropy: 3.8, color_variance: 2800, brightness: 180, periodicity: 8.2, edge_density: 0.16, local_contrast: 50, color_count: 32},
                bright:   {structure: 17, entropy: 2.7, color_variance: 516, brightness: 174, periodicity: 8.1, edge_density: 0.14, local_contrast: 21, color_count: 27},
                dark:     {structure: 16, entropy: 2.5, color_variance: 354, brightness: 78, periodicity: 5.6, edge_density: 0.15, local_contrast: 15, color_count: 19},
            };
            const p = presets[name];
            if (!p) return;
            attrKeys.forEach(k => {
                const el = document.getElementById('attr-' + k);
                if (p[k] !== undefined) {
                    el.value = p[k];
                    updateAttr(k);
                }
            });
            generate();
        }

        function randomize() {
            attrKeys.forEach(k => {
                const r = attrRanges[k];
                const val = r.min + Math.random() * (r.max - r.min);
                document.getElementById('attr-' + k).value = val;
                updateAttr(k);
            });
            generate();
        }

        function resetAll() {
            attrKeys.forEach(k => {
                document.getElementById('attr-' + k).value = attrRanges[k].mean;
                updateAttr(k);
            });
            generate();
        }

        function saveCurrent() {
            const src = document.getElementById('preview-img').src;
            const a = document.createElement('a');
            a.href = src;
            a.download = 'texture.png';
            a.click();
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        attr_keys=attr_keys,
        ranges=attr_ranges,
        palettes=list_palettes(),
    )


@app.route('/generate')
def generate():
    # 读取属性值
    target_attrs = {}
    for key in attr_keys:
        val = request.args.get(key, None, type=float)
        if val is not None:
            target_attrs[key] = val

    palette = request.args.get('palette', None, type=str)
    intensity = request.args.get('intensity', 1.0, type=float)
    saturation = request.args.get('saturation', 1.0, type=float)

    img = generate_from_attributes(target_attrs)

    if palette and palette != "original":
        img = apply_palette(img, palette, intensity, saturation)

    return send_file(img_to_bytes(img), mimetype='image/png')


def main():
    print("=" * 60)
    print("Attribute-Controlled Texture Generator")
    print("=" * 60)

    load_models()

    print(f"\n启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

"""
纹理浏览器

功能:
- 实时预览纹理生成
- 调整扰动强度
- 选择基础样本
- 保存喜欢的纹理
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file
from io import BytesIO

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2
from scripts.tools.palette_remapper import list_palettes, apply_palette


# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v5/vqvae_v5_best.pth",
    "latent_data": "datasets/vqvae_latent_data",
    "pca_model": "checkpoints/vqvae_v5/pca_reconstruction/pca_model.pkl",
    "output_dir": "generated_textures",
    "port": 5002,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

app = Flask(__name__)

# 全局变量
vqvae = None
pca = None
latents_pca = None
names = None
latent_size = 16  # 从 checkpoint config 动态读取


def load_models():
    """加载模型"""
    global vqvae, pca, latents_pca, names, latent_size

    # 加载VQ-VAE
    checkpoint_path = project_root / CONFIG["vqvae_checkpoint"]
    print(f"加载VQ-VAE: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    config = checkpoint.get("config", {})

    latent_size = config.get("latent_size", 16)
    print(f"Latent大小: {latent_size}×{latent_size}")

    vqvae = VQVAEv2(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 64),
        num_embeddings=config.get("num_embeddings", 256),
        latent_size=latent_size,
    ).to(CONFIG["device"])

    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    # 加载PCA模型
    pca_path = project_root / CONFIG["pca_model"]
    print(f"加载PCA模型: {pca_path}")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    # 加载latent数据
    latent_data_path = project_root / CONFIG["latent_data"]
    latents = np.load(latent_data_path / "latents.npy")
    names = json.load(open(latent_data_path / "names.json"))

    # PCA压缩
    latents_flat = latents.reshape(latents.shape[0], -1)
    latents_pca = pca.transform(latents_flat)

    print(f"加载完成: {len(names)} 个样本")


def generate_texture(base_idx, sigma, seed=None, palette=None, intensity=1.0, saturation=1.0):
    """生成纹理"""
    if seed is not None:
        np.random.seed(seed)
        used_seed = seed
    else:
        used_seed = np.random.randint(0, 1000000)
        np.random.seed(used_seed)

    # 获取基础latent
    z_pca_original = latents_pca[base_idx]

    # 添加扰动
    noise = np.random.randn(*z_pca_original.shape) * sigma
    z_pca_new = z_pca_original + noise

    # 逆PCA
    z_new = pca.inverse_transform(z_pca_new.reshape(1, -1))
    z_new = z_new.reshape(1, 64, latent_size, latent_size)

    # 用Decoder生成图片
    z_tensor = torch.FloatTensor(z_new).to(CONFIG["device"])
    with torch.no_grad():
        img = vqvae.decode(z_tensor)

    # 转换为PIL图片
    img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())

    # 量化到32色
    if img_pil.mode == "RGBA":
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.FASTOCTREE)
    else:
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.MEDIANCUT)

    result = img_quantized.convert("RGBA")

    # 应用调色板
    if palette and palette != "original":
        result = apply_palette(result, palette, intensity, saturation)

    return result, used_seed


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>纹理浏览器</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
        .header { margin-bottom: 20px; }
        .controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .control-group { background: #2a2a2a; padding: 15px; border-radius: 8px; }
        .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .control-group input, .control-group select { width: 100%; padding: 8px; }
        .preview { display: flex; gap: 20px; align-items: flex-start; }
        .main-preview { background: #2a2a2a; padding: 15px; border-radius: 8px; }
        .main-preview img { width: 256px; height: 256px; image-rendering: pixelated; }
        .samples { background: #2a2a2a; padding: 15px; border-radius: 8px; }
        .samples-grid { display: grid; grid-template-columns: repeat(4, 64px); gap: 5px; }
        .samples-grid img { width: 64px; height: 64px; image-rendering: pixelated; cursor: pointer; border: 2px solid transparent; }
        .samples-grid img:hover { border-color: #4CAF50; }
        .samples-grid img.selected { border-color: #4CAF50; }
        .sample-item:hover { background: #444; border-radius: 4px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        .info { background: #333; padding: 10px; border-radius: 4px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>纹理浏览器</h1>
        <p>选择基础样本，调整扰动强度，生成新纹理</p>
    </div>

    <div class="controls">
        <div class="control-group" style="max-width: 100%;">
            <label>基础样本 (点击选择)</label>
            <div id="sample-grid" style="display: flex; flex-wrap: wrap; gap: 5px; max-height: 200px; overflow-y: auto; padding: 5px; background: #333; border-radius: 4px;">
                {% for i in range(num_samples) %}
                <div class="sample-item" data-index="{{i}}" onclick="selectSample({{i}})" style="cursor: pointer; border: 2px solid transparent; padding: 2px; border-radius: 4px;">
                    <img src="/thumbnail/{{i}}" style="width: 32px; height: 32px; image-rendering: pixelated; display: block;" title="{{names[i]}}">
                </div>
                {% endfor %}
            </div>
            <div style="margin-top: 5px; font-size: 12px; color: #888;">
                已选: <span id="selected-name">{{names[0]}}</span>
            </div>
        </div>
        <div class="control-group">
            <label>扰动强度 (σ): <span id="sigma-value">0.5</span></label>
            <input type="range" id="sigma" min="0" max="3" step="0.1" value="0.5" onchange="updateSigma(); updatePreview()">
        </div>
        <div class="control-group">
            <label>随机种子</label>
            <input type="number" id="seed" value="" placeholder="留空随机">
        </div>
        <div class="control-group">
            <label>&nbsp;</label>
            <button onclick="generateBatch()">生成4张变体</button>
        </div>
        <div class="control-group">
            <label>&nbsp;</label>
            <button onclick="saveCurrent()">保存当前</button>
        </div>
        <div class="control-group">
            <label>调色板 (Palette)</label>
            <select id="palette" onchange="updatePreview()">
                <option value="original">原始调色板</option>
                {% for p in palettes %}
                <option value="{{p.id}}">{{p.name}} - {{p.description}}</option>
                {% endfor %}
            </select>
        </div>
        <div class="control-group">
            <label>调色板强度: <span id="intensity-value">1.0</span></label>
            <input type="range" id="intensity" min="0" max="1" step="0.05" value="1.0" onchange="updateIntensity(); updatePreview()">
        </div>
        <div class="control-group">
            <label>饱和度: <span id="saturation-value">1.0</span></label>
            <input type="range" id="saturation" min="0" max="2" step="0.05" value="1.0" onchange="updateSaturation(); updatePreview()">
        </div>
    </div>

    <div class="preview">
        <div class="main-preview">
            <h3>预览</h3>
            <img id="preview-img" src="/preview?base=0&sigma=0.5" alt="Preview">
            <div class="info">
                <div>基础: <span id="base-name">{{names[0]}}</span></div>
                <div>σ: <span id="current-sigma">0.5</span></div>
            </div>
        </div>
        <div class="samples">
            <h3>生成的变体</h3>
            <div class="samples-grid" id="samples-grid">
            </div>
        </div>
    </div>

    <script>
        let selectedSample = 0;

        function updateSigma() {
            const sigma = document.getElementById('sigma').value;
            document.getElementById('sigma-value').textContent = sigma;
            document.getElementById('current-sigma').textContent = sigma;
        }

        function updateIntensity() {
            document.getElementById('intensity-value').textContent = document.getElementById('intensity').value;
        }

        function updateSaturation() {
            document.getElementById('saturation-value').textContent = document.getElementById('saturation').value;
        }

        function selectSample(index) {
            document.querySelectorAll('.sample-item').forEach(item => {
                item.style.borderColor = 'transparent';
            });
            const item = document.querySelector(`.sample-item[data-index="${index}"]`);
            item.style.borderColor = '#4CAF50';
            selectedSample = index;
            document.getElementById('selected-name').textContent = item.querySelector('img').title;
            updatePreview();
        }

        function getParams() {
            return {
                sigma: document.getElementById('sigma').value,
                seed: document.getElementById('seed').value,
                palette: document.getElementById('palette').value,
                intensity: document.getElementById('intensity').value,
                saturation: document.getElementById('saturation').value,
            };
        }

        function updatePreview() {
            const p = getParams();
            let url = `/preview?base=${selectedSample}&sigma=${p.sigma}`;
            if (p.seed) url += `&seed=${p.seed}`;
            if (p.palette !== 'original') {
                url += `&palette=${p.palette}&intensity=${p.intensity}&saturation=${p.saturation}`;
            }
            document.getElementById('preview-img').src = url;
            document.getElementById('base-name').textContent = document.getElementById('selected-name').textContent;
        }

        function generateBatch() {
            const p = getParams();
            fetch('/generate_batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    base: selectedSample,
                    sigma: parseFloat(p.sigma),
                    count: 4,
                    palette: p.palette,
                    intensity: parseFloat(p.intensity),
                    saturation: parseFloat(p.saturation)
                })
            })
            .then(response => response.json())
            .then(data => {
                const grid = document.getElementById('samples-grid');
                grid.innerHTML = '';
                data.images.forEach((img, i) => {
                    const imgEl = document.createElement('img');
                    imgEl.src = `/generated/${img.filename}`;
                    imgEl.title = `Seed: ${img.seed}`;
                    imgEl.onclick = function() {
                        document.getElementById('preview-img').src = this.src;
                        document.getElementById('seed').value = img.seed;
                    };
                    grid.appendChild(imgEl);
                });
            });
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
    """主页"""
    palettes = list_palettes()
    return render_template_string(
        HTML_TEMPLATE,
        num_samples=len(names),
        names=names,
        palettes=palettes,
    )


@app.route('/preview')
def preview():
    """预览纹理"""
    base_idx = request.args.get('base', 0, type=int)
    sigma = request.args.get('sigma', 0.5, type=float)
    seed = request.args.get('seed', None, type=int)
    palette = request.args.get('palette', None, type=str)
    intensity = request.args.get('intensity', 1.0, type=float)
    saturation = request.args.get('saturation', 1.0, type=float)

    img, used_seed = generate_texture(base_idx, sigma, seed, palette, intensity, saturation)

    # 转换为bytes
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


@app.route('/generate_batch', methods=['POST'])
def generate_batch():
    """批量生成"""
    data = request.json
    base_idx = data.get('base', 0)
    sigma = data.get('sigma', 0.5)
    count = data.get('count', 4)
    palette = data.get('palette', None)
    intensity = data.get('intensity', 1.0)
    saturation = data.get('saturation', 1.0)

    # 创建输出目录
    output_dir = project_root / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(count):
        img, used_seed = generate_texture(base_idx, sigma, palette=palette,
                                          intensity=intensity, saturation=saturation)

        # 保存
        pal_suffix = f"_{palette}" if palette and palette != "original" else ""
        filename = f"texture_{base_idx}_{sigma}_{used_seed}{pal_suffix}.png"
        img.save(output_dir / filename)

        images.append({"filename": filename, "seed": used_seed})

    return jsonify({"images": images})


@app.route('/generated/<filename>')
def generated(filename):
    """返回生成的图片"""
    output_dir = project_root / CONFIG["output_dir"]
    return send_file(output_dir / filename)


@app.route('/thumbnail/<int:base_idx>')
def thumbnail(base_idx):
    """返回基础样本的缩略图"""
    if base_idx >= len(names):
        return "Not found", 404

    # 用原始latent生成缩略图
    z_original = latents_pca[base_idx]
    z_new = pca.inverse_transform(z_original.reshape(1, -1))
    z_new = z_new.reshape(1, 64, latent_size, latent_size)

    z_tensor = torch.FloatTensor(z_new).to(CONFIG["device"])
    with torch.no_grad():
        img = vqvae.decode(z_tensor)

    img_pil = transforms.ToPILImage()(img.squeeze(0).cpu())

    # 量化并缩小
    if img_pil.mode == "RGBA":
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.FASTOCTREE)
    else:
        img_quantized = img_pil.quantize(colors=32, method=Image.Quantize.MEDIANCUT)

    img_small = img_quantized.convert("RGBA").resize((32, 32), Image.Resampling.NEAREST)

    img_io = BytesIO()
    img_small.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


def main():
    print("=" * 60)
    print("纹理浏览器")
    print("=" * 60)

    load_models()

    print(f"\n启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

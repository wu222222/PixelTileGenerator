"""
Cluster Navigation Browser

交互式探索 latent space 的簇间结构:
- 选择两个簇，滑杆控制插值 α
- 实时预览结构过渡
- 集成 Palette 系统: 同一结构 × 不同调色板
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v2 import VQVAEv2
from scripts.tools.palette_remapper import list_palettes, apply_palette

# 配置
CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v7/vqvae_v7_best.pth",
    "latent_data": "datasets/vqvae_latent_data_v7",
    "data_dir": "datasets/classified/pixel_32_quantized",
    "output_dir": "generated_textures",
    "port": 5003,
    "n_clusters": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

app = Flask(__name__)

# 全局变量
vqvae = None
latents = None
names = None
labels = None
cluster_centers = None
cluster_info = None
cluster_reps = None
images_cache = None
cluster_dist_matrix = None
cluster_paths = None


def get_category(name):
    stem = Path(name).stem
    if "_from_" in stem:
        return stem.split("_from_")[0]
    return stem.split("_")[0]


def load_models():
    global vqvae, latents, names, labels, cluster_centers, cluster_info, cluster_reps, images_cache, cluster_dist_matrix, cluster_paths

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

    # 加载 latent 数据
    latent_dir = project_root / CONFIG["latent_data"]
    latents = np.load(latent_dir / "latents.npy")
    names = json.load(open(latent_dir / "names.json"))
    print(f"加载 {len(names)} 个样本")

    # KMeans 聚类
    print("KMeans 聚类...")
    flat = latents.reshape(latents.shape[0], -1)
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)
    kmeans = KMeans(n_clusters=CONFIG["n_clusters"], random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_pca)

    # 簇中心 (原始 latent 空间)
    cluster_centers = {}
    for c in range(CONFIG["n_clusters"]):
        mask = labels == c
        cluster_centers[c] = latents[mask].mean(axis=0)

    # 簇信息
    cluster_info = []
    for c in range(CONFIG["n_clusters"]):
        mask = labels == c
        indices = np.where(mask)[0]
        cats = defaultdict(int)
        for i in indices:
            cats[get_category(names[i])] += 1
        top_cats = sorted(cats.items(), key=lambda x: -x[1])
        primary = top_cats[0][0] if top_cats else "unknown"

        # 给簇一个可读名称
        cat_display = ", ".join(f"{k}({v})" for k, v in top_cats[:3])
        cluster_info.append({
            "id": c,
            "size": int(mask.sum()),
            "primary": primary,
            "categories": cat_display,
        })

    # 每个簇最靠近中心的样本
    cluster_reps = {}
    for c in range(CONFIG["n_clusters"]):
        mask = labels == c
        indices = np.where(mask)[0]
        center = cluster_centers[c].flatten()
        dists = np.linalg.norm(latents[indices].reshape(len(indices), -1) - center, axis=1)
        sorted_idx = indices[np.argsort(dists)]
        cluster_reps[c] = sorted_idx[:5].tolist()

    # 簇间距离矩阵
    n_c = CONFIG["n_clusters"]
    cluster_dist_matrix = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            cluster_dist_matrix[i, j] = np.linalg.norm(
                cluster_centers[i].flatten() - cluster_centers[j].flatten())

    # 预计算所有簇对的最短路径 (Dijkstra)
    cluster_paths = {}
    for start in range(n_c):
        for end in range(n_c):
            if start == end:
                cluster_paths[(start, end)] = [start]
                continue
            # Dijkstra
            visited = set()
            dist = {start: 0}
            prev = {start: None}
            unvisited = set(range(n_c))
            while unvisited:
                u = min(unvisited, key=lambda x: dist.get(x, float('inf')))
                if u not in dist:
                    break
                unvisited.remove(u)
                visited.add(u)
                for v in range(n_c):
                    if v not in visited:
                        new_dist = dist[u] + cluster_dist_matrix[u, v]
                        if new_dist < dist.get(v, float('inf')):
                            dist[v] = new_dist
                            prev[v] = u
            # 回溯路径
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = prev[node]
            path.reverse()
            cluster_paths[(start, end)] = path

    # 预加载图片
    print("缓存图片...")
    data_dir = project_root / CONFIG["data_dir"]
    images_cache = {}
    for name in names:
        path = data_dir / name
        if path.exists():
            images_cache[name] = Image.open(path).convert("RGBA")

    print(f"加载完成: {CONFIG['n_clusters']} 个簇")


def decode_latent(z_np):
    """解码 latent 为 PIL 图片"""
    device = CONFIG["device"]
    z_tensor = torch.FloatTensor(z_np).unsqueeze(0).to(device)
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
    <title>Cluster Navigation Browser</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
        h1 { margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 14px; }
        .main { display: flex; gap: 20px; }
        .left { flex: 1; }
        .right { width: 300px; }
        .panel { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .panel h3 { margin-bottom: 10px; font-size: 14px; color: #aaa; }
        .preview-box { text-align: center; background: #222; padding: 10px; border-radius: 8px; }
        .preview-box img { width: 256px; height: 256px; image-rendering: pixelated; }
        .transition-grid { display: flex; gap: 4px; justify-content: center; margin-top: 10px; flex-wrap: wrap; }
        .transition-grid img { width: 48px; height: 48px; image-rendering: pixelated; border: 1px solid #444; cursor: pointer; }
        .transition-grid img:hover { border-color: #4CAF50; }
        .transition-grid img.active { border-color: #4CAF50; border-width: 2px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; font-size: 13px; }
        select, input[type=range] { width: 100%; padding: 6px; margin-bottom: 10px; }
        .slider-row { display: flex; align-items: center; gap: 10px; }
        .slider-row input { flex: 1; }
        .slider-val { min-width: 35px; text-align: right; font-family: monospace; }
        button { padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 5px; }
        button:hover { background: #45a049; }
        button.secondary { background: #555; }
        button.secondary:hover { background: #666; }
        .cluster-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
        .info { background: #333; padding: 8px; border-radius: 4px; font-size: 12px; margin-top: 10px; }
        .cluster-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; margin-top: 10px; }
        .cluster-grid .item { text-align: center; cursor: pointer; padding: 5px; border-radius: 4px; border: 2px solid transparent; }
        .cluster-grid .item:hover { border-color: #4CAF50; }
        .cluster-grid .item.selected { border-color: #4CAF50; background: #333; }
        .cluster-grid img { width: 48px; height: 48px; image-rendering: pixelated; display: block; margin: 0 auto; }
        .cluster-grid .label { font-size: 10px; color: #888; margin-top: 2px; }
        .arrow { font-size: 24px; color: #4CAF50; display: flex; align-items: center; justify-content: center; }
    </style>
</head>
<body>
    <h1>Cluster Navigation Browser</h1>
    <p class="subtitle">探索 Terrain Latent Space: 簇间插值 + Palette 换肤</p>

    <div class="main">
        <div class="left">
            <div class="preview-box">
                <img id="preview-img" src="/interpolate?c_from=7&c_to=5&alpha=0.5&mode=sample" alt="Preview">
            </div>
            <div class="info" id="info-text">
                Cluster 7 (Grass) → Cluster 5 (Desert), α = 0.5
            </div>

            <div class="panel">
                <h3>过渡序列 (点击选择 α)</h3>
                <div class="transition-grid" id="transition-grid"></div>
            </div>
        </div>

        <div class="right">
            <div class="panel">
                <h3>簇选择</h3>
                <label>起始簇 (From)</label>
                <select id="cluster-from" onchange="updatePreview()">
                    {% for c in clusters %}
                    <option value="{{c.id}}" {{'selected' if c.id == 7 else ''}}>C{{c.id}}: {{c.primary}} ({{c.size}})</option>
                    {% endfor %}
                </select>
                <label>目标簇 (To)</label>
                <select id="cluster-to" onchange="updatePreview()">
                    {% for c in clusters %}
                    <option value="{{c.id}}" {{'selected' if c.id == 5 else ''}}>C{{c.id}}: {{c.primary}} ({{c.size}})</option>
                    {% endfor %}
                </select>
            </div>

            <div class="panel">
                <h3>插值控制</h3>
                <label>α: <span id="alpha-val">0.50</span></label>
                <div class="slider-row">
                    <span style="font-size:12px">From</span>
                    <input type="range" id="alpha" min="0" max="1" step="0.01" value="0.5" oninput="updateAlpha(); updatePreview()">
                    <span style="font-size:12px">To</span>
                </div>
                <label>模式</label>
                <select id="mode" onchange="updatePreview()">
                    <option value="center">簇中心插值 (模糊)</option>
                    <option value="sample" selected>真实样本插值 (纹理保留)</option>
                    <option value="vector">语义编辑器 (样本+方向向量)</option>
                    <option value="path">路径导航 (沿簇拓扑行走)</option>
                </select>
                <div id="path-info" style="display:none; font-size:11px; color:#888; margin-top:5px;">
                    路径: <span id="path-text">-</span>
                </div>
            </div>

            <div class="panel">
                <h3>Palette 换肤</h3>
                <label>调色板</label>
                <select id="palette" onchange="updatePreview()">
                    <option value="original">原始调色板</option>
                    {% for p in palettes %}
                    <option value="{{p.id}}">{{p.name}}</option>
                    {% endfor %}
                </select>
                <label>强度: <span id="intensity-val">1.0</span></label>
                <input type="range" id="intensity" min="0" max="1" step="0.05" value="1.0"
                    oninput="document.getElementById('intensity-val').textContent=this.value; updatePreview()">
                <label>饱和度: <span id="saturation-val">1.0</span></label>
                <input type="range" id="saturation" min="0" max="2" step="0.05" value="1.0"
                    oninput="document.getElementById('saturation-val').textContent=this.value; updatePreview()">
            </div>

            <div class="panel">
                <h3>快速路径</h3>
                <button onclick="setPath(7,5)">Grass→Desert</button>
                <button onclick="setPath(7,4)">Grass→Stone</button>
                <button onclick="setPath(6,5)">Swamp→Desert</button>
                <button onclick="setPath(7,0)">Grass→Snow</button>
                <button onclick="setPath(0,5)">Snow→Desert</button>
            </div>

            <div class="panel">
                <h3>操作</h3>
                <button onclick="saveCurrent()">保存当前</button>
                <button class="secondary" onclick="randomAlpha()">随机 α</button>
            </div>
        </div>
    </div>

    <script>
        function setPath(from, to) {
            document.getElementById('cluster-from').value = from;
            document.getElementById('cluster-to').value = to;
            updatePreview();
        }

        function updateAlpha() {
            document.getElementById('alpha-val').textContent = parseFloat(document.getElementById('alpha').value).toFixed(2);
        }

        function updatePreview() {
            const c_from = document.getElementById('cluster-from').value;
            const c_to = document.getElementById('cluster-to').value;
            const alpha = document.getElementById('alpha').value;
            const mode = document.getElementById('mode').value;
            const palette = document.getElementById('palette').value;
            const intensity = document.getElementById('intensity').value;
            const saturation = document.getElementById('saturation').value;

            let url = `/interpolate?c_from=${c_from}&c_to=${c_to}&alpha=${alpha}&mode=${mode}`;
            if (palette !== 'original') {
                url += `&palette=${palette}&intensity=${intensity}&saturation=${saturation}`;
            }

            document.getElementById('preview-img').src = url;
            const modeNames = {center: 'Center', sample: 'Sample', vector: 'Vector', path: 'Path'};
            document.getElementById('info-text').textContent =
                `[${modeNames[mode]}] Cluster ${c_from} → Cluster ${c_to}, α = ${parseFloat(alpha).toFixed(2)}` +
                (palette !== 'original' ? ` | Palette: ${palette}` : '');

            // 路径模式: 显示路径信息
            const pathInfo = document.getElementById('path-info');
            if (mode === 'path') {
                pathInfo.style.display = 'block';
                fetch(`/path?c_from=${c_from}&c_to=${c_to}`)
                    .then(r => r.json())
                    .then(data => {
                        const clusterNames = {{ cluster_names | tojson }};
                        document.getElementById('path-text').textContent =
                            data.path.map(c => `C${c}(${clusterNames[c] || '?'})`).join(' → ');
                    });
            } else {
                pathInfo.style.display = 'none';
            }

            // 更新过渡网格高亮
            document.querySelectorAll('.transition-grid img').forEach(img => {
                const a = parseFloat(img.dataset.alpha);
                img.classList.toggle('active', Math.abs(a - parseFloat(alpha)) < 0.06);
            });
        }

        function buildTransitionGrid() {
            const grid = document.getElementById('transition-grid');
            grid.innerHTML = '';
            for (let i = 0; i <= 10; i++) {
                const a = (i / 10).toFixed(1);
                const img = document.createElement('img');
                img.src = `/interpolate?c_from=${document.getElementById('cluster-from').value}&c_to=${document.getElementById('cluster-to').value}&alpha=${a}&mode=${document.getElementById('mode').value}&thumb=1`;
                img.dataset.alpha = a;
                img.title = `α = ${a}`;
                img.onclick = function() {
                    document.getElementById('alpha').value = a;
                    updateAlpha();
                    updatePreview();
                };
                grid.appendChild(img);
            }
        }

        function saveCurrent() {
            const src = document.getElementById('preview-img').src;
            const a = document.createElement('a');
            a.href = src;
            a.download = 'cluster_texture.png';
            a.click();
        }

        function randomAlpha() {
            const a = Math.random().toFixed(2);
            document.getElementById('alpha').value = a;
            updateAlpha();
            updatePreview();
        }

        // 初始化
        updatePreview();
        buildTransitionGrid();

        // 簇变化时重建过渡网格
        document.getElementById('cluster-from').addEventListener('change', buildTransitionGrid);
        document.getElementById('cluster-to').addEventListener('change', buildTransitionGrid);
        document.getElementById('mode').addEventListener('change', buildTransitionGrid);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    cluster_names = {c["id"]: c["primary"] for c in cluster_info}
    return render_template_string(
        HTML_TEMPLATE,
        clusters=cluster_info,
        palettes=list_palettes(),
        cluster_names=cluster_names,
    )


@app.route('/interpolate')
def interpolate():
    c_from = request.args.get('c_from', 7, type=int)
    c_to = request.args.get('c_to', 5, type=int)
    alpha = request.args.get('alpha', 0.5, type=float)
    mode = request.args.get('mode', 'center', type=str)
    palette = request.args.get('palette', None, type=str)
    intensity = request.args.get('intensity', 1.0, type=float)
    saturation = request.args.get('saturation', 1.0, type=float)
    thumb = request.args.get('thumb', None, type=str)

    if mode == 'sample':
        # 真实样本插值 (纹理保留)
        idx_from = cluster_reps[c_from][0]
        idx_to = cluster_reps[c_to][0]
        z_from = latents[idx_from]
        z_to = latents[idx_to]
        z_interp = z_from * (1 - alpha) + z_to * alpha
    elif mode == 'vector':
        # 语义编辑器: 真实样本 + 类别方向向量
        idx_base = cluster_reps[c_from][0]
        z_base = latents[idx_base]
        direction = cluster_centers[c_to] - cluster_centers[c_from]
        z_interp = z_base + alpha * direction
    elif mode == 'path':
        # 路径导航: 沿簇拓扑最短路径逐段插值
        path = cluster_paths.get((c_from, c_to), [c_from, c_to])
        n_segments = len(path) - 1
        if n_segments == 0:
            z_interp = latents[cluster_reps[c_from][0]]
        else:
            # 将 alpha 映射到具体段
            segment_len = 1.0 / n_segments
            seg_idx = min(int(alpha / segment_len), n_segments - 1)
            local_alpha = (alpha - seg_idx * segment_len) / segment_len
            local_alpha = min(local_alpha, 1.0)

            c_a = path[seg_idx]
            c_b = path[seg_idx + 1]
            z_a = latents[cluster_reps[c_a][0]]
            z_b = latents[cluster_reps[c_b][0]]
            z_interp = z_a * (1 - local_alpha) + z_b * local_alpha
    else:
        # 簇中心插值
        z_from = cluster_centers[c_from]
        z_to = cluster_centers[c_to]
        z_interp = z_from * (1 - alpha) + z_to * alpha
    img = decode_latent(z_interp)

    if thumb:
        img = img.resize((48, 48), Image.Resampling.NEAREST)

    if palette and palette != "original":
        img = apply_palette(img, palette, intensity, saturation)

    return send_file(img_to_bytes(img), mimetype='image/png')


@app.route('/path')
def get_path():
    """返回两个簇之间的最短路径"""
    c_from = request.args.get('c_from', 7, type=int)
    c_to = request.args.get('c_to', 5, type=int)
    path = cluster_paths.get((c_from, c_to), [c_from])
    return jsonify({"path": path})


@app.route('/cluster_samples/<int:c_id>')
def cluster_samples(c_id):
    """返回簇的代表样本缩略图"""
    reps = cluster_reps.get(c_id, [])
    if not reps:
        return "Not found", 404

    tile_size = 48
    gap = 2
    n = min(5, len(reps))
    canvas = Image.new("RGBA", (n * tile_size + (n - 1) * gap, tile_size), (40, 40, 40, 255))

    for i, idx in enumerate(reps):
        name = names[idx]
        if name in images_cache:
            img = images_cache[name].resize((tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(img, (i * (tile_size + gap), 0))

    return send_file(img_to_bytes(canvas), mimetype='image/png')


def main():
    print("=" * 60)
    print("Cluster Navigation Browser")
    print("=" * 60)

    load_models()

    print(f"\n启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

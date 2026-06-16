"""
PixelTile Shader Graph — 节点式纹理生成器

Flask + HTML5 Canvas 节点画布
"""

import sys
import json
import base64
import numpy as np
from pathlib import Path
from io import BytesIO

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vq_vae_v4 import VQVAEv4
from scripts.tools.shader_graph.nodes import get_all_node_schemas, get_palette_options, get_tile_categories
from scripts.tools.shader_graph.executor import GraphExecutor
from scripts.tools.shader_graph.presets import list_presets, get_preset

CONFIG = {
    "vqvae_checkpoint": "checkpoints/vqvae_v9/vqvae_v9_best.pth",
    "zq_data": "datasets/vqvae_v9_zq_data",
    "port": 5005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

app = Flask(__name__)

# 全局
executor = None
model_context = {}


def load_models():
    global executor, model_context
    device = CONFIG["device"]

    ckpt_path = project_root / CONFIG["vqvae_checkpoint"]
    print(f"加载 VQ-VAE v9: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    latent_size = config.get("latent_size", 16)

    model = VQVAEv4(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 256),
        embedding_dim=config.get("embedding_dim", 128),
        num_embeddings=config.get("num_embeddings", 1024),
        latent_size=latent_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 加载 z_q (连续向量)
    zq_dir = project_root / CONFIG["zq_data"]
    zq_vectors = np.load(zq_dir / "z_q.npy")
    names = json.load(open(zq_dir / "names.json"))

    # 用 z_q 做聚类
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    flat = zq_vectors.reshape(zq_vectors.shape[0], -1)
    pca = PCA(n_components=50)
    flat_pca = pca.fit_transform(flat)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_pca)

    cluster_centers = {}
    cluster_reps = {}
    for c in range(8):
        mask = labels == c
        cluster_centers[c] = zq_vectors[mask].mean(axis=0)
        indices = np.where(mask)[0]
        center = cluster_centers[c].flatten()
        dists = np.linalg.norm(zq_vectors[indices].reshape(len(indices), -1) - center, axis=1)
        sorted_idx = indices[np.argsort(dists)]
        cluster_reps[c] = sorted_idx[:5].tolist()

    # 缓存 64x64 图片
    print("缓存图片...")
    data_dir = project_root / "datasets" / "classified" / "pixel_64_quantized"
    images_cache = {}
    for name in names:
        path = data_dir / name
        if path.exists():
            images_cache[name] = Image.open(path).convert("RGBA")

    model_context = {
        "_model": model,
        "_device": device,
        "_latents": zq_vectors,
        "_names": names,
        "_cluster_reps": cluster_reps,
        "_images_cache": images_cache,
    }

    executor = GraphExecutor(model_context)
    print(f"加载完成: {len(names)} 样本, 8 簇")


def img_to_base64(img):
    buf = BytesIO()
    img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    schemas = get_all_node_schemas()
    palettes = get_palette_options()
    presets = list_presets()
    tile_categories = get_tile_categories()
    # 更新 base_tile 节点的 category 选项
    if "base_tile" in schemas:
        schemas["base_tile"]["params"]["category"]["options"] = tile_categories
    return render_template('index.html',
                           schemas=schemas,
                           palettes=palettes,
                           presets=presets,
                           tile_categories=tile_categories)


@app.route('/api/execute', methods=['POST'])
def api_execute():
    graph = request.json
    try:
        outputs = executor.execute(graph)
        result = {}
        for node_id, img in outputs.items():
            if img is not None:
                result[node_id] = img_to_base64(img)
        return jsonify({"success": True, "images": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/presets')
def api_presets():
    return jsonify(list_presets())


@app.route('/api/preset/<name>')
def api_preset(name):
    p = get_preset(name)
    if p:
        return jsonify(p["graph"])
    return jsonify({"error": "not found"}), 404


@app.route('/api/tiles')
def api_tiles():
    category = request.args.get('category', 'all')
    page = int(request.args.get('page', 0))
    per_page = int(request.args.get('per_page', 20))

    names = model_context.get("_names", [])
    images_cache = model_context.get("_images_cache", {})

    def get_cat(n):
        stem = n.rsplit('.', 1)[0]
        if "_from_" in stem:
            return stem.split("_from_")[0]
        return stem.split("_")[0]

    if category and category != "all":
        filtered = [(i, n) for i, n in enumerate(names) if get_cat(n) == category]
    else:
        filtered = list(enumerate(names))

    total = len(filtered)
    start = page * per_page
    end = start + per_page
    page_items = filtered[start:end]

    tiles = []
    for local_idx, (real_idx, name) in enumerate(page_items):
        thumb_b64 = ""
        img = images_cache.get(name)
        if img is not None:
            thumb = img.resize((64, 64), Image.Resampling.NEAREST)
            thumb_b64 = img_to_base64(thumb)
        tiles.append({"name": name, "index": start + local_idx, "thumb": thumb_b64})

    return jsonify({"tiles": tiles, "total": total, "page": page, "per_page": per_page})


@app.route('/api/tile_preview/<int:index>')
def api_tile_preview(index):
    names = model_context.get("_names", [])
    images_cache = model_context.get("_images_cache", {})
    if index < 0 or index >= len(names):
        return jsonify({"error": "index out of range"}), 404
    name = names[index]
    img = images_cache.get(name)
    if img is None:
        return jsonify({"error": "image not found"}), 404
    return jsonify({"name": name, "image": img_to_base64(img)})


@app.route('/api/preview', methods=['POST'])
def api_preview():
    graph = request.json
    try:
        outputs = executor.execute(graph)
        for node_id, img in outputs.items():
            if img is not None:
                return jsonify({"success": True, "image": img_to_base64(img)})
        return jsonify({"success": False, "error": "no output"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/export', methods=['POST'])
def api_export():
    data = request.json
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"success": False, "error": "no image"})
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes))
        export_dir = project_root / "generated_textures"
        export_dir.mkdir(exist_ok=True)
        import time
        ts = int(time.time() * 1000) % 1000000
        filename = f"texture_{ts}.png"
        path = export_dir / filename
        img.save(path, "PNG")
        return jsonify({"success": True, "path": str(path), "filename": filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============================================================
# HTML Template — 节点画布
# 模板已移至 templates/index.html
# ============================================================


def main():
    print("=" * 60)
    print("PixelTile Shader Graph")
    print("=" * 60)
    load_models()
    print(f"\n启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

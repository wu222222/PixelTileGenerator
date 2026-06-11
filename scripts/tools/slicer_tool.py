"""
瓦片切分工具

功能:
- 逐个显示spritesheet图片
- 支持多次resize和切分操作
- 切分后可选择瓦片
- 支持撤销到任意步骤
- 用户点击"完成"才保存并切换到下一张
"""

import os
import json
import shutil
import copy
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from io import BytesIO

# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent.parent,
    "input_dir": "datasets/classified/spritesheet",
    "output_dir": "datasets/classified/pixel_32",
    "port": 5001,
}

app = Flask(__name__, template_folder='templates')

# 全局状态
images = []
current_index = 0
status_file = None
image_status = {}


def load_status():
    """加载处理状态"""
    global image_status, status_file
    status_file = CONFIG["project_root"] / CONFIG["input_dir"] / "process_status.json"
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            image_status = json.load(f)
    else:
        image_status = {}


def save_status():
    """保存处理状态"""
    global image_status, status_file
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(image_status, f, indent=2, ensure_ascii=False)


def scan_images():
    """扫描所有spritesheet图片"""
    global images
    images = []

    input_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    if not input_dir.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    load_status()

    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in image_extensions:
            status = image_status.get(f.name, {})
            images.append({
                "id": len(images),
                "path": str(f),
                "name": f.name,
                "processed": status.get("processed", False),
                "skipped": status.get("skipped", False),
            })

    processed_count = sum(1 for i in images if i['processed'])
    skipped_count = sum(1 for i in images if i['skipped'])
    pending_count = len(images) - processed_count - skipped_count

    print(f"扫描完成: 找到 {len(images)} 个图片 (已处理: {processed_count}, 已跳过: {skipped_count}, 待处理: {pending_count})")


def get_image_info(img_path: str) -> dict:
    """获取图片信息"""
    try:
        img = Image.open(img_path)
        width, height = img.size

        try:
            colors = len(img.getcolors(maxcolors=256))
        except TypeError:
            colors = 257

        return {
            "width": width,
            "height": height,
            "colors": colors,
        }
    except Exception as e:
        return {"error": str(e)}


def split_image(img: Image.Image, tile_size: int) -> list:
    """切分图片为多个瓦片"""
    try:
        width, height = img.size
        tiles = []
        tile_id = 0

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if x + tile_size <= width and y + tile_size <= height:
                    tile = img.crop((x, y, x + tile_size, y + tile_size))
                    tiles.append({
                        "id": tile_id,
                        "x": x,
                        "y": y,
                        "image": tile,
                    })
                    tile_id += 1

        return tiles
    except Exception as e:
        print(f"切分失败: {e}")
        return []


def resize_image(img: Image.Image, target_size: int) -> Image.Image:
    """resize图片"""
    try:
        return img.resize((target_size, target_size), Image.NEAREST)
    except Exception as e:
        print(f"resize失败: {e}")
        return None


@app.route('/')
def index():
    """主页"""
    return render_template('slicer.html')


@app.route('/api/init')
def api_init():
    """初始化"""
    scan_images()

    # 统计待处理数量
    pending_count = sum(1 for i in images if not i.get("processed") and not i.get("skipped"))

    return jsonify({
        "total": len(images),
        "pending": pending_count,
        "images": [{
            "name": i["name"],
            "processed": i.get("processed", False),
            "skipped": i.get("skipped", False),
        } for i in images],
    })


@app.route('/api/image/<int:image_id>')
def api_image(image_id):
    """获取图片信息"""
    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]

    # 优先使用临时文件的信息
    img_path = image.get("temp_path", image["path"])
    if not Path(img_path).exists():
        img_path = image["path"]

    info = get_image_info(img_path)

    return jsonify({
        "name": image["name"],
        "processed": image.get("processed", False),
        "original_width": get_image_info(image["path"]).get("width"),
        "original_height": get_image_info(image["path"]).get("height"),
        **info,
    })


@app.route('/api/image/<int:image_id>/file')
def api_image_file(image_id):
    """返回图片文件（优先返回临时文件）"""
    if image_id >= len(images):
        return "Not found", 404

    image = images[image_id]

    # 检查临时文件是否存在
    temp_path = image.get("temp_path")
    if temp_path and Path(temp_path).exists():
        return send_file(temp_path)

    # 否则返回原图
    return send_file(image["path"])


@app.route('/api/resize', methods=['POST'])
def api_resize():
    """resize图片"""
    data = request.json
    image_id = data.get('index')
    size = data.get('size', 32)

    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]

    # 获取当前图片（可能是临时文件）
    img_path = image.get("temp_path", image["path"])
    img = Image.open(img_path)

    # 检查是否有选中的瓦片需要resize
    tiles_to_resize = image.get("tiles", [])
    selected_tiles = set(data.get("selected_tiles", []))

    if tiles_to_resize and selected_tiles:
        # 对选中的瓦片进行resize
        resized_tiles = []
        for tile_data in tiles_to_resize:
            if tile_data["id"] in selected_tiles:
                tile_img = Image.open(tile_data["path"])
                resized_tile = resize_image(tile_img, size)
                if resized_tile:
                    # 保存resize后的瓦片
                    resized_path = CONFIG["project_root"] / CONFIG["input_dir"] / f"_resized_{tile_data['id']}_{image['name']}"
                    resized_tile.save(resized_path, "PNG")
                    resized_tiles.append({
                        "id": tile_data["id"],
                        "x": tile_data["x"],
                        "y": tile_data["y"],
                        "path": str(resized_path),
                    })

        # 更新瓦片数据
        image["tiles"] = resized_tiles
        image["tile_size"] = size

        # 重新组合瓦片成一张图，用于预览
        if resized_tiles:
            # 计算组合后的图片尺寸
            max_x = max(t["x"] for t in resized_tiles) + size
            max_y = max(t["y"] for t in resized_tiles) + size

            # 创建新图片
            combined_img = Image.new("RGBA", (max_x, max_y), (0, 0, 0, 0))

            # 粘贴每个瓦片
            for tile_data in resized_tiles:
                tile_img = Image.open(tile_data["path"])
                combined_img.paste(tile_img, (tile_data["x"], tile_data["y"]))

            # 保存组合后的图片作为临时文件
            temp_path = CONFIG["project_root"] / CONFIG["input_dir"] / f"_temp_{image['name']}"
            combined_img.save(temp_path, "PNG")
            images[image_id]["temp_path"] = str(temp_path)

        return jsonify({
            "success": True,
            "message": f"已将 {len(resized_tiles)} 个瓦片resize到 {size}×{size}",
            "tiles": [{"id": t["id"], "x": t["x"], "y": t["y"]} for t in resized_tiles],
            "tile_size": size,
        })
    else:
        # 对整张图进行resize
        resized = resize_image(img, size)
        if resized is None:
            return jsonify({"error": "resize失败"})

        # 保存到临时文件
        temp_path = CONFIG["project_root"] / CONFIG["input_dir"] / f"_temp_{image['name']}"
        resized.save(temp_path, "PNG")
        images[image_id]["temp_path"] = str(temp_path)

        return jsonify({
            "success": True,
            "message": f"已resize到 {size}×{size}",
        })


@app.route('/api/skip', methods=['POST'])
def api_skip():
    """跳过图片（清理临时文件）"""
    data = request.json
    image_id = data.get('index')

    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]

    # 清理临时文件
    temp_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    for temp_file in temp_dir.glob(f"_temp_{image['name']}"):
        temp_file.unlink()
    for temp_file in temp_dir.glob(f"_tile_*_{image['name']}"):
        temp_file.unlink()
    for temp_file in temp_dir.glob(f"_resized_*_{image['name']}"):
        temp_file.unlink()

    # 清除临时路径
    if "temp_path" in images[image_id]:
        del images[image_id]["temp_path"]
    if "tiles" in images[image_id]:
        del images[image_id]["tiles"]

    # 标记为已跳过
    images[image_id]["skipped"] = True
    image_status[image["name"]] = {
        "processed": False,
        "skipped": True,
    }
    save_status()

    return jsonify({
        "success": True,
        "message": "已跳过并清理临时文件",
    })


@app.route('/api/split', methods=['POST'])
def api_split():
    """切分图片"""
    data = request.json
    image_id = data.get('index')
    tile_size = data.get('tile_size', 32)

    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]

    # 清理之前的临时瓦片文件
    temp_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    for temp_file in temp_dir.glob(f"_tile_*_{image['name']}"):
        temp_file.unlink()

    # 使用临时文件（如果有）或原图
    img_path = image.get("temp_path", image["path"])
    if not Path(img_path).exists():
        img_path = image["path"]

    img = Image.open(img_path)

    tiles = split_image(img, tile_size)

    if not tiles:
        return jsonify({"error": "切分失败"})

    # 保存瓦片数据到临时文件
    tiles_data = []
    for tile in tiles:
        tile_path = temp_dir / f"_tile_{tile['id']}_{image['name']}"
        tile["image"].save(tile_path, "PNG")
        tiles_data.append({
            "id": tile["id"],
            "x": tile["x"],
            "y": tile["y"],
            "path": str(tile_path),
        })

    # 保存瓦片信息到全局
    images[image_id]["tiles"] = tiles_data
    images[image_id]["tile_size"] = tile_size

    return jsonify({
        "success": True,
        "tiles": [{"id": t["id"], "x": t["x"], "y": t["y"]} for t in tiles_data],
        "current_size": {"width": img.width, "height": img.height},
    })


@app.route('/api/finish', methods=['POST'])
def api_finish():
    """完成图片处理"""
    data = request.json
    image_id = data.get('index')
    operations = data.get('operations', [])
    selected_tile_ids = set(data.get('selected_tiles', []))
    tile_size = data.get('tile_size', 32)

    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]
    output_dir = CONFIG["project_root"] / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    name_stem = Path(image["name"]).stem
    saved_count = 0

    # 检查是否有切分操作
    has_split = any(op['type'] == 'split' for op in operations)

    if has_split and selected_tile_ids:
        # 保存选中的瓦片
        tiles_data = image.get("tiles", [])
        for tile_data in tiles_data:
            if tile_data["id"] in selected_tile_ids:
                tile_path = Path(tile_data["path"])
                if tile_path.exists():
                    output_path = output_dir / f"{name_stem}_{tile_data['id']:04d}.png"
                    shutil.copy2(tile_path, output_path)
                    saved_count += 1
    else:
        # 没有切分或没有选择瓦片，保存整张图
        img_path = image.get("temp_path", image["path"])
        if Path(img_path).exists():
            output_path = output_dir / image["name"]
            shutil.copy2(img_path, output_path)
            saved_count = 1

    # 清理临时文件
    temp_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    for temp_file in temp_dir.glob(f"_temp_{image['name']}"):
        temp_file.unlink()
    for temp_file in temp_dir.glob(f"_tile_*_{image['name']}"):
        temp_file.unlink()
    for temp_file in temp_dir.glob(f"_resized_*_{image['name']}"):
        temp_file.unlink()

    # 更新状态
    image_status[image["name"]] = {
        "processed": True,
        "operations": operations,
        "selected_tiles": list(selected_tile_ids),
        "saved_count": saved_count,
    }
    save_status()

    return jsonify({
        "success": True,
        "message": f"完成！已保存 {saved_count} 个文件",
    })


@app.route('/api/delete', methods=['POST'])
def api_delete():
    """标记删除图片"""
    data = request.json
    image_id = data.get('index')

    if image_id >= len(images):
        return jsonify({"error": "图片不存在"})

    image = images[image_id]

    image_status[image["name"]] = {
        "processed": True,
        "action": "delete",
    }
    save_status()

    return jsonify({
        "success": True,
        "message": "已标记删除",
    })


@app.route('/api/import', methods=['POST'])
def api_import():
    """导入外部PNG图片"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"})

    # 检查文件类型
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"不支持的文件类型: {ext}"})

    # 保存文件
    input_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    input_dir.mkdir(parents=True, exist_ok=True)

    save_path = input_dir / file.filename
    file.save(save_path)

    # 重新扫描图片
    scan_images()

    return jsonify({
        "success": True,
        "message": f"导入成功: {file.filename}",
        "total": len(images),
    })


@app.route('/api/import_folder', methods=['POST'])
def api_import_folder():
    """导入文件夹中的所有图片"""
    data = request.json
    folder_path = data.get('folder_path', '')

    if not folder_path:
        return jsonify({"error": "文件夹路径为空"})

    folder = Path(folder_path)
    if not folder.exists():
        return jsonify({"error": f"文件夹不存在: {folder_path}"})

    if not folder.is_dir():
        return jsonify({"error": f"不是文件夹: {folder_path}"})

    # 复制图片到输入目录
    input_dir = CONFIG["project_root"] / CONFIG["input_dir"]
    input_dir.mkdir(parents=True, exist_ok=True)

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    copied_count = 0

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            dst = input_dir / file_path.name
            if not dst.exists():
                shutil.copy2(file_path, dst)
                copied_count += 1

    # 重新扫描图片
    scan_images()

    return jsonify({
        "success": True,
        "message": f"导入成功: {copied_count} 张图片",
        "total": len(images),
    })


def main():
    print("=" * 60)
    print("瓦片切分工具")
    print("=" * 60)
    print(f"输入目录: {CONFIG['input_dir']}")
    print(f"输出目录: {CONFIG['output_dir']}")
    print(f"启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

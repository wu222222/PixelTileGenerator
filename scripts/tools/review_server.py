"""
Web审核工具

功能:
- 显示所有资源的缩略图
- 左键选择/取消选择
- 右键放大查看原图
- 分页显示
- 批量操作
- 提交后移动文件到good目录
"""

import os
import json
import shutil
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image
from io import BytesIO

# 配置
CONFIG = {
    "project_root": Path(__file__).parent.parent,
    "raw_dir": "datasets/raw/opengameart",
    "output_dir": "datasets/reviewed",
    "thumbnail_size": (128, 128),
    "port": 5000,
}

app = Flask(__name__)

# 全局数据存储
resources = []
submitted_ids = set()
submitted_file = None


def load_submitted():
    """加载已提交记录"""
    global submitted_ids, submitted_file
    submitted_file = CONFIG["project_root"] / CONFIG["output_dir"] / "submitted.json"
    if submitted_file.exists():
        with open(submitted_file, "r") as f:
            data = json.load(f)
            submitted_ids = set(data.get("submitted", []))
    else:
        submitted_ids = set()


def save_submitted():
    """保存已提交记录"""
    global submitted_ids, submitted_file
    submitted_file.parent.mkdir(parents=True, exist_ok=True)
    with open(submitted_file, "w") as f:
        json.dump({"submitted": list(submitted_ids)}, f)


def find_preview_image(resource_dir: Path, prefix: str) -> Path:
    """查找资源的预览图"""
    # 查找preview文件
    for f in resource_dir.parent.glob(f"{prefix}*preview*"):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return f

    # 如果没有preview，查找第一个图片文件
    for f in resource_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return f

    return None


def scan_resources():
    """扫描所有资源"""
    global resources
    resources = []

    raw_dir = CONFIG["project_root"] / CONFIG["raw_dir"]
    if not raw_dir.exists():
        print(f"[错误] 目录不存在: {raw_dir}")
        return

    # 加载已提交记录
    load_submitted()

    # 图片扩展名
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

    # 记录已处理的资源（避免重复）
    processed_resources = set()

    # 扫描所有子目录
    for category_dir in raw_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        # 1. 先找到所有解压后的文件夹
        for item in category_dir.iterdir():
            if item.is_dir():
                # 这是一个解压后的资源文件夹
                # 查找里面的图片
                images = []
                for f in item.rglob("*"):
                    if f.is_file() and f.suffix.lower() in image_extensions:
                        images.append(f)

                if images:
                    # 生成资源名称
                    resource_name = item.name

                    # 查找preview图
                    prefix = resource_name.rsplit("_", 1)[0] if "_" in resource_name else resource_name
                    preview_path = find_preview_image(item, prefix)

                    # 如果找不到preview，用第一张图片作为preview
                    if not preview_path:
                        preview_path = images[0]

                    resource_id = len(resources)
                    resources.append({
                        "id": resource_id,
                        "name": resource_name,
                        "category": category,
                        "preview_path": str(preview_path),
                        "resource_dir": str(item),
                        "image_count": len(images),
                        "submitted": resource_id in submitted_ids,
                    })

                    # 标记已处理（用hash前缀）
                    # 例如 "20x20 Tileset_fe20ed49_0" -> "20x20 Tileset_fe20ed49"
                    processed_resources.add(prefix)

        # 2. 找到没有对应文件夹的单独图片文件
        # 收集所有preview文件和对应的原始文件
        preview_files = list(category_dir.glob("*_preview.*"))

        for preview_path in preview_files:
            if preview_path.suffix.lower() not in image_extensions:
                continue

            # 检查是否已经有对应的文件夹
            stem = preview_path.stem.replace("_preview", "")
            # 提取hash前缀，例如 "20x20 Tileset_fe20ed49" -> "20x20 Tileset_fe20ed49"
            parts = stem.rsplit("_", 1)
            if len(parts) > 1:
                prefix = parts[0]
            else:
                prefix = stem

            # 如果已经有对应的文件夹，跳过
            if prefix in processed_resources:
                continue

            # 查找对应的原始文件
            original_files = []
            for f in category_dir.iterdir():
                if f == preview_path:
                    continue
                if f.is_file() and f.suffix.lower() in image_extensions:
                    # 检查文件名是否匹配
                    if f.stem.startswith(stem) and "_preview" not in f.stem:
                        original_files.append(f)

            if original_files:
                resource_id = len(resources)
                resources.append({
                    "id": resource_id,
                    "name": stem,
                    "category": category,
                    "preview_path": str(preview_path),
                    "resource_dir": str(category_dir),  # 指向父目录
                    "image_count": len(original_files),
                    "submitted": resource_id in submitted_ids,
                    "is_single_files": True,  # 标记为单独文件
                })

    print(f"扫描完成: 找到 {len(resources)} 个资源 (已提交: {len(submitted_ids)})")


def generate_thumbnail(img_path, size=(128, 128)):
    """生成缩略图"""
    try:
        img = Image.open(img_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)

        # 转换为RGB（如果是RGBA）
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except Exception as e:
        print(f"生成缩略图失败 {img_path}: {e}")
        return None


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>瓦片审核工具</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-primary {
            background: #4CAF50;
            color: white;
        }
        .btn-secondary {
            background: #666;
            color: white;
        }
        select, input {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #555;
            background: #333;
            color: white;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .item {
            position: relative;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .item:hover {
            transform: scale(1.05);
        }
        .item.selected {
            outline: 3px solid #4CAF50;
            outline-offset: -3px;
        }
        .item img {
            width: 100%;
            height: 128px;
            object-fit: contain;
            background: #333;
        }
        .item-info {
            padding: 8px;
            font-size: 11px;
            text-align: center;
        }
        .item-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-weight: bold;
        }
        .item-meta {
            color: #888;
            font-size: 10px;
            margin-top: 4px;
        }
        .item-checkbox {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 24px;
            height: 24px;
            background: rgba(0,0,0,0.5);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }
        .item.selected .item-checkbox {
            background: #4CAF50;
        }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 5px;
            margin-top: 20px;
        }
        .pagination button {
            min-width: 40px;
        }
        .pagination button.active {
            background: #4CAF50;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            max-width: 90%;
            max-height: 90%;
            position: relative;
        }
        .modal-content img {
            max-width: 100%;
            max-height: 90vh;
            object-fit: contain;
        }
        .modal-close {
            position: fixed;
            top: 20px;
            right: 20px;
            font-size: 30px;
            cursor: pointer;
            color: white;
            background: rgba(0,0,0,0.5);
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }
        .modal-info {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
        }
        .stats {
            font-size: 14px;
            color: #aaa;
        }
        .selected-count {
            color: #4CAF50;
            font-weight: bold;
        }
        .item.submitted {
            opacity: 0.5;
        }
        .item.submitted::after {
            content: '已提交';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .help-text {
            font-size: 12px;
            color: #888;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>瓦片审核工具</h1>
        <div class="stats">
            总数: <span id="total">{{ total }}</span> |
            已提交: <span id="submitted-count">{{ submitted_count }}</span> |
            已选: <span id="selected" class="selected-count">0</span>
            <span class="help-text">左键选择 | 右键放大</span>
        </div>
        <div class="controls">
            <label>每页:
                <select id="per-page" onchange="changePerPage()">
                    <option value="20">20</option>
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                    <option value="200">200</option>
                </select>
            </label>
            <button class="btn-secondary" onclick="selectAll()">全选</button>
            <button class="btn-secondary" onclick="deselectAll()">取消全选</button>
            <button class="btn-primary" onclick="submitSelected()">提交已选 (移入good)</button>
            <label>
                <input type="checkbox" id="hide-submitted" onchange="toggleHideSubmitted()"> 隐藏已提交
            </label>
        </div>
    </div>

    <div class="grid" id="grid">
        {% for item in items %}
        <div class="item {% if item.submitted %}submitted{% endif %}"
             data-id="{{ item.id }}"
             data-submitted="{{ item.submitted|lower }}"
             oncontextmenu="event.preventDefault(); showLarge({{ item.id }})"
             onclick="toggleSelect(this, {{ item.id }})">
            <div class="item-checkbox">✓</div>
            <img src="/thumbnail/{{ item.id }}" alt="{{ item.name }}" draggable="false">
            <div class="item-info">
                <div class="item-name" title="{{ item.name }}">{{ item.name }}</div>
                <div class="item-meta">{{ item.category }} | {{ item.image_count }} 张图片</div>
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="pagination">
        {% if page > 1 %}
        <button onclick="goToPage({{ page - 1 }})">上一页</button>
        {% endif %}

        {% for p in page_range %}
            {% if p == page %}
            <button class="active">{{ p }}</button>
            {% else %}
            <button onclick="goToPage({{ p }})">{{ p }}</button>
            {% endif %}
        {% endfor %}

        {% if page < total_pages %}
        <button onclick="goToPage({{ page + 1 }})">下一页</button>
        {% endif %}
    </div>

    <div class="modal" id="modal" onclick="closeModal()">
        <div class="modal-close" onclick="closeModal()">×</div>
        <div class="modal-content">
            <img id="modal-img" src="" alt="">
        </div>
        <div class="modal-info" id="modal-info"></div>
    </div>

    <script>
        let selected = new Set();
        let currentModalId = null;

        function toggleSelect(element, id) {
            // 跳过已提交的资源
            if (element.dataset.submitted === 'true') {
                return;
            }

            if (selected.has(id)) {
                selected.delete(id);
                element.classList.remove('selected');
            } else {
                selected.add(id);
                element.classList.add('selected');
            }
            document.getElementById('selected').textContent = selected.size;
        }

        function selectAll() {
            document.querySelectorAll('.item').forEach(item => {
                if (item.dataset.submitted === 'true') {
                    return;
                }
                const id = parseInt(item.dataset.id);
                selected.add(id);
                item.classList.add('selected');
            });
            document.getElementById('selected').textContent = selected.size;
        }

        function deselectAll() {
            selected.clear();
            document.querySelectorAll('.item').forEach(item => {
                item.classList.remove('selected');
            });
            document.getElementById('selected').textContent = 0;
        }

        function showLarge(id) {
            currentModalId = id;
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalInfo = document.getElementById('modal-info');
            modalImg.src = '/preview/' + id;
            modalInfo.textContent = '加载中...';

            // 获取资源信息
            fetch('/info/' + id)
                .then(response => response.json())
                .then(data => {
                    modalInfo.textContent = data.name + ' | ' + data.category + ' | ' + data.image_count + ' 张图片';
                });

            modal.classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
            currentModalId = null;
        }

        function goToPage(page) {
            const hideSubmitted = document.getElementById('hide-submitted').checked;
            window.location.href = '/?page=' + page + '&per_page=' + document.getElementById('per-page').value + '&hide_submitted=' + hideSubmitted;
        }

        function changePerPage() {
            const hideSubmitted = document.getElementById('hide-submitted').checked;
            window.location.href = '/?page=1&per_page=' + document.getElementById('per-page').value + '&hide_submitted=' + hideSubmitted;
        }

        function toggleHideSubmitted() {
            const hideSubmitted = document.getElementById('hide-submitted').checked;
            const items = document.querySelectorAll('.item');
            items.forEach(item => {
                if (item.dataset.submitted === 'true') {
                    item.style.display = hideSubmitted ? 'none' : '';
                }
            });
        }

        function submitSelected() {
            if (selected.size === 0) {
                alert('请先选择要保留的图片');
                return;
            }

            if (!confirm('确定要将选中的 ' + selected.size + ' 个资源移入good目录吗？')) {
                return;
            }

            fetch('/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ids: Array.from(selected)})
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                if (data.success) {
                    location.reload();
                }
            });
        }

        // ESC关闭模态框
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });

        // 禁用右键菜单
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)

    # 计算分页
    total = len(resources)
    submitted_count = sum(1 for r in resources if r["submitted"])
    total_pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    items = resources[start:end]

    # 页码范围
    page_range = range(max(1, page - 5), min(total_pages + 1, page + 6))

    return render_template_string(
        HTML_TEMPLATE,
        items=items,
        page=page,
        per_page=per_page,
        total=total,
        submitted_count=submitted_count,
        total_pages=total_pages,
        page_range=page_range,
    )


@app.route('/thumbnail/<int:resource_id>')
def thumbnail(resource_id):
    """返回缩略图"""
    if resource_id >= len(resources):
        return "Not found", 404

    resource = resources[resource_id]
    preview_path = resource["preview_path"]

    # 生成缩略图
    img = generate_thumbnail(preview_path, CONFIG["thumbnail_size"])
    if img is None:
        return "Error", 500

    # 转换为BytesIO
    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/preview/<int:resource_id>')
def preview(resource_id):
    """返回预览图(大图)"""
    if resource_id >= len(resources):
        return "Not found", 404

    resource = resources[resource_id]
    preview_path = resource["preview_path"]

    return send_file(preview_path)


@app.route('/info/<int:resource_id>')
def info(resource_id):
    """返回资源信息"""
    if resource_id >= len(resources):
        return jsonify({"error": "Not found"}), 404

    resource = resources[resource_id]
    return jsonify({
        "name": resource["name"],
        "category": resource["category"],
        "image_count": resource["image_count"],
    })


@app.route('/submit', methods=['POST'])
def submit():
    """提交选中的资源"""
    global submitted_ids

    data = request.json
    ids = data.get('ids', [])

    if not ids:
        return jsonify({"success": False, "message": "没有选中任何资源"})

    # 创建输出目录
    output_dir = CONFIG["project_root"] / CONFIG["output_dir"] / "good"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    moved_count = 0

    for resource_id in ids:
        if resource_id >= len(resources):
            continue

        resource = resources[resource_id]
        resource_dir = Path(resource["resource_dir"])

        # 判断是文件夹还是单独文件
        if resource.get("is_single_files"):
            # 单独文件：复制preview和对应的原始文件
            preview_path = Path(resource["preview_path"])
            stem = preview_path.stem.replace("_preview", "")

            # 创建资源子目录
            dest_dir = output_dir / resource["name"]
            dest_dir.mkdir(parents=True, exist_ok=True)

            # 复制所有相关文件
            for f in resource_dir.iterdir():
                if f.is_file() and f.suffix.lower() in image_extensions:
                    if f.stem.startswith(stem) and "_preview" not in f.stem:
                        shutil.copy2(f, dest_dir / f.name)

            moved_count += 1
        else:
            # 文件夹：复制整个文件夹
            if resource_dir.exists():
                dest = output_dir / resource["name"]
                if dest.exists():
                    dest = output_dir / f"{resource['name']}_{resource_id}"

                shutil.copytree(resource_dir, dest)
                moved_count += 1

        # 标记为已提交
        submitted_ids.add(resource_id)
        resource["submitted"] = True

    # 保存已提交记录
    save_submitted()

    return jsonify({
        "success": True,
        "message": f"成功移动 {moved_count} 个资源到 good 目录"
    })


def main():
    print("=" * 60)
    print("瓦片审核工具")
    print("=" * 60)
    print(f"扫描资源目录...")
    scan_resources()
    print(f"启动服务器: http://localhost:{CONFIG['port']}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=CONFIG['port'], debug=False)


if __name__ == '__main__':
    main()

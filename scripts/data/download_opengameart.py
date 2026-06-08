"""
OpenGameArt 批量下载脚本

功能:
- 搜索指定关键词的瓦片资源
- 只下载CC0许可的资源
- 只下载zip压缩包
- 自动解压并过滤非图片文件
- 串行下载+随机延迟避免被封
- 按关键词分目录存储
- 支持断点续传
"""

import os
import re
import json
import time
import random
import hashlib
import zipfile
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Optional


# 配置
CONFIG = {
    # 搜索关键词
    "keywords": ["tileset", "terrain", "ground"],

    # 只下载CC0许可
    "license_filter": ["CC0"],

    # 项目根目录
    "project_root": Path(__file__).parent.parent,

    # 下载目录
    "download_dir": "datasets/raw/opengameart",

    # 请求头
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    },

    # 延迟范围(秒)
    "delay_min": 2,
    "delay_max": 5,

    # 最大重试次数
    "max_retries": 3,

    # 文件大小限制(MB)
    "max_file_size_mb": 50,

    # 图片扩展名
    "image_extensions": {".png", ".jpg", ".jpeg", ".gif", ".bmp"},
}


class OpenGameArtDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.base_url = "https://opengameart.org"
        self.session = requests.Session()
        self.session.headers.update(config["headers"])

        # 下载目录
        self.download_dir = config["project_root"] / config["download_dir"]
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # 记录文件(用于断点续传)
        self.records_file = self.download_dir / "download_records.json"
        self.records = self.load_records()

    def load_records(self) -> dict:
        """加载下载记录"""
        if self.records_file.exists():
            with open(self.records_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"downloaded": [], "failed": []}

    def save_records(self):
        """保存下载记录"""
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def delay(self):
        """随机延迟"""
        delay = random.uniform(self.config["delay_min"], self.config["delay_max"])
        time.sleep(delay)

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """获取页面"""
        for attempt in range(self.config["max_retries"]):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.text, "html.parser")
            except Exception as e:
                print(f"  [错误] 获取页面失败 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(5)
        return None

    def search_resources(self, keyword: str, page: int = 0) -> list:
        """搜索资源，返回资源链接列表"""
        url = f"{self.base_url}/art-search-advanced?keys={keyword}&page={page}"
        print(f"搜索: {keyword} (第 {page} 页)")

        soup = self.fetch_page(url)
        if not soup:
            return []

        # 提取资源链接
        links = []
        # 搜索结果中的资源链接
        for item in soup.select("div.views-row a[href*='/content/']"):
            href = item.get("href")
            if href:
                full_url = urljoin(self.base_url, href)
                if full_url not in links:
                    links.append(full_url)

        return links

    def get_total_pages(self, keyword: str) -> int:
        """获取搜索结果总页数"""
        url = f"{self.base_url}/art-search-advanced?keys={keyword}&page=0"
        soup = self.fetch_page(url)
        if not soup:
            return 1

        # 查找分页器
        pager = soup.select("li.pager-item a")
        if not pager:
            return 1

        max_page = 1
        for link in pager:
            href = link.get("href", "")
            match = re.search(r"page=(\d+)", href)
            if match:
                page_num = int(match.group(1)) + 1
                max_page = max(max_page, page_num)

        return max_page

    def parse_file_size(self, size_text: str) -> float:
        """解析文件大小，返回MB"""
        if not size_text:
            return 0

        match = re.search(r'(\d+\.?\d*)\s*(KB|MB|Kb|Mb|GB|Gb)', size_text)
        if not match:
            return 0

        value = float(match.group(1))
        unit = match.group(2).upper()

        if unit == "KB" or unit == "KB":
            return value / 1024
        elif unit == "MB" or unit == "MB":
            return value
        elif unit == "GB" or unit == "GB":
            return value * 1024
        return 0

    def parse_resource_page(self, url: str) -> Optional[dict]:
        """解析资源详情页"""
        soup = self.fetch_page(url)
        if not soup:
            return None

        result = {
            "url": url,
            "title": "",
            "license": "",
            "preview_images": [],
            "download_links": [],
        }

        # 标题 - OpenGameArt使用h2标签
        # 找到第一个非"User login"的h2
        for h2 in soup.find_all("h2"):
            text = h2.get_text(strip=True)
            if text and text not in ["User login", "Comments", "Navigation"]:
                result["title"] = text
                break

        # 备用: 从title标签提取
        if not result["title"]:
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text(strip=True).replace(" | OpenGameArt.org", "")

        # 许可证信息 - 查找"License(s)::"字段
        for field in soup.find_all("div", class_="field"):
            label = field.find("div", class_="field-label")
            if label and "license" in label.get_text(strip=True).lower():
                items = field.find("div", class_="field-items")
                if items:
                    result["license"] = items.get_text(strip=True)
                    break

        # 预览图 - 查找截图区域
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if not src:
                continue

            # 只要opengameart.org的图片
            if "opengameart.org" not in src and not src.startswith("/"):
                continue

            # 过滤掉小图标、logo、许可证图标
            skip_patterns = ["license_images", "sara-logo", "icon", "badge", "file/icons"]
            if any(pattern in src.lower() for pattern in skip_patterns):
                continue

            # 过滤掉太小的图片(宽度<50的通常是图标)
            width = img.get("width", "")
            if width and width.isdigit() and int(width) < 50:
                continue

            full_url = urljoin(self.base_url, src)
            if full_url not in result["preview_images"]:
                result["preview_images"].append(full_url)

        # 下载链接 - 只查找zip文件
        file_field = soup.find("div", class_=re.compile(r"field-name-field-art-files"))
        if not file_field:
            file_field = soup

        for link in file_field.find_all("a", href=True):
            href = link["href"]
            # 只匹配zip文件
            if re.search(r'\.zip$', href, re.IGNORECASE):
                full_url = urljoin(self.base_url, href)
                # 获取文件大小信息
                file_size = ""
                parent_span = link.find_parent("span", class_="file")
                if parent_span:
                    size_text = parent_span.get_text()
                    size_match = re.search(r'(\d+\.?\d*)\s*(KB|MB|Kb|Mb|GB|Gb)', size_text)
                    if size_match:
                        file_size = f"{size_match.group(1)} {size_match.group(2)}"

                result["download_links"].append({
                    "url": full_url,
                    "filename": link.get_text(strip=True),
                    "size": file_size,
                })

        return result

    def is_cc0_license(self, license_text: str) -> bool:
        """检查是否是CC0许可"""
        if not license_text:
            return False
        license_upper = license_text.upper()
        return "CC0" in license_upper or "CC 0" in license_upper or "PUBLIC DOMAIN" in license_upper

    def download_file(self, url: str, save_path: Path) -> bool:
        """下载文件"""
        if save_path.exists():
            print(f"  [跳过] 文件已存在: {save_path.name}")
            return True

        for attempt in range(self.config["max_retries"]):
            try:
                response = self.session.get(url, timeout=120, stream=True)
                response.raise_for_status()

                # 确保目录存在
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # 写入文件
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"  [完成] 下载: {save_path.name}")
                return True

            except Exception as e:
                print(f"  [错误] 下载失败 (尝试 {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(5)

        return False

    def extract_zip(self, zip_path: Path, extract_dir: Path) -> list:
        """解压zip文件，返回图片文件列表"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # 过滤出图片文件
            image_files = []
            for f in extract_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in self.config["image_extensions"]:
                    image_files.append(f)

            print(f"  [解压] 完成: {len(image_files)} 个图片文件")
            return image_files

        except Exception as e:
            print(f"  [错误] 解压失败: {e}")
            return []

    def has_zip_files(self, resource_url: str, keyword: str) -> bool:
        """检查已下载的资源是否有zip文件"""
        # 生成和process_resource中相同的目录名
        url_hash = hashlib.md5(resource_url.encode()).hexdigest()[:8]
        save_dir = self.download_dir / keyword

        # 查找可能的zip文件或解压目录
        for item in save_dir.glob(f"*{url_hash}*"):
            if item.suffix == ".zip":
                return True
            if item.is_dir() and url_hash in item.name:
                # 检查目录中是否有图片文件
                images = list(item.glob("*.png")) + list(item.glob("*.jpg"))
                if images:
                    return True
        return False

    def process_resource(self, resource_url: str, keyword: str):
        """处理单个资源"""
        print(f"\n处理资源: {resource_url}")

        # 检查是否已处理
        if resource_url in self.records["downloaded"]:
            # 检查是否有zip文件
            if self.has_zip_files(resource_url, keyword):
                print("  [跳过] 已下载(有zip文件)")
                return
            else:
                # 没有zip文件，从记录中移除，重新下载
                print("  [重新下载] 之前下载的不是zip文件")
                self.records["downloaded"].remove(resource_url)
                self.save_records()

        # 解析资源页面
        resource = self.parse_resource_page(resource_url)
        if not resource:
            print("  [错误] 无法解析资源页面")
            self.records["failed"].append(resource_url)
            self.save_records()
            return

        # 检查许可证
        if not self.is_cc0_license(resource["license"]):
            print(f"  [跳过] 非CC0许可: {resource['license'][:50]}...")
            return

        # 检查是否有zip下载链接
        if not resource["download_links"]:
            print("  [跳过] 没有zip下载链接")
            return

        print(f"  标题: {resource['title']}")
        print(f"  许可证: {resource['license'][:50]}...")
        print(f"  可下载zip文件: {len(resource['download_links'])} 个")

        # 创建保存目录
        save_dir = self.download_dir / keyword
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名前缀(用URL的hash)
        url_hash = hashlib.md5(resource_url.encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^\w\s-]', '', resource['title'])[:30].strip()
        prefix = f"{safe_title}_{url_hash}"

        # 下载zip资源包
        downloaded_count = 0
        for i, dl_info in enumerate(resource["download_links"][:2]):  # 最多下载2个zip文件
            dl_url = dl_info["url"]
            filename = dl_info["filename"]
            file_size_text = dl_info["size"]

            # 检查文件大小
            file_size_mb = self.parse_file_size(file_size_text)
            if file_size_mb > 0 and file_size_mb > self.config["max_file_size_mb"]:
                print(f"  [跳过] 文件太大: {filename} ({file_size_text})")
                continue

            print(f"  下载: {filename} ({file_size_text})")

            # 保存路径
            save_path = save_dir / f"{prefix}_{i}.zip"

            if self.download_file(dl_url, save_path):
                downloaded_count += 1

                # 解压zip文件
                extract_dir = save_dir / f"{prefix}_{i}"
                image_files = self.extract_zip(save_path, extract_dir)

                # 删除非图片文件
                for f in extract_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() not in self.config["image_extensions"]:
                        f.unlink()

            self.delay()

        # 下载预览图(如果不存在)
        if resource["preview_images"]:
            preview_url = resource["preview_images"][0]
            ext = Path(urlparse(preview_url).path).suffix.lower()
            if ext not in self.config["image_extensions"]:
                ext = ".png"
            preview_path = save_dir / f"{prefix}_preview{ext}"
            if not preview_path.exists():
                self.download_file(preview_url, preview_path)
                self.delay()

        if downloaded_count > 0:
            self.records["downloaded"].append(resource_url)
            print(f"  [成功] 下载并解压了 {downloaded_count} 个资源包")
        else:
            self.records["failed"].append(resource_url)
            print("  [警告] 没有下载到任何文件")

        self.save_records()

    def run(self):
        """主运行函数"""
        print("=" * 60)
        print("OpenGameArt 批量下载脚本 (仅ZIP)")
        print("=" * 60)
        print(f"下载目录: {self.download_dir}")
        print(f"搜索关键词: {', '.join(self.config['keywords'])}")
        print(f"许可证过滤: {', '.join(self.config['license_filter'])}")
        print(f"文件大小限制: {self.config['max_file_size_mb']}MB")
        print("=" * 60)

        for keyword in self.config["keywords"]:
            print(f"\n{'=' * 60}")
            print(f"搜索关键词: {keyword}")
            print(f"{'=' * 60}")

            # 获取总页数
            total_pages = self.get_total_pages(keyword)
            print(f"总页数: {total_pages}")

            # 遍历所有页面
            for page in range(total_pages):
                print(f"\n--- 第 {page + 1}/{total_pages} 页 ---")

                # 搜索资源
                resource_links = self.search_resources(keyword, page)
                print(f"找到 {len(resource_links)} 个资源")

                # 处理每个资源
                for link in resource_links:
                    self.process_resource(link, keyword)
                    self.delay()

        # 打印统计
        print("\n" + "=" * 60)
        print("下载完成!")
        print(f"成功: {len(self.records['downloaded'])}")
        print(f"失败: {len(self.records['failed'])}")
        print("=" * 60)


def main():
    downloader = OpenGameArtDownloader(CONFIG)
    downloader.run()


if __name__ == "__main__":
    main()

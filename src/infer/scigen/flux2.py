import os
import json
import time
import requests
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 假设 prompts.py 在同级目录下，如果报错请检查路径
try:
    from prompts import PROMPT_TEMPLATE
except ImportError:
    # 如果没有 prompts 文件，使用默认模板作为兜底
    PROMPT_TEMPLATE = "Please generate an image based on the following description: {question}"

# ================= 配置区域 =================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_KEY = os.getenv("BFL_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set BFL_API_KEY environment variable")

# BFL API 地址
SUBMIT_URL = "https://api.bfl.ai/v1/flux-2-flex"

# 输入输出配置（相对于项目根目录）
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "scigen", "flux2")

# 并发配置
MAX_WORKERS = 15  # 控制并发提交的任务数

# 生成参数配置
GEN_CONFIG = {
    # "width": 1024,          # 根据需求调整，科学图表通常 1024x1024 或 1280x720
    # "height": 1024,
    "safety_tolerance": 2,  # 允许的某些内容容忍度
    # "output_format": "jpeg" # 或者 "png"，BFL flex 可能默认输出 jpg
}
# ===========================================

# 构造请求头
HEADERS = {
    'accept': 'application/json',
    'x-key': API_KEY,
    'Content-Type': 'application/json'
}

def download_image(url, save_path):
    """辅助函数：下载生成的图片"""
    try:
        img_resp = requests.get(url, stream=True, timeout=60)
        if img_resp.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in img_resp.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except Exception:
        return False

def process_item(item):
    """
    单个任务的处理函数
    流程: 提交任务 -> 获取 polling_url -> 轮询状态 -> 下载图片
    """
    # 1. ID 获取逻辑
    if "id" in item:
        item_id = str(item["id"])
    elif "original_id" in item:
        item_id = str(item["original_id"])
    else:
        item_id = str(hash(item.get("question", "")))
        
    question = item.get("question", "")
    # 注意：Flux 生成的通常是 jpg/jpeg，如果 API 返回 png 可以改后缀
    save_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    # [断点续传]
    if os.path.exists(save_path):
        return 'skipped', item_id, "Already exists"

    # 构造 Prompt
    prompt_text = PROMPT_TEMPLATE.format(question=question)

    # 构造 Payload
    payload = {
        "prompt": prompt_text,
        # "width": GEN_CONFIG["width"],
        # "height": GEN_CONFIG["height"],
        # "safety_tolerance": GEN_CONFIG["safety_tolerance"]
    }

    try:
        # ==========================================
        # 第一步：提交任务 (POST)
        # ==========================================
        response = requests.post(SUBMIT_URL, headers=HEADERS, json=payload, timeout=30)
        
        if response.status_code != 200:
            return 'failed', item_id, f"Submit Failed HTTP {response.status_code}: {response.text[:100]}"

        submit_data = response.json()
        polling_url = submit_data.get("polling_url")
        task_id = submit_data.get("id")

        if not polling_url:
            return 'failed', item_id, f"No polling_url in response: {submit_data}"

        # ==========================================
        # 第二步：轮询状态 (Polling)
        # ==========================================
        status = "Pending"
        max_retries = 150  # 最大轮询次数（从60增加到150，总计5分钟）
        retry_interval = 2 # 每次间隔秒数
        poll_count = 0
        content_moderated_count = 0  # 记录 Content Moderated 出现次数
        max_content_moderated_retries = 10  # Content Moderated 最多等待 20 秒
        
        for _ in range(max_retries):
            time.sleep(retry_interval)
            poll_count += 1
            
            # 查询任务状态
            poll_resp = requests.get(polling_url, headers=HEADERS, timeout=30)
            if poll_resp.status_code != 200:
                continue # 网络抖动则重试，不立即报错
            
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            
            # 每30秒打印一次状态（帮助调试）
            if poll_count % 15 == 0:
                tqdm.write(f"[Debug] {item_id}: status={status}, polled {poll_count*retry_interval}s")

            if status == "Ready":
                # ==========================================
                # 第三步：获取结果并下载
                # ==========================================
                # BFL 成功后的结构通常包含 result -> sample (图片链接)
                result = poll_data.get("result", {})
                sample_url = result.get("sample")
                
                if sample_url:
                    if download_image(sample_url, save_path):
                        return 'success', item_id, ""
                    else:
                        return 'failed', item_id, "Download image failed"
                else:
                    return 'failed', item_id, f"No sample URL in result: {result}"
            
            elif status == "Failed":
                return 'failed', item_id, f"Task Failed on Server: {poll_data}"
            
            elif status == "Request Moderated":
                return 'failed', item_id, "Safety Filter Triggered (Request Moderated)"
            
            elif status == "Content Moderated":
                # Content Moderated 通常表示内容正在审核或被审核系统阻止
                # 如果连续出现多次，说明审核系统永久阻止了，不要继续等待
                content_moderated_count += 1
                if content_moderated_count >= max_content_moderated_retries:
                    return 'failed', item_id, f"Content Moderation Failed (stuck after {content_moderated_count*retry_interval}s)"
            else:
                # 重置计数器，如果状态不是 Content Moderated
                content_moderated_count = 0
            
            # 如果是 'Pending' 或 'Processing'，继续循环
        
        # 超时后返回最后的状态信息
        return 'failed', item_id, f"Polling Timed Out (last status: {status}, {poll_count*retry_interval}s)"

    except Exception as e:
        return 'failed', item_id, f"Exception: {str(e)}"


def generate_images_concurrently():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 2. 读取数据
    print(f"正在读取数据: {INPUT_FILE} ...")
    data_items = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        # 根据 source 字段过滤数据（scigen 目录只处理 source=scigen）
        data_items = [item for item in all_data if item.get('source') == 'scigen']
        print(f"共加载 {len(data_items)} 条 scigen 任务 (从 {len(all_data)} 条总数据中过滤)")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 简单切片测试用 (如果需要测试前5条，取消注释下一行)
    # data_items = data_items[:5]

    print(f"共加载 {len(data_items)} 条任务，开始并发生成 (Workers={MAX_WORKERS})...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    # 3. 线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_items}
        
        pbar = tqdm(as_completed(future_to_item), total=len(data_items), desc="BFL Flux Processing")
        
        for future in pbar:
            status, item_id, msg = future.result()
            
            if status == 'success':
                success_count += 1
            elif status == 'skipped':
                skip_count += 1
            else:
                fail_count += 1
                tqdm.write(f"[Error] ID {item_id}: {msg}")
            
            pbar.set_postfix({"Ok": success_count, "Skip": skip_count, "Fail": fail_count})

    print(f"\n任务结束。")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"图片保存在: {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_images_concurrently()
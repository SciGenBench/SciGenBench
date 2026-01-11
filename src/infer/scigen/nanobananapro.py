import os
import json
import base64
import time
import requests
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import PROMPT_TEMPLATE

# ================= 配置区域 =================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
BASE_HOST = 'http://35.220.164.252:3888' 

# 输入输出配置（相对于项目根目录）
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "scigen", "nanobanana-pro")

# 模型配置
MODEL_NAME = "gemini-3-pro-image-preview"

# 并发配置
MAX_WORKERS = 5  # 建议根据 API 限流情况调整，设置为 5-20 之间比较安全
# ===========================================

# 构造请求 URL (全局变量，方便线程调用)
HOST_CLEAN = BASE_HOST.rstrip('/')
if HOST_CLEAN.endswith('/v1'): 
    HOST_CLEAN = HOST_CLEAN[:-3]
API_URL = f"{HOST_CLEAN}/v1beta/models/{MODEL_NAME}:generateContent"

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def process_item(item):
    """
    单个任务的处理函数
    返回: (status_code, item_id, message)
    status_code: 'success', 'skipped', 'failed'
    """
    # ID 获取逻辑：优先使用 id（新数据格式），兼容 original_id（旧数据格式）
    if "id" in item:
        item_id = str(item["id"])
    elif "original_id" in item:
        item_id = str(item["original_id"])
    else:
        item_id = "unknown"
        
    question = item.get("question", "")
    save_path = os.path.join(OUTPUT_DIR, f"{item_id}.png")

    # [断点续传]
    if os.path.exists(save_path):
        return 'skipped', item_id, "Already exists"

    # 构造 Prompt
    prompt_text = PROMPT_TEMPLATE.format(question=question)

    # 构造 Payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        }
    }

    try:
        # 发送请求
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=120)
        
        if response.status_code != 200:
            return 'failed', item_id, f"HTTP {response.status_code}: {response.text[:100]}"

        response_json = response.json()
        
        # 解析响应
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                return 'failed', item_id, f"No candidates (Safety?): {str(response_json)[:100]}"
            
            parts = candidates[0].get("content", {}).get("parts", [])
            image_data_b64 = None
            # print(parts)
            for part in parts:
                if "inlineData" in part:
                    image_data_b64 = part["inlineData"]["data"]
                    break
            
            if image_data_b64:
                image_bytes = base64.b64decode(image_data_b64)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                return 'success', item_id, ""
            else:
                return 'failed', item_id, "No image data found in response"

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return 'failed', item_id, f"Parse error: {str(e)}"

    except Exception as e:
        return 'failed', item_id, f"Request exception: {str(e)}"


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

    print(f"共加载 {len(data_items)} 条任务，开始并发生成 (Workers={MAX_WORKERS})...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    # 3. 线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_item, item): item for item in data_items}
        
        # 使用 tqdm 监控完成的任务
        pbar = tqdm(as_completed(future_to_item), total=len(data_items), desc="Processing")
        
        for future in pbar:
            status, item_id, msg = future.result()
            
            if status == 'success':
                success_count += 1
            elif status == 'skipped':
                skip_count += 1
            else:
                fail_count += 1
                # 使用 tqdm.write 避免打乱进度条
                tqdm.write(f"[Error] ID {item_id}: {msg}")
            
            # 更新进度条后缀信息
            pbar.set_postfix({"Ok": success_count, "Skip": skip_count, "Fail": fail_count})

    print(f"\n任务结束。")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"图片保存在: {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_images_concurrently()
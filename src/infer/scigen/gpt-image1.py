import os
import json
import base64
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError

# === Import Prompts ===
# 确保 prompts.py 文件在同级目录下
from prompts import PROMPT_TEMPLATE

# ================= 配置区域 =================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
# 注意：OpenAI SDK 通常会自动处理 /v1 后缀，但如果报错 404，请检查是否需要显式加上 /v1
BASE_HOST = 'http://35.220.164.252:3888/v1' 

# 输入输出配置（相对于项目根目录）
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "scigen", "gpt-image1")

# 模型配置
MODEL_NAME = "gpt-image-1"

# 并发配置
MAX_WORKERS = 10
# ===========================================

# 初始化 OpenAI 客户端 (全局变量，线程安全)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_HOST
)

def process_item(item):
    """
    单个任务的处理函数
    返回: (status_code, item_id, message)
    """
    # 1. 获取 ID
    if "original_id" in item:
        item_id = str(item["original_id"])
    else:
        item_id = str(item.get("id", "unknown"))
        
    question = item.get("question", "")
    save_path = os.path.join(OUTPUT_DIR, f"{item_id}.png")

    # [断点续传] 如果图片已存在，直接跳过
    if os.path.exists(save_path):
        return 'skipped', item_id, "Already exists"

    # 2. 构造 Prompt
    prompt_text = PROMPT_TEMPLATE.format(question=question)

    try:
        # 3. 调用 OpenAI SDK 生成图片
        response = client.images.generate(
            model=MODEL_NAME,
            prompt=prompt_text,
            n=1
        )

        # 4. 解析响应
        # OpenAI SDK 返回的对象结构: response.data[0].b64_json
        if response.data and len(response.data) > 0:
            b64_data = response.data[0].b64_json
            
            if b64_data:
                # 解码并保存
                image_bytes = base64.b64decode(b64_data)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                return 'success', item_id, ""
            else:
                return 'failed', item_id, "Response contained empty b64_json"
        else:
            return 'failed', item_id, "No data in response"

    except APIError as e:
        # 捕获 OpenAI API层面的错误 (如 400, 401, 500 等)
        return 'failed', item_id, f"OpenAI API Error: {e}"
    except Exception as e:
        # 捕获其他 Python 错误
        return 'failed', item_id, f"Exception: {str(e)}"


def generate_images_concurrently():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 2. 读取数据
    print(f"正在读取数据: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        # 根据 source 字段过滤数据（scigen 目录只处理 source=scigen）
        data_items = [item for item in all_data if item.get('source') == 'scigen']
        print(f"共加载 {len(data_items)} 条 scigen 任务 (从 {len(all_data)} 条总数据中过滤)")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"开始并发生成 (Workers={MAX_WORKERS})...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    # 3. 线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_items}
        
        pbar = tqdm(as_completed(future_to_item), total=len(data_items), desc="Processing")
        
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
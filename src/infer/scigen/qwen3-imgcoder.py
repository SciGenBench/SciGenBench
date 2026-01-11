import os
import json
import re
import threading
import traceback
import matplotlib
# 必须在导入 pyplot 之前设置后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= 导入 Prompt =================
try:
    from prompts import IMGCODER_PYTHON_PROMPT 
except ImportError:
    # 简单的 fallback，防止找不到文件报错
    IMGCODER_PYTHON_PROMPT = "Please write python code to draw: {question}"
    print("⚠️ Warning: Could not import IMGCODER_PYTHON_PROMPT from prompts.py")

# ================= 配置区域 =================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
BASE_HOST = 'http://35.220.164.252:3888'
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct" 

# 路径配置（相对于项目根目录）
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "scigen.json")
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "scigen", "qwen3-imgcoder")

# 自动生成子目录
IMAGE_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
CODE_DIR = os.path.join(BASE_OUTPUT_DIR, "codes")

# 字体目录 (解决中文/特殊符号乱码，可选)
# 如果环境变量未设置，则使用默认路径或跳过字体加载
FONT_DIR = os.getenv("FONT_DIR", "") 

# 并发配置
MAX_WORKERS = 100      # API 请求并发数
PLOT_LOCK = threading.Lock() # 绘图互斥锁

# ================= 初始化 =================

# 1. 客户端初始化
clean_host = BASE_HOST.rstrip('/')
if not clean_host.endswith('/v1'):
    clean_host += '/v1'

client = OpenAI(api_key=API_KEY, base_url=clean_host)

# 2. 字体初始化
def init_all_fonts():
    """遍历 FONT_DIR 下的所有字体文件并注册"""
    if not os.path.exists(FONT_DIR):
        # print(f"⚠️ 警告: 字体目录不存在: {FONT_DIR}")
        return

    loaded_cnt = 0
    for filename in os.listdir(FONT_DIR):
        if filename.lower().endswith(('.ttf', '.ttc', '.otf')):
            file_path = os.path.join(FONT_DIR, filename)
            try:
                fm.fontManager.addfont(file_path)
                loaded_cnt += 1
            except Exception:
                pass 

    if loaded_cnt > 0:
        print(f"✅ 已加载 {loaded_cnt} 个自定义字体。")
        # 设置全局默认字体栈
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans', 'Microsoft YaHei']
        plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
        plt.rcParams['axes.unicode_minus'] = False 

# ================= 核心工具函数 =================

def extract_python_code(text):
    """提取 Markdown 代码块"""
    if not text: return None
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return None

def save_code_to_file(entry_id, code_content):
    """保存代码到文件"""
    if not code_content: return
    file_path = os.path.join(CODE_DIR, f"{entry_id}.py")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
    except Exception as e:
        print(f"⚠️ Code Save Failed {entry_id}: {e}")

# ================= 核心流程函数 =================

def call_llm_for_code(question):
    """Step 1: 调用 API 生成代码"""
    prompt_text = IMGCODER_PYTHON_PROMPT.format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
        )
        
        full_text = response.choices[0].message.content
        code = extract_python_code(full_text)
        
        if not code:
            preview = full_text[:200].replace('\n', '\\n') if full_text else "Empty"
            return None, f"No code block found. (Model said: {preview}...)"
            
        return code, None

    except Exception as e:
        return None, f"API Error: {str(e)}"

def render_code(code_str, output_path):
    """Step 2: 执行渲染 (需在锁内调用)"""
    # 1. 环境重置
    plt.close('all') 
    # 每次渲染都重置字体设置，防止被上一个脚本污染
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 代码预处理：替换 plt.show() 为 savefig
    save_cmd = f"plt.savefig(r'{output_path}', dpi=100, bbox_inches='tight')"
    
    # 移除 close，防止提前关闭
    code_str = re.sub(r'plt\.close\(\)', 'pass', code_str)

    if "plt.show()" in code_str:
        modified_code = code_str.replace("plt.show()", save_cmd)
    else:
        modified_code = code_str + "\n" + save_cmd

    # 3. 定义上下文
    exec_globals = {
        "plt": plt,
        "np": np,
        "patches": patches,
        "__name__": "__main__", # 关键：触发 if __name__ == "__main__"
        "matplotlib": matplotlib
    }

    # 4. 执行
    exec(modified_code, exec_globals)
    
    # 5. 检查文件
    if not os.path.exists(output_path):
        raise RuntimeError("Code executed but image file was not created.")

def process_item(item):
    """单任务全流程"""
    # ID 获取
    if "id" in item:
        item_id = str(item["id"])
    elif "original_id" in item:
        item_id = str(item["original_id"])
    else:
        item_id = "unknown"

    image_path = os.path.join(IMAGE_DIR, f"{item_id}.png")
    
    # 跳过已存在
    if os.path.exists(image_path):
        return 'skipped', item_id, "Image exists"

    question = item.get("question", "")
    if not question:
        return 'failed', item_id, "Empty question"

    # --- Phase 1: API (并发) ---
    code, err = call_llm_for_code(question)
    if not code:
        return 'failed', item_id, f"LLM Fail: {err}"

    # 保存代码 (无锁)
    save_code_to_file(item_id, code)

    # --- Phase 2: Render (串行) ---
    with PLOT_LOCK:
        try:
            render_code(code, image_path)
            return 'success', item_id, ""
        except Exception as e:
            # traceback.print_exc() # 调试时打开
            return 'failed', item_id, f"Render Error: {str(e)}"
        finally:
            plt.close('all')

# ================= 主程序 =================

def main():
    # 目录创建
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CODE_DIR, exist_ok=True)
    
    # 字体加载
    init_all_fonts()

    print(f"Loading: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"Tasks: {len(data_items)} | Workers: {MAX_WORKERS}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print("-" * 50)

    success_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    
    # 结果记录文件 (JSONL)
    result_log_path = os.path.join(BASE_OUTPUT_DIR, "results.jsonl")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_items}
        
        pbar = tqdm(as_completed(future_to_item), total=len(data_items), desc="Processing")
        
        # 打开日志文件，实时写入
        with open(result_log_path, 'a', encoding='utf-8') as f_log:
            for future in pbar:
                status, item_id, msg = future.result()
                
                if status == 'success':
                    success_cnt += 1
                    # 记录成功日志
                    log_entry = {"id": item_id, "status": "success"}
                    f_log.write(json.dumps(log_entry) + "\n")
                    f_log.flush()

                elif status == 'skipped':
                    skip_cnt += 1
                else:
                    fail_cnt += 1
                    # 记录失败日志
                    log_entry = {"id": item_id, "status": "failed", "error": msg}
                    f_log.write(json.dumps(log_entry) + "\n")
                    f_log.flush()
                    
                    tqdm.write(f"[Fail {item_id}] {msg.splitlines()[0]}")

                pbar.set_postfix({"Ok": success_cnt, "Skip": skip_cnt, "Fail": fail_cnt})

    print("-" * 50)
    print(f"完成。成功: {success_cnt}, 失败: {fail_cnt}, 跳过: {skip_cnt}")

if __name__ == "__main__":
    main()
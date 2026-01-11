import os
import json
import base64
import io
import time
import random
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from openai import OpenAI
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Optional

# === API Config ===
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
BASE_HOST = 'https://api.boyuerichdata.opensphereai.com/v1'
# MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME = "gpt-4.1-nano"
MODEL_NAME = "gemini-3-flash-preview"

# === Concurrency Config ===
MAX_WORKERS = 50  # 根据你的 API 限流情况调整
csv_write_lock = threading.Lock()

# === Prompts ===
QUESTION_EVAL_PROMPT = """
You are a strict expert evaluator of scientific and technical diagrams (e.g., geometry, physics, chemistry).

Evaluate the image against the caption on these 5 dimensions:

### 1. Correctness & Fidelity (0–2)
Core Question: Does the image completely and accurately represent all elements, labels, and spatial/logical relationships from the caption, with no omissions OR hallucinations?
* **2 (High):** Perfect match. All elements (points, lines, shapes, labels) from the caption are present and correct. All specified spatial and logical relationships are perfectly accurate. NO spurious or "hallucinated" elements.
* **1 (Medium):** Mostly correct. Most key elements are present, but with minor omissions, misplacements, or simplifications. Spatial/logical relationships are mostly right but have slight inaccuracies.
* **0 (Low):** Major mismatch. Key elements are missing, incorrect, or relationships are wrong. Or, the image contains significant spurious content.

---

### 2. Layout & Precision (0–2)
Core Question: How clear, balanced, and technically precise is the layout?
* **2 (High):** Professional quality. Layout is clear, balanced, and uncluttered. Lines are precise, connections are exact.
* **1 (Medium):** Generally readable. Layout is understandable but may have slight alignment issues or minor imprecision.
* **0 (Low):** Sloppy or confusing. Layout is cluttered, chaotic, or elements are poorly proportioned. Lines are visibly imprecise.

---

### 3. Readability (Occlusion) (0–2)
Core Question: Do visual elements or labels overlap or occlude each other in a way that obscures meaning or reduces readability?
* **2 (High):** No occlusion. Every element is fully distinct and clearly separated.
* **1 (Medium):** Minor overlap. Some elements or labels slightly touch or overlap, but it only marginally affects readability.
* **0 (Low):** Significant occlusion. Key elements or labels overlap heavily, making parts of the diagram unreadable.

---

### 4. Scientific Plausibility (0–2)
Core Question: Does the image visually conform to the basic principles and conventions of its scientific domain?
* **2 (High):** Visually plausible. The image "looks right" for its domain (e.g., angles, physics vectors, chemical bonds look standard).
* **1 (Medium):** Minor implausibility. Functional but has minor visual flaws (e.g., awkward angles).
* **0 (Low):** Visually implausible. Clearly violates basic scientific/logical principles in its visual representation.

---

### 5. Expressiveness & Richness (0–2)
Core Question: Does the image completely and vividly reproduce the scenario described in the problem?
* **2 (High):** Comprehensive reproduction. Effectively conveys the full context or situation of the problem. Visually rich.
* **1 (Medium):** Basic representation. Depicts necessary elements but lacks contextual richness.
* **0 (Low):** Incomplete scenario. Fails to convey the setting or context, making the "story" hard to understand.

---

### **Output Format**
Provide short reasoning for each dimension, then output a JSON object with integer scores.

**Example Output:**
Reasoning:
* **Correctness & Fidelity:** The image depicts the main triangle and the circumcircle, but the center point 'O' is missing, which was explicitly requested in the caption.
* **Layout & Precision:** The lines are generally clear, but the circle is slightly elliptical (distorted) rather than perfectly round. The connections at vertices A and B are slightly loose.
* **Readability (Occlusion):** The label for angle 'alpha' slightly overlaps with the side AB, making it a bit hard to read, though still decipherable.
* **Scientific Plausibility:** The geometric construction looks generally valid; the triangle fits inside the circle logically.
* **Expressiveness & Richness:** The image is a basic line drawing. It captures the core geometry but lacks the visual emphasis on the "tangent point" described in the problem scenario, appearing somewhat flat.

```json
{{
  "Correctness_Fidelity": 1,
  "Layout_Precision": 1,
  "Readability_Occlusion": 1,
  "Scientific_Plausibility": 2,
  "Expressiveness_Richness": 1
}}
```

Question: {question}

Reason & JSON output:
"""

class ImageQualityEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_HOST)

    def _get_image(self, image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except:
            return None

    def _encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _extract_json(self, text):
            try:
                # 修改正则：匹配 ```json (忽略大小写) 和 ``` 之间的内容
                # \s* 处理 json 关键字后的换行符
                # (.*?) 是捕获组，获取中间的 JSON 字符串
                match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
                
                if match:
                    # group(1) 提取的是反引号里面的内容
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                
                # (可选保底策略) 如果没找到代码块，可以尝试回退到之前的逻辑，
                # 或者直接返回 None 表示必须要有代码块
                # match_fallback = re.search(r'\{.*\}', text, re.DOTALL)
                # if match_fallback:
                #     return json.loads(match_fallback.group(0))
                    
                return None
            except Exception as e:
                # print(f"JSON Extraction Error: {e}") # 调试时可以打开
                return None

    def evaluate_single_image(self, item, model_name, image_base_dir):
        item_id = str(item["id"])
        img_type = item.get("image_type", "Unknown")
        question_text = item.get("question", "")

        row_base = {
            "id": item_id,
            "model": model_name,
            "image_type": img_type,
            "question_snippet": question_text[:50]
        }

        # 尝试多个可能的图片路径（支持 png 和 jpg）
        possible_paths = [
            os.path.join(image_base_dir, model_name, f"{item_id}.png"),  # 直接路径 PNG
            os.path.join(image_base_dir, model_name, f"{item_id}.jpg"),  # 直接路径 JPG
            os.path.join(image_base_dir, model_name, "images", f"{item_id}.png"),  # images子目录 PNG
            os.path.join(image_base_dir, model_name, "images", f"{item_id}.jpg"),  # images子目录 JPG
        ]
        
        image = None
        found_path = None
        for img_path in possible_paths:
            image = self._get_image(img_path)
            if image:
                found_path = img_path
                break

        # 1. 如果本地连图片都没有，重试也没用，直接报错返回
        if not image:
            return {**row_base, "error": f"Image not found (tried {len(possible_paths)} paths)"}

        # 准备 Prompt 和 图片编码 (只做一次)
        base64_img = self._encode_image(image)
        prompt = QUESTION_EVAL_PROMPT.format(question=question_text)

        # 2. 核心修改：无限重试循环
        retry_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                    timeout=60 # 设置超时，防止挂死
                )
                
                content = response.choices[0].message.content
                scores = self._extract_json(content)

                if scores:
                    # 成功！返回结果，跳出循环
                    return {
                        **row_base,
                        "Correctness_Fidelity": scores.get("Correctness_Fidelity", 0),
                        "Layout_Precision": scores.get("Layout_Precision", 0),
                        "Readability_Occlusion": scores.get("Readability_Occlusion", 0),
                        "Scientific_Plausibility": scores.get("Scientific_Plausibility", 0),
                        "Expressiveness_Richness": scores.get("Expressiveness_Richness", 0),
                        "raw_response": content
                    }
                else:
                    # JSON 解析失败，抛出异常触发重试
                    raise ValueError(f"JSON Parsing Failed. Content snippet: {content}...")

            except Exception as e:
                retry_count += 1
                # 打印错误并重试
                print(f"\n[Retry #{retry_count}] ID {item_id} Error: {e}")
                # 随机休眠 1-3 秒，避免请求太频繁被封
                time.sleep(random.uniform(1, 3))
                continue

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["scigen", "seephys", "all"],
        default="all",
        help="Run llm-as-judge on scigen_734 / seephys_451, aligned with quiz.py and quiz_seephys.py",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (if not provided, evaluates all models)",
    )
    return p.parse_args()


def run_dataset(dataset: str, model: Optional[str] = None):
    """
    Keep paths/models consistent with:
      - eval/quiz.py (scigen)
      - eval/quiz_seephys.py (seephys)
    """
    # 获取项目根目录（假设脚本从项目根目录运行）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if dataset == "scigen":
        IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, "images", "scigen")
        RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "scigen", "llm_as_judge")
        all_models = ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
    elif dataset == "seephys":
        IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, "images", "seephys")
        RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "seephys", "llm_as_judge")
        all_models = ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 统一使用 scigenbench.json
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
    
    # 如果指定了模型，只评估该模型；否则评估所有模型
    if model:
        if model not in all_models:
            print(f"Warning: Model '{model}' not in default model list. Will try to evaluate anyway.")
        models = [model]
    else:
        models = all_models

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # 根据 source 字段过滤数据
    benchmark_data = [item for item in all_data if item.get('source') == dataset]
    print(f"Loaded {len(benchmark_data)} items from {dataset} (filtered from {len(all_data)} total items)")

    id_to_type_map = {str(item["id"]): item.get("image_type", "Unknown") for item in benchmark_data}

    evaluator = ImageQualityEvaluator()

    score_cols = [
        "Correctness_Fidelity", 
        "Layout_Precision", 
        "Readability_Occlusion", 
        "Scientific_Plausibility", 
        "Expressiveness_Richness"
    ]
    ALL_COLUMNS = ["id", "model", "image_type", "question_snippet"] + score_cols + ["raw_response", "error"]

    for model_name in models:
        print(f"\n>>> Starting Image Quality Evaluation for: {model_name}")
        output_csv_path = os.path.join(RESULTS_DIR, f"{model_name}_quality_scores.csv")
        
        # === Resume Logic (宽容读取模式) ===
        processed_ids = set()
        if os.path.exists(output_csv_path):
            try:
                df_exist = pd.read_csv(
                    output_csv_path, 
                    dtype={'id': str}, 
                    on_bad_lines='skip', 
                    engine='python'
                )
                
                # 先去重，保留最后一次的结果（防止kill时重复写入）
                df_exist = df_exist.drop_duplicates(subset=['id'], keep='last')
                
                existing_cols = [c for c in score_cols if c in df_exist.columns]
                
                if len(existing_cols) == len(score_cols):
                    for col in score_cols:
                        df_exist[col] = pd.to_numeric(df_exist[col], errors='coerce')
                    
                    valid_rows = df_exist.dropna(subset=score_cols)
                    processed_ids = set(valid_rows["id"].astype(str))
                
                print(f"-> Found {len(processed_ids)} successfully scored images.")
                remaining_count = len(benchmark_data) - len(processed_ids)
                if remaining_count > 0:
                    print(f"-> {remaining_count} images need processing/retrying.")
            except Exception as e:
                print(f"Resume logic warning: {e}")

        items_to_process = [item for item in benchmark_data if str(item["id"]) not in processed_ids]

        if items_to_process:
            print(f"-> Processing {len(items_to_process)} remaining/failed images...")
            
            # 使用多线程，但每个线程内部会无限重试直到成功
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(
                        evaluator.evaluate_single_image,
                        item,
                        model_name,
                        IMAGE_BASE_DIR,
                    ): item 
                    for item in items_to_process
                }

                for future in tqdm(as_completed(future_to_item), total=len(items_to_process)):
                    try:
                        result = future.result()
                        if result:
                            with csv_write_lock:
                                df_row = pd.DataFrame([result])
                                df_row = df_row.reindex(columns=ALL_COLUMNS)
                                is_first = not os.path.exists(output_csv_path)
                                df_row.to_csv(
                                    output_csv_path, 
                                    mode='a', 
                                    header=is_first, 
                                    index=False, 
                                    encoding='utf-8-sig',
                                    quoting=1 # Quote All
                                )
                    except Exception as e:
                        # 这里的 Exception 一般不会触发，因为 evaluate_single_image 内部 catch 了所有异常
                        print(f"Critical Worker Error: {e}")

        # === Statistics Section ===
        if os.path.exists(output_csv_path):
            try:
                df = pd.read_csv(
                    output_csv_path, 
                    dtype={'id': str},
                    on_bad_lines='skip',
                    engine='python'
                )
                
                df = df.drop_duplicates(subset=['id'], keep='last')
                
                for col in score_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                valid_df = df.dropna(subset=score_cols).copy()
                
                if not valid_df.empty:
                    if 'image_type' not in valid_df.columns or valid_df['image_type'].isnull().all():
                        valid_df['image_type'] = valid_df['id'].map(id_to_type_map).fillna('Unknown')

                    print(f"\n=== Quality Scores Summary for {model_name} ===")
                    print(f"Valid Samples: {len(valid_df)} / {len(benchmark_data)}")
                    
                    print("\n[Global Averages (0-2 scale)]")
                    global_means = valid_df[score_cols].mean()
                    for col in score_cols:
                        print(f"{col:<25}: {global_means[col]:.2f}")

                    print("\n[Averages by Image Type]")
                    grouped = valid_df.groupby('image_type')[score_cols].mean()
                    grouped['Count'] = valid_df.groupby('image_type').size()
                    cols_order = ['Count'] + score_cols
                    print(grouped[cols_order].round(2).to_string())
                    
                else:
                    print("No valid score data found.")
                    if 'error' in df.columns:
                        print("\nTop Errors found in CSV:")
                        print(df[df['error'].notna()][['id', 'error']].tail(3).to_string())

            except Exception as e:
                print(f"Stats Error: {e}")

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "all":
        run_dataset("scigen", args.model)
        run_dataset("seephys", args.model)
    else:
        run_dataset(args.dataset, args.model)
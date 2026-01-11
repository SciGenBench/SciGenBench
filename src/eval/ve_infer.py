import os
import json
import base64
import io
import pandas as pd
from tqdm import tqdm
from PIL import Image
# 解除PIL的decompression bomb限制，允许打开大图片
Image.MAX_IMAGE_PIXELS = None
import torch
from openai import OpenAI
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# === Import Prompts ===
# 假设 prompts.py 中有这些变量。如果没有，请使用下文定义的默认值
from prompts import ANSWER_VERIFICATION_COT_PROMPT,process_judgment

VQA_ANSWER_PROMPT = """
You are taking a visual exam. Look at the provided scientific image and answer the following question. Analyze the image carefully and output the answer.
Question:
{quiz_content}
"""

# === API Config ===
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
# BASE_HOST = 'http://35.220.164.252:3888/v1' 
BASE_HOST = 'https://api.boyuerichdata.opensphereai.com/v1'
VQA_MODEL_NAME = "gemini-3-flash-preview"
VERIFICATION_MODEL_NAME = "gpt-4.1-nano"

# === Concurrency Config ===
# 现在的并发直接就是图片数量，因为每张图只问一次
MAX_WORKERS = 50 

# Global Lock for CSV writing
csv_write_lock = threading.Lock()

class SciGenEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Evaluator on {device}...")
        
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_HOST
        )

    def _get_image(self, image_path):
        try:
            img = Image.open(image_path)
            # 验证图片是否有效（检查尺寸）
            img.verify()  # 验证图片完整性
            # verify() 后需要重新打开，因为 verify() 会关闭文件
            img = Image.open(image_path)
            img = img.convert("RGB")
            # 再次检查尺寸是否有效
            if img.size[0] <= 0 or img.size[1] <= 0:
                return None
            return img
        except Exception as e:
            return None

    def _encode_image(self, image):
        """编码图片，如果图片太大则先压缩"""
        # 检查图片是否有效
        if image is None:
            return None
        try:
            # 验证图片尺寸
            if image.size[0] <= 0 or image.size[1] <= 0:
                return None
        except Exception:
            return None
        
        # 检查图片大小，如果太大则压缩
        max_size = 1024  # 最大尺寸（长边）
        max_base64_size = 2 * 1024 * 1024  # 最大Base64大小：2MB
        
        # 确保图片是RGB模式（避免RGBA等格式导致文件过大）
        try:
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
                    image = rgb_image
                else:
                    image = image.convert('RGB')
        except Exception:
            return None
        
        # 如果图片太大，先调整尺寸
        try:
            if max(image.size) > max_size:
                # 计算缩放比例
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                # 再次验证新尺寸
                if new_size[0] <= 0 or new_size[1] <= 0:
                    return None
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        except Exception:
            return None
        
        # 尝试编码，如果还是太大则进一步压缩
        # 使用更激进的压缩策略
        try:
            for quality in [95, 85, 75, 65, 55, 45, 35, 25, 15, 10, 5]:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=quality, optimize=True)
                
                encoded = base64.b64encode(buffered.getvalue())
                if len(encoded) <= max_base64_size:
                    return encoded.decode('utf-8')
        except Exception:
            return None
        
        # 如果JPEG压缩到最低质量还是太大，强制缩小尺寸
        # 逐步缩小直到Base64大小符合要求
        try:
            current_size = max(image.size)
            min_size = 256  # 最小尺寸
            while current_size > min_size:
                current_size = int(current_size * 0.8)  # 缩小20%
                ratio = current_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                # 验证新尺寸
                if new_size[0] <= 0 or new_size[1] <= 0:
                    return None
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=5, optimize=True)
                encoded = base64.b64encode(buffered.getvalue())
                if len(encoded) <= max_base64_size:
                    return encoded.decode('utf-8')
        except Exception:
            return None
        
        # 最后尝试：使用最小尺寸和最低质量
        try:
            image = image.resize((min_size, min_size), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=5, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception:
            return None

    def _call_openai_vision(self, prompt, base64_image, temperature=0.0):
        try:
            # 检查 base64_image 是否有效
            if base64_image is None:
                print(f"Vision API Error: base64_image is None")
                return None
            
            # 打印 base64 大小（用于调试）
            base64_size_mb = len(base64_image) / 1024 / 1024
            if base64_size_mb > 2:
                print(f"Warning: Large base64 image ({base64_size_mb:.2f} MB)")
            
            response = self.client.chat.completions.create(
                model=VQA_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                # temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API Error: {type(e).__name__}: {str(e)}")
            return None

    def _call_openai_text(self, prompt, temperature=0.0):
        try:
            response = self.client.chat.completions.create(
                model=VERIFICATION_MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Text API Error: {type(e).__name__}: {str(e)}")
            return None

    def evaluate_item(self, item, image, model_name):
        """
        核心评测逻辑：
        1. 获取 multimodal_question
        2. 调用视觉模型生成答案
        3. 调用文本模型验证答案 (对比 answer 字段)
        """
        item_id = str(item["id"])
        
        # 标准化返回结构（与 quiz.py 格式一致）
        # 使用 quiz_idx = -2 来标识这是 VQA 结果（-1 已被 quiz.py 用于失败记录）
        result_row = {
            "id": item_id,
            "model": model_name,
            "quiz_idx": -2,  # -2 表示 VQA 结果，与 quiz 区分
            "question_text": "",
            "pred": "",
            "gt": "",
            "is_correct": 0,
            "error_msg": ""
        }

        try:
            # 1. 准备数据
            # 优先使用 multimodal_question，如果没有则使用 question（用于 seephys）
            question_text = item.get('multimodal_question', '') or item.get('question', '')
            ground_truth = item.get('answer', '')
            
            if not question_text:
                result_row['error_msg'] = "Missing 'multimodal_question' or 'question' field"
                return result_row
            
            result_row['question_text'] = question_text
            result_row['gt'] = ground_truth
            
            base64_img = self._encode_image(image)
            
            # 检查图片编码是否成功
            if base64_img is None:
                result_row['error_msg'] = "Image encoding failed"
                print(f"  [Encoding Fail] Image encoding failed for {item_id}")
                return result_row  # 返回错误记录，写入CSV

            # 2. VQA: 让模型回答问题
            # 注意：VQA_ANSWER_PROMPT 应该是一个类似 "Please answer the following question based on the image: {quiz_content}" 的模板
            vqa_input = VQA_ANSWER_PROMPT.format(quiz_content=question_text)
            response = self._call_openai_vision(vqa_input, base64_img)
            
            if not response: 
                result_row['error_msg'] = "Vision API call failed"
                print(f"  [API Fail] Vision API failed for {item_id}")
                return result_row  # 返回错误记录，写入CSV
            
            response = response.strip()
            result_row['pred'] = response

            # 3. Verify: 验证答案一致性
            # 注意：ANSWER_VERIFICATION_PROMPT 需要接收 question, answer (GT), model_response
            verify_input = ANSWER_VERIFICATION_COT_PROMPT.format(
                question=question_text,
                answer=ground_truth,
                model_response=response
            )
            
            verify_resp = self._call_openai_text(verify_input)
            
            if not verify_resp:
                result_row['error_msg'] = "Verification API call failed"
                print(f"  [API Fail] Text API failed for {item_id}")
                return result_row  # 返回错误记录，写入CSV
            
            # 4. 解析验证结果
            judgement = process_judgment(verify_resp)

            result_row['is_correct'] = int(judgement == 'A')
            
        except Exception as e:
            print(f"  [Exception] Error processing {item_id}: {e}")
            result_row['error_msg'] = str(e)
            
        return result_row

    def process_image_entry(self, item, model_name, image_base_dir):
        """
        外层调用接口：加载图片 -> 调用评测
        """
        item_id = str(item["id"])
        
        # 尝试多个可能的图片路径（支持 png 和 jpg，参考 quiz.py）
        possible_paths = [
            os.path.join(image_base_dir, model_name, f"{item_id}.png"),  # 直接路径 PNG
            os.path.join(image_base_dir, model_name, f"{item_id}.jpg"),  # 直接路径 JPG
            os.path.join(image_base_dir, model_name, "images", f"{item_id}.png"),  # images子目录 PNG
            os.path.join(image_base_dir, model_name, "images", f"{item_id}.jpg"),  # images子目录 JPG
        ]
        
        image = None
        for img_path in possible_paths:
            image = self._get_image(img_path)
            if image:
                break
        
        if image:
            return self.evaluate_item(item, image, model_name)
        else:
            return {
                "id": item_id,
                "model": model_name,
                "question_text": "",
                "pred": "",
                "gt": "",
                "is_correct": 0,
                "error_msg": "Image not found"
            }

def run_evaluation(dataset_name, model: str = None):
    """运行单个数据集的评估"""
    # 获取项目根目录（假设脚本从项目根目录运行）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 数据集配置
    # 作为 quiz 的一部分，结果保存到 quiz 目录
    dataset_configs = {
        "scigen": {
            "data_path": os.path.join(PROJECT_ROOT, "data", "scigenbench.json"),
            "image_base_dir": os.path.join(PROJECT_ROOT, "images", "scigen"),
            "results_dir": os.path.join(PROJECT_ROOT, "results", "scigen", "quiz"),  # 保存到 quiz 目录
            "models": ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
        },
        "seephys": {
            "data_path": os.path.join(PROJECT_ROOT, "data", "scigenbench.json"),
            "image_base_dir": os.path.join(PROJECT_ROOT, "images", "seephys"),
            "results_dir": os.path.join(PROJECT_ROOT, "results", "seephys", "quiz"),  # 保存到 quiz 目录
            "models": ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
        }
    }
    
    if dataset_name not in dataset_configs:
        print(f"Error: Unknown dataset '{dataset_name}'. Available: {list(dataset_configs.keys())}")
        return
    
    config = dataset_configs[dataset_name]
    DATA_PATH = config["data_path"]
    IMAGE_BASE_DIR = config["image_base_dir"]
    RESULTS_DIR = config["results_dir"]
    all_models = config["models"]
    
    # 如果指定了模型，只评估该模型；否则评估所有模型
    if model:
        if model not in all_models:
            print(f"Warning: Model '{model}' not in default model list. Will try to evaluate anyway.")
        models = [model]
    else:
        models = all_models
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    print("Loading benchmark data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # 根据 source 字段过滤数据
    benchmark_data = [item for item in all_data if item.get('source') == dataset_name]
    print(f"Loaded {len(benchmark_data)} items from {dataset_name} (filtered from {len(all_data)} total items)")

    # 建立 ID 到 Image Type 的映射，用于最后统计
    id_to_type_map = {str(item["id"]): item.get("image_type", "Unknown") for item in benchmark_data}

    evaluator = SciGenEvaluator()
    print(f"Processing {len(models)} models: {models}")

    for model_name in models:
        print(f"\n=========================================")
        print(f"Starting evaluation for model: {model_name}")
        print(f"=========================================")
        
        # VQA 结果追加到 quiz 的 CSV 文件中（与 quiz.py 使用同一个文件）
        output_csv_path = os.path.join(RESULTS_DIR, f"{model_name}_detailed_evaluation.csv")
        
        # === 断点续跑逻辑 ===
        # 只检查 quiz_idx = -2 的记录（VQA 结果）
        processed_ids = set()
        if os.path.exists(output_csv_path):
            try:
                existing_df = pd.read_csv(output_csv_path)
                if "id" in existing_df.columns and "quiz_idx" in existing_df.columns:
                    # 只统计 quiz_idx = -2 且没有错误的记录为已处理
                    vqa_df = existing_df[existing_df["quiz_idx"] == -2]
                    if "error_msg" in vqa_df.columns:
                        valid_df = vqa_df[vqa_df["error_msg"].isna() | (vqa_df["error_msg"] == "")]
                        processed_ids = set(valid_df["id"].astype(str))
                        error_count = len(vqa_df) - len(valid_df)
                        if error_count > 0:
                            print(f"-> Found {error_count} failed records, will retry them.")
                    else:
                        processed_ids = set(vqa_df["id"].astype(str))
                print(f"-> Resuming... Found {len(processed_ids)} successfully processed items.")
            except Exception as e:
                print(f"-> Warning: Read error. Starting fresh. Error: {e}")

        # 对于 scigen 数据集，只处理有 multimodal_question 的题目
        # 对于 seephys 数据集，使用 question 字段
        is_scigen = dataset_name == "scigen"
        
        items_to_process = []
        skipped_count = 0
        
        for item in benchmark_data:
            item_id = str(item["id"])
            if item_id in processed_ids:
                continue
            
            # scigen 必须有 multimodal_question，seephys 使用 question
            if is_scigen:
                if item.get('multimodal_question', ''):
                    items_to_process.append(item)
                else:
                    skipped_count += 1
            else:  # seephys
                # seephys 使用 question 字段，不需要 multimodal_question
                if item.get('question', ''):
                    items_to_process.append(item)
                else:
                    skipped_count += 1
        
        if skipped_count > 0:
            field_name = "multimodal_question" if is_scigen else "question"
            print(f"-> Skipping {skipped_count} items without '{field_name}'")

        if items_to_process:
            print(f"-> Processing {len(items_to_process)} remaining items...")
            
            # === 并发执行 ===
            # 这里不需要嵌套并发了，直接对 items_to_process 进行并发
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {
                    executor.submit(
                        evaluator.process_image_entry, 
                        item, 
                        model_name, 
                        IMAGE_BASE_DIR
                    ): item 
                    for item in items_to_process
                }

                for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc=f"Eval {model_name}"):
                    try:
                        result = future.result()
                        
                        # 无论成功失败都写入CSV，避免重复尝试
                        if result:
                            with csv_write_lock:
                                df_row = pd.DataFrame([result])
                                is_first_write = not os.path.exists(output_csv_path)
                                
                                df_row.to_csv(
                                    output_csv_path, 
                                    mode='a', 
                                    header=is_first_write, 
                                    index=False, 
                                    encoding='utf-8-sig'
                                )
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"FATAL Error for item {item.get('id')}: {e}")
                        traceback.print_exc()

        # =========================================================
        # Final Scoring Section (只统计 quiz_idx = -2 的结果)
        # =========================================================
        print(f"\nCalculating Final Score for {model_name}...")
        if os.path.exists(output_csv_path):
            try:
                final_df = pd.read_csv(output_csv_path)
                final_df['id'] = final_df['id'].astype(str)
                
                # 只统计 quiz_idx = -2 的记录
                if 'quiz_idx' in final_df.columns:
                    target_df = final_df[final_df['quiz_idx'] == -2].copy()
                else:
                    target_df = pd.DataFrame()
                    print("-> Warning: quiz_idx column not found, cannot filter results.")
                
                # 过滤掉 error_msg 不为空的行，只统计成功的记录
                if not target_df.empty and 'error_msg' in target_df.columns:
                    valid_df = target_df[target_df['error_msg'].isna() | (target_df['error_msg'] == "")].copy()
                    error_count = len(target_df) - len(valid_df)
                    if error_count > 0:
                        print(f"-> Note: {error_count} records have errors and are excluded from statistics.")
                elif not target_df.empty:
                    valid_df = target_df.copy()
                else:
                    valid_df = pd.DataFrame()
                
                if not valid_df.empty:
                    # 确保 is_correct 是数值类型
                    valid_df['is_correct'] = pd.to_numeric(valid_df['is_correct'], errors='coerce').fillna(0)
                    
                    if 'image_type' not in valid_df.columns:
                         valid_df['image_type'] = valid_df['id'].map(id_to_type_map)
                    valid_df['image_type'] = valid_df['image_type'].fillna('Unknown')

                    # 1. Global Metrics
                    total_items = len(valid_df)
                    total_correct = valid_df['is_correct'].sum()
                    overall_acc = total_correct / total_items if total_items > 0 else 0

                    print(f"-"*40)
                    print(f"Model: {model_name}")
                    print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{total_items})")
                    print(f"-"*40)

                    # 2. Detailed Metrics by Image Type
                    stats = valid_df.groupby('image_type')['is_correct'].agg(['count', 'sum', 'mean']).reset_index()
                    stats.rename(columns={'count': 'Total', 'sum': 'Correct', 'mean': 'Accuracy'}, inplace=True)
                    stats['Accuracy'] = stats['Accuracy'] * 100
                    stats = stats.sort_values(by='Accuracy', ascending=False)

                    print(f"{'Image Type':<25} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
                    print(f"-"*60)
                    for _, row in stats.iterrows():
                        print(f"{row['image_type']:<25} | {int(row['Total']):<8} | {int(row['Correct']):<8} | {row['Accuracy']:.2f}%")
                    print(f"-"*60)
                    
                else:
                    print(f"No data found in {output_csv_path}")
            except Exception as e:
                print(f"Error calculating score: {e}")
                traceback.print_exc()
        else:
            print("Output file not found.")

    print(f"\nEvaluation completed for dataset: {dataset_name.upper()}")

def main():
    parser = argparse.ArgumentParser(description='Run VQA evaluation on different datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['seephys', 'scigen', 'all'],
                        help='Dataset to evaluate: seephys, scigen, or all')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to evaluate (if not provided, evaluates all models)')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        # 依次处理 scigen 和 seephys
        datasets = ['scigen', 'seephys']
        for dataset in datasets:
            run_evaluation(dataset, args.model)
        print("\n" + "="*60)
        print("All datasets evaluation completed!")
        print("="*60)
    else:
        run_evaluation(args.dataset, args.model)

if __name__ == "__main__":
    main()
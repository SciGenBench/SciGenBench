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

# === Import Prompts ===
# 假设这些prompts文件在当前目录下，如果报错请确保文件存在
try:
    from prompts import QUESTION_EVAL_PROMPT, VQA_ANSWER_PROMPT, ANSWER_VERIFICATION_PROMPT
except ImportError:
    #以此作为Fallback，防止没有prompts文件时运行报错
    VQA_ANSWER_PROMPT = "Question: {quiz_content}\nAnswer:"
    ANSWER_VERIFICATION_PROMPT = "Question: {question}\nCorrect Answer: {answer}\nModel Answer: {model_response}\nIs the model correct? (JSON with is_correct)"

# === API Config ===
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
BASE_HOST = 'http://35.220.164.252:3888/v1' 
# BASE_HOST = 'https://api.boyuerichdata.opensphereai.com/v1'
MODEL_NAME = "gemini-3-flash-preview"
# MODEL_NAME = "gpt-4.1-nano"

# === Concurrency Config ===
MAX_IMAGE_WORKERS = 20
MAX_QUIZ_WORKERS = 30

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
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
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
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API Error: {e}")
            return None

    def _call_openai_text(self, prompt, temperature=0.0):
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Text API Error: {e}")
            return None

    def process_single_quiz(self, idx, q, base64_image, parent_id):
        """
        处理单个问题。
        如果API调用失败，返回 None，不写入CSV（避免污染数据）。
        """
        result = {
            "quiz_idx": idx,
            "question_text": q.get('question', 'Unknown'),
            "pred": "",
            "gt": q.get('correct_option', ''),
            "is_correct": 0,
            "error_msg": ""
        }
        
        try:
            if isinstance(q['options'], dict):
                options_str = "\n".join([f"{k}: {v}" for k, v in q['options'].items()])
            else:
                options_str = "\n".join(q['options'])
            
            full_question = f"{q['question']}\nOptions:\n{options_str}"
            result["question_text"] = full_question
            
            vqa_input = VQA_ANSWER_PROMPT.format(quiz_content=full_question)
            
            # === API Call 1: Vision (已包含重试机制) ===
            response = self._call_openai_vision(vqa_input, base64_image)
            
            if not response: 
                # API调用失败，返回 None，不写入CSV
                print(f"  [API Fail] Vision API failed for {parent_id} Q{idx}")
                return None
            
            response = response.strip()

            verify_input = ANSWER_VERIFICATION_PROMPT.format(
                question=full_question,
                answer=q['correct_option'],
                model_response=response
            )
            
            # === API Call 2: Text Verification (已包含重试机制) ===
            verify_resp = self._call_openai_text(verify_input)
            
            if not verify_resp:
                # API调用失败，返回 None，不写入CSV
                print(f"  [API Fail] Text API failed for {parent_id} Q{idx}")
                return None

            clean_verify = verify_resp.strip().replace("```json", "").replace("```", "").strip()
            try:
                verify_json = json.loads(clean_verify)
                is_correct = verify_json.get("is_correct", 0)
            except:
                is_correct = 0 

            result["pred"] = response
            result["is_correct"] = is_correct
            
        except Exception as e:
            # 代码逻辑错误，返回 None，不写入CSV
            print(f"  [Exception] Error processing {parent_id} Q{idx}: {e}")
            return None
            
        return result

    def run_quiz_eval_concurrent(self, image, quizzes, parent_id):
        if not image or not quizzes:
            return []

        base64_img = self._encode_image(image)
        # 如果图片编码失败（坏图），返回所有quiz都算错
        if base64_img is None:
            # 坏图：所有quiz都算错
            details = []
            for idx, q in enumerate(quizzes):
                if isinstance(q['options'], dict):
                    options_str = "\n".join([f"{k}: {v}" for k, v in q['options'].items()])
                else:
                    options_str = "\n".join(q['options'])
                full_question = f"{q['question']}\nOptions:\n{options_str}"
                details.append({
                    "quiz_idx": idx,
                    "question_text": full_question,
                    "pred": "",
                    "gt": q.get('correct_option', ''),
                    "is_correct": 0,  # 坏图，所有题目都算错
                    "error_msg": "Corrupted image"
                })
            return details
        
        details = []
        has_failure = False  # 标记是否发生过API失败

        with ThreadPoolExecutor(max_workers=MAX_QUIZ_WORKERS) as executor:
            future_to_quiz = {
                executor.submit(self.process_single_quiz, idx, q, base64_img, parent_id): idx 
                for idx, q in enumerate(quizzes)
            }
            
            for future in as_completed(future_to_quiz):
                try:
                    res = future.result()
                    if res is None:
                        # API失败，标记失败，整个图片不写入CSV
                        has_failure = True
                    else:
                        details.append(res)
                except Exception as e:
                    print(f"Thread exception for {parent_id}: {str(e)}")
                    has_failure = True
        
        # 如果任何quiz失败，返回 None，不写入CSV（避免污染数据）
        if has_failure:
            return None

        details.sort(key=lambda x: x['quiz_idx'])
        return details

    def process_image_entry(self, item, model_name, image_base_dir):
            item_id = str(item["id"])
            
            # 标准字段模板
            base_schema = {
                "id": item_id,
                "model": model_name,
                "quiz_idx": -1,
                "question_text": "",
                "pred": "",
                "gt": "",
                "is_correct": 0,
                "error_msg": ""
            }
            
            # 尝试多个可能的图片路径（支持 png 和 jpg）
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
            
            rows_buffer = []

            if image:
                quiz_details = self.run_quiz_eval_concurrent(image, item["quizzes"], parent_id=item_id)
                
                if quiz_details is None:
                    # API失败，返回空列表，不写入CSV（避免污染数据）
                    return []
                
                if quiz_details:
                    for detail in quiz_details:
                        row = base_schema.copy()
                        row.update(detail)
                        rows_buffer.append(row)
                else:
                    # 图片存在但没有题目列表（数据问题，非API问题），可以记录错误
                    row = base_schema.copy()
                    row["error_msg"] = "No quizzes found in data"
                    rows_buffer.append(row)
            else:
                # 图片不存在（永久性错误），可以记录错误，防止无限重试
                row = base_schema.copy()
                row["error_msg"] = "Image not found"
                rows_buffer.append(row)
                
            return rows_buffer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (if not provided, evaluates all models)",
    )
    args = parser.parse_args()
    
    # === Config Path ===
    # 获取项目根目录（假设脚本从项目根目录运行）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "seephys.json")
    IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, "images", "seephys")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "seephys", "quiz")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("Loading benchmark data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # 建立 ID 到 Image Type 的映射
    id_to_type_map = {str(item["id"]): item.get("image_type", "Unknown") for item in benchmark_data}

    evaluator = SciGenEvaluator()
    all_models = ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
    
    # 如果指定了模型，只评估该模型；否则评估所有模型
    if args.model:
        if args.model not in all_models:
            print(f"Warning: Model '{args.model}' not in default model list. Will try to evaluate anyway.")
        models = [args.model]
    else:
        models = all_models

    for model_name in models:
        print(f"\n=========================================")
        print(f"Starting evaluation for model: {model_name}")
        print(f"=========================================")
        
        output_csv_path = os.path.join(RESULTS_DIR, f"{model_name}_detailed_evaluation.csv")
        
        processed_ids = set()
        if os.path.exists(output_csv_path):
            try:
                # 读取已存在的文件，获取已处理的ID
                existing_df = pd.read_csv(output_csv_path)
                if "id" in existing_df.columns:
                    # 只有成功处理的ID才算处理过（quiz_idx >= 0）
                    # API失败的图片不会写入CSV，下次会重新尝试
                    valid_rows = existing_df[existing_df["quiz_idx"] >= 0]
                    processed_ids = set(valid_rows["id"].astype(str))
                    # 清理失败记录（quiz_idx = -1），避免重复数据
                    failed_rows = existing_df[existing_df["quiz_idx"] == -1]
                    if len(failed_rows) > 0:
                        # 只保留成功处理的记录
                        existing_df = existing_df[existing_df["quiz_idx"] >= 0]
                        existing_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                        print(f"-> Cleaned {len(failed_rows)} failed records (quiz_idx = -1) from CSV.")
                print(f"-> Resuming... Found {len(processed_ids)} SUCCESSFULLY processed images.")
            except Exception as e:
                print(f"-> Warning: Read error or file empty. Starting fresh. Error: {e}")

        items_to_process = [
            item for item in benchmark_data 
            if str(item["id"]) not in processed_ids
        ]

        if items_to_process:
            print(f"-> Processing {len(items_to_process)} remaining images...")
            with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
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
                        rows = future.result()
                        # 只有当 rows 不为空时才写入。
                        # 如果 API 失败，process_image_entry 返回 []，这里就不会写入。
                        if rows:
                            with csv_write_lock:
                                df_chunk = pd.DataFrame(rows)
                                is_first_write = not os.path.exists(output_csv_path)
                                df_chunk.to_csv(
                                    output_csv_path, 
                                    mode='a', 
                                    header=is_first_write, 
                                    index=False, 
                                    encoding='utf-8-sig'
                                )
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"FATAL Error in thread logic for image {item.get('id')}: {e}")
                        traceback.print_exc()

        # =========================================================
        # Final Scoring Section
        # =========================================================
        print(f"\nCalculating Final Score for {model_name}...")
        if os.path.exists(output_csv_path):
            try:
                final_df = pd.read_csv(output_csv_path)
                if final_df.empty:
                    print("CSV is empty.")
                    continue

                final_df['id'] = final_df['id'].astype(str)
                
                # 过滤掉 error_msg 不为空的行（如图片缺失）以及 quiz_idx 为 -1 的行
                valid_df = final_df[(final_df['quiz_idx'] >= 0) & (final_df['error_msg'].isna())].copy()
                
                if not valid_df.empty:
                    if 'image_type' not in valid_df.columns:
                         valid_df['image_type'] = valid_df['id'].map(id_to_type_map)
                    
                    valid_df['image_type'] = valid_df['image_type'].fillna('Unknown')

                    # 1. Global Metrics
                    # Question Level
                    total_questions = len(valid_df)
                    total_correct = valid_df['is_correct'].sum()
                    overall_acc = total_correct / total_questions
                    
                    # Image Level (Perfect Images)
                    image_stats = valid_df.groupby('id')['is_correct'].mean() 
                    total_images = len(image_stats)
                    perfect_images = (image_stats == 1.0).sum()
                    perfect_rate = perfect_images / total_images

                    print(f"-"*40)
                    print(f"Model: {model_name}")
                    print(f"Overall Question Acc : {overall_acc:.2%} ({total_correct}/{total_questions})")
                    print(f"Overall Perfect Img  : {perfect_rate:.2%} ({perfect_images}/{total_images})")
                    print(f"-"*40)

                    # 2. Detailed Metrics by Image Type
                    q_level_stats = valid_df.groupby('image_type')['is_correct'].agg(['count', 'sum', 'mean']).reset_index()
                    q_level_stats.rename(columns={'count': 'Total Qs', 'sum': 'Correct Qs', 'mean': 'Q Acc'}, inplace=True)
                    
                    img_level_agg = valid_df.groupby(['id', 'image_type'])['is_correct'].mean().reset_index()
                    img_level_agg['is_perfect'] = (img_level_agg['is_correct'] == 1.0).astype(int)
                    
                    type_perfect_stats = img_level_agg.groupby('image_type')['is_perfect'].mean().reset_index()
                    type_perfect_stats.rename(columns={'is_perfect': 'Perfect Img Rate'}, inplace=True)

                    final_stats = pd.merge(q_level_stats, type_perfect_stats, on='image_type')

                    final_stats['Q Acc'] = final_stats['Q Acc'] * 100
                    final_stats['Perfect Img Rate'] = final_stats['Perfect Img Rate'] * 100
                    
                    final_stats = final_stats.sort_values(by='Q Acc', ascending=False)

                    print(f"{'Image Type':<25} | {'Total Qs':<8} | {'Corr Qs':<8} | {'Q Acc':<8} | {'Perfect Rate':<12}")
                    print(f"-"*75)
                    for _, row in final_stats.iterrows():
                        print(f"{row['image_type']:<25} | {int(row['Total Qs']):<8} | {int(row['Correct Qs']):<8} | {row['Q Acc']:.2f}%   | {row['Perfect Img Rate']:.2f}%")
                    print(f"-"*75)
                    
                else:
                    print(f"No valid quiz data found in {output_csv_path}")
            except Exception as e:
                print(f"Error calculating score: {e}")
                traceback.print_exc()
        else:
            print("Output file not found.")

    print("\nEvaluation completed.")

if __name__ == "__main__":
    main()
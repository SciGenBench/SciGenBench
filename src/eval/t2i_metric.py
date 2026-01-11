import os
import json
import re
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# === Metrics Libraries ===
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

# --- NEW: token-level truncation for CLIP text (77 tokens) ---
from transformers import CLIPTokenizer

# ============================================================
# Configuration (SeePhys T2I)
# ============================================================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 统一使用 scigenbench.json
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
GEN_BASE_DIR = os.path.join(PROJECT_ROOT, "images", "seephys")
GT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "seephys_images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "seephys", "t2i")

MODELS = ["gemini-3-flash-imgcoder", "gemini-3-pro-imgcoder", "qwen3-imgcoder", "hunyuan", "nanobanana-pro", "nanobanana", "flux2", "qwen-image-plus", "seedream4.0", "gpt-image1", "gpt-image1_5"]
DEBUG_MAX_ITEMS = None

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: CLIP max length is 77 TOKENS (not 77 chars)
CLIP_MAX_TOKENS = 77

PERFECT_IMAGE_TXT = os.path.join(PROJECT_ROOT, "data", "perfect_image_seephys.txt")


def load_perfect_image_ids(perfect_image_txt: str) -> set:
    """
    Load perfect image IDs from text file.
    These IDs will be excluded from T2I metric calculation.
    
    Returns:
        Set of perfect image IDs (as strings), empty set if file doesn't exist
    """
    if not os.path.exists(perfect_image_txt):
        return set()
    
    try:
        with open(perfect_image_txt, 'r', encoding='utf-8') as f:
            ids = {line.strip() for line in f if line.strip()}
        return ids
    except Exception as e:
        print(f"Warning: Failed to load perfect image IDs from {perfect_image_txt}: {e}")
        return set()


def _read_existing_ids(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={"id": str})
        if "id" not in df.columns:
            return set()
        return set(df["id"].astype(str))
    except Exception:
        return set()


def _load_summary_report(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    parts = re.split(r"\n-{20}\n", text.strip())
    out = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"^Model:\s*(.+)$", p, flags=re.MULTILINE)
        if not m:
            continue
        out[m.group(1).strip()] = p
    return out


def _write_summary_report(path: str, model: str, avg_psnr: float, avg_ssim: float, avg_clip: float, fid: float):
    blocks = _load_summary_report(path)
    blocks[model] = "\n".join(
        [
            f"Model: {model}",
            f"PSNR: {avg_psnr:.4f}",
            f"SSIM: {avg_ssim:.4f}",
            f"CLIP: {avg_clip:.4f}",
            f"FID : {fid:.4f}",
        ]
    )
    content = "\n" + ("\n" + "-" * 20 + "\n").join(blocks[m] for m in sorted(blocks.keys())) + "\n" + "-" * 20 + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.lstrip("\n"))


class ImageQualityEvaluator:
    def __init__(self, device: str):
        self.device = device
        print(f"Loading Metrics Models on {device}...")

        # 1) CLIP Score (text-image alignment)
        self.clip_metric = CLIPScore(model_name_or_path=CLIP_MODEL_ID).to(device)

        # Tokenizer for 77-token truncation (your requirement)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)

        # 2) FID (distribution distance)
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)

        # --- Split transforms: FID vs CLIP ---
        # FID uses Inception features -> 299x299 is standard
        self.fid_transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

        # CLIP standard input is 224 (TorchMetrics CLIPScore can handle floats in [0,1])
        self.clip_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def reset_fid(self):
        self.fid_metric.reset()

    def _clip_truncate_to_77_tokens(self, text: str) -> str:
        """
        Enforce your requirement: truncate to 77 CLIP tokens.
        Note: This is token-level, NOT character-level.
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        enc = self.clip_tokenizer(
            text,
            truncation=True,
            max_length=CLIP_MAX_TOKENS,
            return_tensors="pt",
        )
        # decode back to string (skip special tokens)
        return self.clip_tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

    def compute_pixel_metrics(self, gen_img_pil, gt_img_pil):
        """CPU PSNR/SSIM."""
        try:
            if gen_img_pil.size != gt_img_pil.size:
                gt_img_pil = gt_img_pil.resize(gen_img_pil.size, Image.BICUBIC)

            gen_arr = np.array(gen_img_pil)
            gt_arr = np.array(gt_img_pil)

            if gen_arr.shape[-1] == 4:
                gen_arr = gen_arr[..., :3]
            if gt_arr.shape[-1] == 4:
                gt_arr = gt_arr[..., :3]

            psnr_val = psnr_func(gt_arr, gen_arr, data_range=255)

            # NOTE: win_size=3 is allowed but a bit small; keep as-is if intentional
            ssim_val = ssim_func(gt_arr, gen_arr, data_range=255, channel_axis=2, win_size=3)
            return psnr_val, ssim_val
        except Exception:
            return None, None

    def update_gpu_metrics(self, gen_img_pil, gt_img_pil, text_prompt: str):
        """CLIP score + update FID state."""
        try:
            # ---- CLIP ----
            prompt_77tok = self._clip_truncate_to_77_tokens(text_prompt)
            gen_clip = self.clip_transform(gen_img_pil).to(self.device).unsqueeze(0)
            clip_val = self.clip_metric(gen_clip, [prompt_77tok])

            # ---- FID ----
            gen_fid = self.fid_transform(gen_img_pil).to(self.device).unsqueeze(0)
            gt_fid = self.fid_transform(gt_img_pil).to(self.device).unsqueeze(0)

            # convert to uint8 properly
            gen_uint8 = (gen_fid.clamp(0, 1) * 255.0).round().to(torch.uint8)
            gt_uint8 = (gt_fid.clamp(0, 1) * 255.0).round().to(torch.uint8)

            self.fid_metric.update(gt_uint8, real=True)
            self.fid_metric.update(gen_uint8, real=False)

            return float(clip_val.item())
        except Exception as e:
            print(f"GPU metric error: {e}")
            return None

    def compute_final_fid(self):
        try:
            return float(self.fid_metric.compute().item())
        except Exception as e:
            print(f"FID Compute error (maybe not enough images): {e}")
            return -1.0


def process_model(evaluator, model_name, data, id_to_prompt, results_dir, perfect_image_ids=None):
    """
    Process a single model and save results to the specified directory.
    Default behavior: exclude perfect image IDs (hard mode).
    
    Args:
        evaluator: ImageQualityEvaluator instance
        model_name: Model name
        data: List of data items
        id_to_prompt: Dict mapping id to prompt
        results_dir: Output directory
        perfect_image_ids: Set of perfect image IDs to exclude
    """
    os.makedirs(results_dir, exist_ok=True)
    
    output_csv = os.path.join(results_dir, f"{model_name}_t2i_metrics.csv")
    summary_path = os.path.join(results_dir, "summary_report.txt")
    
    valid_items = []
    for item in data:
        item_id = str(item["id"])
        
        # Skip perfect image IDs (hard mode by default)
        if perfect_image_ids and item_id in perfect_image_ids:
            continue
        
        # Try multiple possible paths for generated image (支持 png 和 jpg)
        possible_gen_paths = [
            os.path.join(GEN_BASE_DIR, model_name, f"{item_id}.png"),  # 直接路径 PNG
            os.path.join(GEN_BASE_DIR, model_name, f"{item_id}.jpg"),  # 直接路径 JPG
            os.path.join(GEN_BASE_DIR, model_name, "images", f"{item_id}.png"),  # images子目录 PNG
            os.path.join(GEN_BASE_DIR, model_name, "images", f"{item_id}.jpg"),  # images子目录 JPG
        ]
        gen_path = None
        for path in possible_gen_paths:
            if os.path.exists(path):
                gen_path = path
                break
        
        # GT 图片通常是 PNG（真实图片）
        gt_path = os.path.join(GT_BASE_DIR, f"{item_id}.png")
        if gen_path and os.path.exists(gt_path):
            valid_items.append(
                {"id": item_id, "gen_path": gen_path, "gt_path": gt_path, "prompt": id_to_prompt.get(item_id, "")}
            )
    
    excluded_count = len(perfect_image_ids) if perfect_image_ids else 0
    print(f"Found {len(valid_items)} valid image pairs" + (f" (excluded {excluded_count} perfect image IDs)" if excluded_count > 0 else ""))
    
    if DEBUG_MAX_ITEMS is not None:
        valid_items = valid_items[:DEBUG_MAX_ITEMS]
        print(f"[Debug] Truncated to {len(valid_items)} pairs (DEBUG_MAX_ITEMS={DEBUG_MAX_ITEMS})")
    
    expected_ids = {x["id"] for x in valid_items}
    
    done_ids = _read_existing_ids(output_csv)
    
    # If CSV already complete, skip recompute but still ensure summary exists
    if expected_ids and expected_ids.issubset(done_ids) and os.path.exists(output_csv):
        print(f"[Skip] {output_csv} already covers {len(done_ids & expected_ids)}/{len(expected_ids)} ids. Skipping recompute.")
        df_exist = pd.read_csv(output_csv, dtype={"id": str})
        df_exist = df_exist[df_exist["id"].astype(str).isin(expected_ids)].copy()
        
        evaluator.reset_fid()
        for it in tqdm(valid_items, desc=f"FID update {model_name}"):
            try:
                gen_img = Image.open(it["gen_path"]).convert("RGB")
                gt_img = Image.open(it["gt_path"]).convert("RGB")
                _ = evaluator.update_gpu_metrics(gen_img, gt_img, it["prompt"])
            except Exception:
                continue
        
        fid_score = evaluator.compute_final_fid()
        avg_psnr = pd.to_numeric(df_exist["psnr"], errors="coerce").mean()
        avg_ssim = pd.to_numeric(df_exist["ssim"], errors="coerce").mean()
        avg_clip = pd.to_numeric(df_exist["clip_score"], errors="coerce").mean()
        _write_summary_report(summary_path, model_name, float(avg_psnr), float(avg_ssim), float(avg_clip), float(fid_score))
        return
    
    to_process = [it for it in valid_items if it["id"] not in done_ids]
    if not to_process:
        if not valid_items:
            print(f"[Warning] No valid image pairs found for model {model_name}!")
            print(f"[Info] Check if generated images exist in: {os.path.join(GEN_BASE_DIR, model_name)}")
            print(f"[Info] Check if GT images exist in: {GT_BASE_DIR}")
        else:
            print(f"[Info] Nothing to process (all {len(valid_items)} items already done).")
        return
    
    print(f"[Resume] Processing {len(to_process)} remaining pairs (existing={len(done_ids & expected_ids)}).")
    
    evaluator.reset_fid()
    rows = []
    for it in tqdm(to_process, desc=f"Evaluating {model_name}"):
        try:
            gen_img = Image.open(it["gen_path"]).convert("RGB")
            gt_img = Image.open(it["gt_path"]).convert("RGB")
            psnr, ssim = evaluator.compute_pixel_metrics(gen_img, gt_img)
            clip_score = evaluator.update_gpu_metrics(gen_img, gt_img, it["prompt"])
            rows.append({"id": it["id"], "model": model_name, "psnr": psnr, "ssim": ssim, "clip_score": clip_score})
        except Exception as e:
            print(f"Error processing {it['id']}: {e}")
    
    if rows:
        df_new = pd.DataFrame(rows)
        write_header = not os.path.exists(output_csv)
        df_new.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        print(f"Updated metrics CSV: {output_csv} (+{len(df_new)})")
    
    df_all = pd.read_csv(output_csv, dtype={"id": str})
    df_all = df_all[df_all["id"].astype(str).isin(expected_ids)].copy()
    
    evaluator.reset_fid()
    for it in tqdm(valid_items, desc=f"FID update {model_name}"):
        try:
            gen_img = Image.open(it["gen_path"]).convert("RGB")
            gt_img = Image.open(it["gt_path"]).convert("RGB")
            _ = evaluator.update_gpu_metrics(gen_img, gt_img, it["prompt"])
        except Exception:
            continue
    fid_score = evaluator.compute_final_fid()
    
    avg_psnr = pd.to_numeric(df_all["psnr"], errors="coerce").mean()
    avg_ssim = pd.to_numeric(df_all["ssim"], errors="coerce").mean()
    avg_clip = pd.to_numeric(df_all["clip_score"], errors="coerce").mean()
    
    print(f"\n--- {model_name} Final Report ---")
    print(f"PSNR : {float(avg_psnr):.4f}")
    print(f"SSIM : {float(avg_ssim):.4f}")
    print(f"CLIP : {float(avg_clip):.4f}")
    print(f"FID  : {float(fid_score):.4f}")
    print(f"---------------------------------")
    
    _write_summary_report(summary_path, model_name, float(avg_psnr), float(avg_ssim), float(avg_clip), float(fid_score))


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
    
    # Load perfect image IDs
    perfect_image_ids = load_perfect_image_ids(PERFECT_IMAGE_TXT)
    if perfect_image_ids:
        print(f"Loaded {len(perfect_image_ids)} perfect image IDs from {PERFECT_IMAGE_TXT}")
    else:
        print(f"No perfect image IDs found (or file doesn't exist): {PERFECT_IMAGE_TXT}")
    
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # 根据 source 字段过滤数据（t2i_metric.py 只处理 seephys）
    data = [item for item in all_data if item.get('source') == 'seephys']
    print(f"Loaded {len(data)} items from seephys (filtered from {len(all_data)} total items)")
    
    id_to_prompt = {str(item["id"]): item.get("question", "") for item in data}
    
    evaluator = ImageQualityEvaluator(DEVICE)
    
    # 如果指定了模型，只评估该模型；否则评估所有模型
    if args.model:
        if args.model not in MODELS:
            print(f"Warning: Model '{args.model}' not in default model list. Will try to evaluate anyway.")
        models_to_eval = [args.model]
    else:
        models_to_eval = MODELS
    
    for model_name in models_to_eval:
        print(f"\n{'=' * 40}")
        print(f"Processing Model: {model_name}")
        print(f"{'=' * 40}")
        
        # Process with hard mode (exclude perfect image IDs by default)
        process_model(evaluator, model_name, data, id_to_prompt, RESULTS_DIR, perfect_image_ids=perfect_image_ids)

    print("\nAll evaluations completed.")


if __name__ == "__main__":
    main()

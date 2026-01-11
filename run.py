#!/usr/bin/env python3
"""
SciGenBench Unified Runner
统一启动脚本：支持图像生成和评估
"""

import os
import sys
import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 配置 ====================

# 可用数据集
DATASETS = ["scigen", "seephys"]

# 可用模型（按数据集分类）
AVAILABLE_MODELS = {
    "scigen": [
        "gemini-3-pro-imgcoder",
        "gemini-3-flash-imgcoder",
        "qwen3-imgcoder",
        "gpt-image1",
        "gpt-image1_5",
        "gpt-image1-mini",
        "nanobanana",
        "nanobananapro",
        "qwen-image-plus",
        "hunyuan",
        "flux2",
        "seedream",
    ],
    "seephys": [
        "gemini-3-pro-imgcoder",
        "gemini-3-flash-imgcoder",
        "qwen3-imgcoder",
        "gpt-image1",
        "gpt-image1_5",
        "nanobanana",
        "nanopro",  # seephys 使用 nanopro 而不是 nanobananapro
        "qwen-image-plus",
        "hunyuan",
        "flux2",
        "seedream",
    ]
}

# 评估指标
EVAL_METRICS = {
    "judge": "LLM-as-Judge 评估（5维评分）",
    "quiz": "Inverse Quiz Validation（逆向验证，包含 VQA）",
    "t2i": "Text-to-Image Metrics (PSNR, SSIM, CLIP, FID) - 仅适用于 seephys",
    "all": "所有评估指标",
}

# ==================== 工具函数 ====================

def get_model_script_path(dataset: str, model: str) -> Path:
    """获取模型脚本路径"""
    # 处理模型名称映射
    model_file_map = {
        "nanobananapro": "nanobananapro.py",
        "nanopro": "nanopro.py",
        "seedream": "seedream.py",
    }
    
    model_file = model_file_map.get(model, f"{model}.py")
    script_path = PROJECT_ROOT / "src" / "infer" / dataset / model_file
    
    if not script_path.exists():
        # 尝试其他可能的文件名
        alt_path = PROJECT_ROOT / "src" / "infer" / dataset / f"{model.replace('-', '_')}.py"
        if alt_path.exists():
            return alt_path
        
        # 提供更详细的错误信息
        dataset_dir = PROJECT_ROOT / "src" / "infer" / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # 列出目录中的文件
        existing_files = list(dataset_dir.glob("*.py"))
        existing_names = [f.name for f in existing_files]
        raise FileNotFoundError(
            f"Model script not found: {script_path}\n"
            f"Available files in {dataset_dir}: {', '.join(sorted(existing_names))}"
        )
    
    return script_path

def get_eval_script_path(metric: str, dataset: str) -> Path:
    """获取评估脚本路径"""
    if metric == "judge":
        script_path = PROJECT_ROOT / "src" / "eval" / "llm_as_judge.py"
    elif metric == "quiz":
        if dataset == "scigen":
            script_path = PROJECT_ROOT / "src" / "eval" / "quiz.py"
        else:  # seephys
            script_path = PROJECT_ROOT / "src" / "eval" / "quiz_seephys.py"
    elif metric == "t2i":
        if dataset != "seephys":
            raise ValueError(f"T2I metric is only available for seephys dataset, got {dataset}")
        script_path = PROJECT_ROOT / "src" / "eval" / "t2i_metric.py"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Eval script not found: {script_path}")
    
    return script_path

def run_generation(dataset: str, model: str, verbose: bool = False) -> bool:
    """运行图像生成"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Image Generation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    original_cwd = os.getcwd()
    try:
        script_path = get_model_script_path(dataset, model)
        
        # 切换到脚本所在目录，确保相对导入正常工作
        script_dir = script_path.parent
        
        # 运行脚本
        cmd = [sys.executable, str(script_path.name)]
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {script_dir}\n")
        
        # 不捕获输出，让进度条正常显示
        # tqdm 进度条需要直接输出到终端
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=True,
            # 不捕获输出，让进度条和实时输出正常显示
            stdout=None,
            stderr=None,
        )
        
        if result.returncode == 0:
            print(f"\n✅ Generation completed successfully!")
            return True
        else:
            print(f"\n❌ Generation failed with return code {result.returncode}")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation process failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 恢复工作目录
        try:
            os.chdir(original_cwd)
        except:
            pass

def run_vqa_evaluation(dataset: str, model: str = None, verbose: bool = False) -> bool:
    """运行 VQA 评估（作为 quiz 的一部分）"""
    original_cwd = os.getcwd()
    try:
        script_path = PROJECT_ROOT / "src" / "eval" / "ve_infer.py"
        script_dir = script_path.parent
        
        cmd = [sys.executable, str(script_path.name), "--dataset", dataset]
        if model:
            cmd.extend(["--model", model])
        
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {script_dir}\n")
        
        # 捕获输出以便调试
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=False,  # 不抛出异常，让我们自己处理
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # 打印输出以便调试
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            return True
        else:
            print(f"\n❌ VQA evaluation failed with return code {result.returncode}")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.chdir(original_cwd)
        except:
            pass

def run_evaluation(dataset: str, metric: str, model: str = None, verbose: bool = False) -> bool:
    """运行评估"""
    print(f"\n{'='*60}")
    print(f"📊 Starting Evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Metric: {metric} ({EVAL_METRICS[metric]})")
    print(f"{'='*60}\n")
    
    original_cwd = os.getcwd()
    try:
        script_path = get_eval_script_path(metric, dataset)
        
        # 切换到脚本所在目录
        script_dir = script_path.parent
        
        # 构建命令
        cmd = [sys.executable, str(script_path.name)]
        
        # 对于 judge，需要传递 dataset 参数
        if metric == "judge":
            cmd.extend(["--dataset", dataset])
        
        # 传递模型参数
        if model and metric in ["judge", "quiz", "t2i"]:
            cmd.extend(["--model", model])
        
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {script_dir}\n")
        
        # 不捕获输出，让进度条正常显示
        # 但对于 t2i，我们需要捕获一些输出以便调试
        if metric == "t2i":
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                check=False,  # 不抛出异常，让我们自己处理
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # 打印输出以便调试
            if result.stdout:
                print(result.stdout)
            if result.returncode == 0:
                print(f"\n✅ Evaluation completed successfully!")
                return True
            else:
                print(f"\n❌ Evaluation failed with return code {result.returncode}")
                if result.stdout:
                    # 打印最后几行错误信息
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 10:
                        print("Last 10 lines of output:")
                        for line in lines[-10:]:
                            print(f"  {line}")
                return False
        else:
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                check=True,
                # 不捕获输出，让进度条和实时输出正常显示
                stdout=None,
                stderr=None,
            )
            
            if result.returncode == 0:
                print(f"\n✅ Evaluation completed successfully!")
                return True
            else:
                print(f"\n❌ Evaluation failed with return code {result.returncode}")
                return False
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation process failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 恢复工作目录
        try:
            os.chdir(original_cwd)
        except:
            pass

def summarize_results(dataset: str, model: str, metrics: List[str]) -> None:
    """汇总评估结果"""
    print(f"\n{'='*60}")
    print(f"📈 Results Summary")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    results_dir = PROJECT_ROOT / "results" / dataset
    
    # Judge 结果
    if "judge" in metrics:
        judge_dir = results_dir / "llm_as_judge"
        judge_file = judge_dir / f"{model}_quality_scores.csv"
        
        if judge_file.exists():
            try:
                df = pd.read_csv(judge_file)
                score_cols = [
                    "Correctness_Fidelity",
                    "Layout_Precision",
                    "Readability_Occlusion",
                    "Scientific_Plausibility",
                    "Expressiveness_Richness"
                ]
                
                valid_df = df.dropna(subset=score_cols)
                if not valid_df.empty:
                    print("📊 LLM-as-Judge Scores (0-2 scale):")
                    print("-" * 60)
                    for col in score_cols:
                        mean_score = valid_df[col].mean()
                        print(f"  {col:<30}: {mean_score:.2f}")
                    print(f"  {'Total Samples':<30}: {len(valid_df)}")
                    print("-" * 60)
                else:
                    print("⚠️  No valid judge scores found")
            except Exception as e:
                print(f"⚠️  Error reading judge results: {e}")
        else:
            print("⚠️  Judge results file not found")
    
    # Quiz 结果（合并 quiz 和 vqa 的结果）
    if "quiz" in metrics:
        quiz_dir = results_dir / "quiz"
        quiz_file = quiz_dir / f"{model}_detailed_evaluation.csv"
        vqa_dir = results_dir / "vqa"
        vqa_file = vqa_dir / f"{model}_eval_cot.csv"
        
        quiz_df = None
        vqa_df = None
        
        # 读取 quiz 结果
        if quiz_file.exists():
            try:
                quiz_df = pd.read_csv(quiz_file)
                quiz_df = quiz_df[quiz_df['quiz_idx'] >= 0].copy()
            except Exception as e:
                print(f"⚠️  Error reading quiz results: {e}")
        
        # 读取 vqa 结果（仅 scigen 数据集有 VQA）
        vqa_df = None
        if dataset == "scigen":
            if vqa_file.exists():
                try:
                    vqa_df = pd.read_csv(vqa_file)
                    # 过滤掉有错误的记录
                    if 'error_msg' in vqa_df.columns:
                        vqa_df = vqa_df[vqa_df['error_msg'].isna() | (vqa_df['error_msg'] == "")].copy()
                except Exception as e:
                    print(f"⚠️  Error reading VQA results: {e}")
        
        # 合并 quiz 和 vqa 结果（仅 scigen）
        if dataset == "scigen" and quiz_df is not None and vqa_df is not None and not vqa_df.empty:
            # 合并两个 DataFrame
            # quiz 使用 id, quiz_idx, is_correct
            # vqa 使用 id, is_correct，需要转换为相同格式
            quiz_combined = quiz_df[['id', 'is_correct']].copy()
            vqa_combined = vqa_df[['id', 'is_correct']].copy()
            
            # 合并
            combined_df = pd.concat([quiz_combined, vqa_combined], ignore_index=True)
            
            total_questions = len(combined_df)
            total_correct = combined_df['is_correct'].sum()
            overall_acc = total_correct / total_questions if total_questions > 0 else 0
            
            # Image level perfect rate（基于 id）
            image_stats = combined_df.groupby('id')['is_correct'].mean()
            perfect_images = (image_stats == 1.0).sum()
            perfect_rate = perfect_images / len(image_stats) if len(image_stats) > 0 else 0
            
            print("\n📝 Inverse Quiz Validation Results (Quiz + VQA combined):")
            print("-" * 60)
            print(f"  Quiz Questions    : {len(quiz_combined)}")
            print(f"  VQA Questions     : {len(vqa_combined)}")
            print(f"  Total Questions   : {total_questions}")
            print(f"  Question Accuracy : {overall_acc:.2%} ({total_correct}/{total_questions})")
            print(f"  Perfect Image Rate : {perfect_rate:.2%} ({perfect_images}/{len(image_stats)})")
            print("-" * 60)
        elif quiz_df is not None and not quiz_df.empty:
            # 只有 quiz 结果（seephys 或 scigen 没有 vqa 结果时）
            total_questions = len(quiz_df)
            total_correct = quiz_df['is_correct'].sum()
            overall_acc = total_correct / total_questions if total_questions > 0 else 0
            
            image_stats = quiz_df.groupby('id')['is_correct'].mean()
            perfect_images = (image_stats == 1.0).sum()
            perfect_rate = perfect_images / len(image_stats) if len(image_stats) > 0 else 0
            
            if dataset == "scigen" and vqa_file.exists():
                print("\n📝 Inverse Quiz Validation Results (Quiz only, VQA file exists but no valid data):")
            else:
                print("\n📝 Inverse Quiz Validation Results:")
            print("-" * 60)
            print(f"  Question Accuracy : {overall_acc:.2%} ({total_correct}/{total_questions})")
            print(f"  Perfect Image Rate : {perfect_rate:.2%} ({perfect_images}/{len(image_stats)})")
            print("-" * 60)
        else:
            print("⚠️  No valid quiz or VQA results found")
    
    # T2I 结果（仅 seephys）
    if "t2i" in metrics and dataset == "seephys":
        t2i_dir = results_dir / "t2i"
        t2i_file = t2i_dir / f"{model}_t2i_metrics.csv"
        
        if t2i_file.exists():
            try:
                df = pd.read_csv(t2i_file)
                valid_df = df.dropna(subset=['psnr', 'ssim', 'clip_score'])
                
                if not valid_df.empty:
                    avg_psnr = valid_df['psnr'].mean()
                    avg_ssim = valid_df['ssim'].mean()
                    avg_clip = valid_df['clip_score'].mean()
                    
                    print("\n📊 Text-to-Image Metrics:")
                    print("-" * 60)
                    print(f"  PSNR (avg)        : {avg_psnr:.4f}")
                    print(f"  SSIM (avg)        : {avg_ssim:.4f}")
                    print(f"  CLIP Score (avg) : {avg_clip:.4f}")
                    print(f"  Total Samples    : {len(valid_df)}")
                    print("-" * 60)
                    print("  Note: FID score is computed separately and may be in summary_report.txt")
                else:
                    print("⚠️  No valid T2I metrics found")
            except Exception as e:
                print(f"⚠️  Error reading T2I results: {e}")
        else:
            print("⚠️  T2I results file not found")
    
    print()

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="SciGenBench Unified Runner - Generate images and evaluate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images only
  python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode generate
  
  # Evaluate only
  python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode eval --metric judge
  
  # Generate and evaluate all metrics
  python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode all
  
  # List available models
  python run.py --list-models --dataset scigen
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
        help="Dataset to use (scigen or seephys)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use for generation/evaluation"
    )
    
    parser.add_argument(
        "--mode",
        choices=["generate", "eval", "all"],
        default="all",
        help="Mode: generate (only), eval (only), or all (generate + eval)"
    )
    
    parser.add_argument(
        "--metric",
        choices=list(EVAL_METRICS.keys()),
        default="all",
        help="Evaluation metric to use (only for eval mode)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models for the dataset"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip results summary at the end"
    )
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print(f"\nAvailable models for dataset '{args.dataset}':")
        print("-" * 60)
        for model in AVAILABLE_MODELS[args.dataset]:
            print(f"  - {model}")
        print()
        return
    
    # 验证模型
    if not args.model:
        parser.error("--model is required (use --list-models to see available models)")
    
    if args.model not in AVAILABLE_MODELS[args.dataset]:
        print(f"❌ Error: Model '{args.model}' not available for dataset '{args.dataset}'")
        print(f"\nAvailable models:")
        for model in AVAILABLE_MODELS[args.dataset]:
            print(f"  - {model}")
        sys.exit(1)
    
    # 确定要运行的评估指标
    if args.mode in ["eval", "all"]:
        if args.metric == "all":
            # 根据数据集选择可用的评估指标
            if args.dataset == "seephys":
                metrics_to_run = ["judge", "quiz", "t2i"]
            else:  # scigen
                metrics_to_run = ["judge", "quiz"]
        else:
            metrics_to_run = [args.metric]
            # 检查 t2i 是否用于正确的数据集
            if "t2i" in metrics_to_run and args.dataset != "seephys":
                print(f"❌ Error: T2I metric is only available for seephys dataset")
                sys.exit(1)
    else:
        metrics_to_run = []
    
    # 执行任务
    success = True
    
    # 1. 生成图像
    gen_success = True
    if args.mode in ["generate", "all"]:
        gen_success = run_generation(args.dataset, args.model, args.verbose)
        if not gen_success:
            print("\n⚠️  Generation failed. Continuing with evaluation if requested...")
    
    # 2. 评估（即使生成失败，如果用户指定了 eval 模式，也要运行评估）
    eval_success = True
    if args.mode in ["eval", "all"]:
        for metric in metrics_to_run:
            metric_success = run_evaluation(args.dataset, metric, args.model, args.verbose)
            if not metric_success:
                print(f"\n⚠️  Evaluation '{metric}' failed. Continuing...")
                eval_success = False
            
            # 如果运行了 quiz，且数据集是 scigen，自动运行 VQA（ve_infer.py）
            # VQA 只适用于 scigen 数据集
            if metric == "quiz" and args.dataset == "scigen":
                print(f"\n{'='*60}")
                print(f"📊 Running VQA Evaluation (part of quiz, scigen only)")
                print(f"{'='*60}\n")
                vqa_success = run_vqa_evaluation(args.dataset, args.model, args.verbose)
                if not vqa_success:
                    print(f"\n⚠️  VQA evaluation failed. Continuing...")
                    eval_success = False
    
    # 整体成功状态：生成和评估都要成功
    success = gen_success and eval_success
    
    # 3. 汇总结果
    if not args.skip_summary and metrics_to_run:
        summarize_results(args.dataset, args.model, metrics_to_run)
    
    # 退出状态
    if success:
        print(f"\n✅ All tasks completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tasks failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()


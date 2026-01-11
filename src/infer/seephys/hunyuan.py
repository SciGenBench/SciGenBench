import os
import json
import time
import requests
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.aiart.v20221229 import aiart_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

from prompts import PROMPT_TEMPLATE

# ================= 配置区域 =================
# 获取项目根目录（假设脚本从项目根目录运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 密钥配置 (请确保填入真实的 AKID 和 Key)
SECRET_ID = os.getenv("TENCENT_SECRET_ID", "")
if not SECRET_ID:
    raise ValueError("Please set TENCENT_SECRET_ID environment variable")
SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "")
if not SECRET_KEY:
    raise ValueError("Please set TENCENT_SECRET_KEY environment variable")
REGION = "ap-shanghai"

# 输入输出配置（相对于项目根目录）
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "scigenbench.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images", "seephys", "hunyuan")

# 混元生图必须保持单线程
MAX_WORKERS = 1 
# ===========================================

def get_aiart_client():
    cred = credential.Credential(SECRET_ID, SECRET_KEY)
    httpProfile = HttpProfile()
    httpProfile.endpoint = "aiart.tencentcloudapi.com"
    httpProfile.reqTimeout = 30
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    return aiart_client.AiartClient(cred, REGION, clientProfile)

def download_image(url, save_path, max_retries=3):
    """下载图片，支持重试（URL可能需要延迟才能访问）"""
    if not url or url.strip() == "":
        return False
    
    for attempt in range(max_retries):
        try:
            # 第一次失败后等待一下，URL可能需要时间生效
            if attempt > 0:
                time.sleep(2)
            
            img_resp = requests.get(url, stream=True, timeout=60)
            if img_resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in img_resp.iter_content(1024):
                        f.write(chunk)
                # 验证文件是否成功写入
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    return True
            elif img_resp.status_code == 404:
                # 404说明URL无效，不用重试
                return False
        except Exception as e:
            if attempt == max_retries - 1:
                # 最后一次尝试失败，打印错误
                tqdm.write(f"[DEBUG] Download failed after {max_retries} attempts: {str(e)[:100]}")
    
    return False

def process_item(item):
    if "id" in item:
        item_id = str(item["id"])
    elif "original_id" in item:
        item_id = str(item["original_id"])
    else:
        return 'skipped', "no_id", "No ID found"
        
    question = item.get("question", "")
    if not question:
         return 'skipped', item_id, "No question found"

    save_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")
    
    if os.path.exists(save_path):
        return 'skipped', item_id, "Already exists"

    prompt_text = PROMPT_TEMPLATE.format(question=question)

    client = get_aiart_client()

    try:
        # 1. 提交任务
        req = models.SubmitTextToImageJobRequest()
        req.Prompt = prompt_text
        req.Resolution = "1024:1024"
        req.LogoAdd = 0
        
        job_id = None
        
        # 提交重试循环
        while True:
            try:
                resp = client.SubmitTextToImageJob(req)
                job_id = resp.JobId
                break 
            except TencentCloudSDKException as err:
                err_msg = getattr(err, 'message', str(err))
                if "limit" in str(err) or "上限" in str(err) or "RequestLimitExceeded" in str(err):
                    # 遇到拥堵，静默等待
                    time.sleep(5)
                    continue
                elif "文本可能包含敏感信息" in err_msg or "敏感" in err_msg:
                    # 敏感信息过滤，无法绕过，直接失败
                    return 'failed', item_id, f"内容审核拦截: 敏感信息"
                else:
                    return 'failed', item_id, f"SDK提交失败: {err_msg}"
            except Exception as e:
                return 'failed', item_id, f"提交异常: {str(e)}"
        
        # 2. 轮询等待
        query_req = models.QueryTextToImageJobRequest()
        query_req.JobId = job_id
        
        # 混元排队可能较久，这里最多等 ~5 分钟（150 * 2s）
        network_retry_count = 0
        max_network_retries = 3
        
        for i in range(150): 
            time.sleep(2)
            try:
                query_resp = client.QueryTextToImageJob(query_req)
                # 成功查询后重置网络重试计数
                network_retry_count = 0
                
                # 转为字典来取值
                resp_json = json.loads(query_resp.to_json_string())

                # 新老字段都兼容一下
                status = resp_json.get("JobStatus")
                status_code = str(resp_json.get("JobStatusCode", "")).strip()
                status_msg = str(resp_json.get("JobStatusMsg", "")).strip()
                
                # 第一次打印一下结构方便调试
                if i == 0 and not status and not status_code:
                    tqdm.write(f"[DEBUG] 响应结构: {resp_json}")
                
                # 根据字段推断状态：
                # - 实测返回中：JobStatusCode=5, JobStatusMsg=处理完成 且有 ResultImage 时其实是成功
                # - FAIL / 含“失败”“错误”或有 JobErrorCode 时才视为失败
                err_code = str(resp_json.get("JobErrorCode", "")).strip()
                # 成功条件：显式 SUCCESS，或状态码 5 / 文案含“处理完成”“成功”，并且无错误码
                is_success = (
                    status == "SUCCESS"
                    or status_code in ("4", "5")
                    or ("处理完成" in status_msg or "成功" in status_msg)
                ) and not err_code
                # 失败条件：显式 FAIL，或有错误码，或文案含失败/错误
                is_fail = (
                    status == "FAIL"
                    or bool(err_code)
                    or ("失败" in status_msg or "错误" in status_msg)
                )

                if is_success:
                    # 检查是否有 ImageIllegalDetected（内容审核失败）
                    result_details = resp_json.get("ResultDetails", [])
                    if any("ImageIllegalDetected" in str(d) for d in result_details):
                        return 'failed', item_id, "图片内容违规 (ImageIllegalDetected)"
                    
                    img_field = resp_json.get("ResultImage")
                    # SDK 返回的 ResultImage 可能是字符串或列表，这里统一处理
                    img_url = None
                    if isinstance(img_field, list):
                        # 过滤空字符串
                        valid_urls = [u for u in img_field if u and u.strip()]
                        img_url = valid_urls[0] if valid_urls else None
                    else:
                        img_url = img_field if img_field and str(img_field).strip() else None

                    if img_url and download_image(img_url, save_path):
                        return 'success', item_id, ""
                    else:
                        # 成功状态但没有有效图片，打印完整响应方便排查
                        tqdm.write(f"[DEBUG][{item_id}] SUCCESS but no image, resp: {json.dumps(resp_json, ensure_ascii=False)[:500]}")
                        return 'failed', item_id, "下载失败 (URL无效或为空)"
                
                if is_fail and not is_success:
                    err_msg = resp_json.get("JobErrorMsg", "") or status_msg or "Unknown Error"
                    # 打印更详细的失败信息，包含完整响应（截断）
                    tqdm.write(
                        f"[DEBUG][{item_id}] FAIL status_code={status_code}, "
                        f"status_msg={status_msg}, err_code={err_code}, err_msg={err_msg}, "
                        f"resp={json.dumps(resp_json, ensure_ascii=False)[:500]}"
                    )
                    return 'failed', item_id, f"生成失败: {err_msg}"
                
                # 其他状态（INIT / WAIT / RUN / 运行中）继续循环等待
            
            except TencentCloudSDKException as sdk_err:
                # 网络超时可以重试
                if "Read timed out" in str(sdk_err) or "timeout" in str(sdk_err).lower():
                    network_retry_count += 1
                    if network_retry_count <= max_network_retries:
                        tqdm.write(f"[DEBUG][{item_id}] Network timeout, retry {network_retry_count}/{max_network_retries}")
                        time.sleep(5)  # 等待更长时间
                        continue  # 继续轮询
                    else:
                        return 'failed', item_id, f"轮询超时 (网络超时 {max_network_retries} 次)"
                else:
                    return 'failed', item_id, f"轮询SDK异常: {str(sdk_err)}"
            except Exception as e:
                # 其他异常
                return 'failed', item_id, f"轮询解析异常: {str(e)}"
        
        return 'failed', item_id, "生成超时 (约 300 秒)"

    except Exception as e:
        return 'failed', item_id, f"未知错误: {str(e)}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"读取数据: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        # 根据 source 字段过滤数据（seephys 目录只处理 source=seephys）
        data_items = [item for item in all_data if item.get('source') == 'seephys']
        print(f"共加载 {len(data_items)} 条 seephys 任务 (从 {len(all_data)} 条总数据中过滤)")
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    print(f"开始生成 (单线程排队模式)...")
    
    success_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_items}
        
        pbar = tqdm(as_completed(future_to_item), total=len(data_items))
        
        for future in pbar:
            status, item_id, msg = future.result()
            
            if status == 'success':
                success_count += 1
            elif status == 'skipped':
                skip_count += 1
            else:
                fail_count += 1
                # 只在出错时打印 ID 和 原因，保持界面清爽
                tqdm.write(f"[Fail] {item_id}: {msg}")
            
            pbar.set_postfix({"Ok": success_count, "Fail": fail_count})

if __name__ == "__main__":
    main()
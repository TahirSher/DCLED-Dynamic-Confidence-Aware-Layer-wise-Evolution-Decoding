import torch
import os
import numpy as np
import math
import gc
import time
import json
import logging
import pandas as pd  
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from collections import defaultdict
from config import EPS, LOG_EPS, PROB_CLAMP_MIN, PROB_CLAMP_MAX, LOGIT_CLIP_MAX

def clear_cuda_memory():
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    
def aggressive_memory_cleanup():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    time.sleep(0.1)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dcled_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_device():
    if not torch.cuda.is_available():
        logger.info("[Device] CUDA not available, using CPU")
        return torch.device("cpu")
    for gpu_id in [2, 0, 3, 1]:
        try:
            torch.cuda.set_device(gpu_id)
            test_tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
            del test_tensor
            name = torch.cuda.get_device_name(gpu_id)
            logger.info(f"[GPU] Using GPU {gpu_id}: {name}")
            return torch.device(f"cuda:{gpu_id}")
        except Exception:
            continue
    logger.info("[Device] No suitable GPU found, using CPU")
    return torch.device("cpu")

DEVICE = get_device()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_model_size_category(model_name: str) -> str:
    
    model_name_lower = model_name.lower()
    
    if '1b' in model_name_lower or '1.3b' in model_name_lower:
        return 'small'
    elif '3b' in model_name_lower or '2.7b' in model_name_lower:
        return 'small'
    elif '7b' in model_name_lower or '8b' in model_name_lower:
        return 'medium'
    elif '13b' in model_name_lower or '14b' in model_name_lower:
        return 'large'
    else:
        return 'medium'

def get_model_size(model_name):
    if '1B' in model_name or '1b' in model_name:
        return '1B'
    elif '3B' in model_name or '3b' in model_name:
        return '3B'
    elif '8B' in model_name or '8b' in model_name:
        return '8B'
    return 'Unknown'

def stable_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    
    x = x / max(temperature, 0.01)
    x = torch.clamp(x, min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True).clamp(min=EPS)
    result = exp_x / sum_exp
    return result.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)

def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
   
    x = torch.clamp(x, min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    shifted = x - max_x
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True).clamp(min=EPS))
    return shifted - log_sum_exp

def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
   
    probs = probs.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    if probs.dim() > 0 and probs.numel() > 1:
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=EPS)
    log_probs = torch.log(probs + LOG_EPS)
    return -torch.sum(probs * log_probs, dim=dim).clamp(min=0.0)

def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    
    p = p.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    q = q.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    p = p / p.sum(dim=dim, keepdim=True)
    q = q / q.sum(dim=dim, keepdim=True)
    kl = (p * (torch.log(p + LOG_EPS) - torch.log(q + LOG_EPS))).sum(dim=dim)
    return kl.clamp(min=0.0)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    
    p = p.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    q = q.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    
    kl_pm = (p * (torch.log(p + LOG_EPS) - torch.log(m + LOG_EPS))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + LOG_EPS) - torch.log(m + LOG_EPS))).sum(dim=-1)
    
    return 0.5 * (kl_pm + kl_qm).clamp(min=0.0)

def get_relative_top_filter(
    scores: torch.FloatTensor, 
    relative_top: float = 0.1,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    
    scores_normalized = stable_log_softmax(scores, dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True, dim=-1)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1:min_tokens_to_keep]
    probs_max = torch.max(scores_normalized, dim=-1, keepdim=True).values
    probs_thresh = probs_max + math.log(relative_top + EPS)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    return scores_normalized < probs_thresh

def bootstrap_confidence_interval(
    data: List[float], 
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:

    if not data or len(data) == 0:
        return 0.0, 0.0, 0.0
    
    data = np.array(data)
    n = len(data)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    mean = np.mean(data)
    
    return float(mean), float(ci_lower), float(ci_upper)

def paired_ttest(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    
    if len(scores1) != len(scores2) or len(scores1) == 0:
        return 0.0, 1.0
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    return float(t_stat), float(p_value)

def wilcoxon_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    
    if len(scores1) != len(scores2) or len(scores1) < 3:
        return 0.0, 1.0
    try:
        stat, p_value = stats.wilcoxon(scores1, scores2)
        return float(stat), float(p_value)
    except:
        return 0.0, 1.0

def cohen_d(scores1: List[float], scores2: List[float]) -> float:
    
    if not scores1 or not scores2:
        return 0.0
    
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
    n1, n2 = len(scores1), len(scores2)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((mean1 - mean2) / pooled_std)

def convert_to_serializable(obj):
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def format_best(best_answer: str) -> str:
    return " " + best_answer.strip()

def split_multi_answer(ans: str, sep: str = ';') -> List[str]:
    if not ans or pd.isna(ans):
        return []
    answers = ans.strip().split(sep)
    return [" " + a.strip() for a in answers if a.strip()]

def build_prompt_and_answer(question: str, answer: str) -> Tuple[str, str]:
    prompt = f"Q: {question}\nA:"
    return prompt, answer

def MC_calcs(scores_true: List[float], scores_false: List[float],
             ref_true: List[str], ref_false: List[str], ref_best: str) -> Dict[str, float]:  
    if not scores_true or not scores_false:
        return {
            'MC1': 0.0, 'MC2': 0.0, 'MC3': 0.0,
            'predicted_answer': '', 
            'predicted_is_true': False, 
            'predicted_score': -np.inf
        }    
    max_true = max(scores_true)
    max_false = max(scores_false)
    max_true_idx = scores_true.index(max_true)
    max_false_idx = scores_false.index(max_false)
    mc1 = 1.0 if max_true > max_false else 0.0
    
    if max_true > max_false:
        predicted_answer = ref_true[max_true_idx]
        predicted_is_true = True
        predicted_score = max_true
    else:
        predicted_answer = ref_false[max_false_idx]  
        predicted_is_true = False
        predicted_score = max_false
    
    all_scores = scores_true + scores_false
    max_score = max(all_scores)
    all_scores_exp = [np.exp(s - max_score) for s in all_scores]
    total_exp = sum(all_scores_exp)
    
    if total_exp == 0 or np.isnan(total_exp) or np.isinf(total_exp):
        mc2 = 0.0
    else:
        true_prob_mass = sum(all_scores_exp[:len(scores_true)]) / total_exp
        mc2 = true_prob_mass
    
    all_with_labels = [(s, True, ref_true[i]) for i, s in enumerate(scores_true)] + \
                      [(s, False, ref_false[i]) for i, s in enumerate(scores_false)]
    all_with_labels.sort(key=lambda x: x[0], reverse=True)
    mc3 = 1.0 if all_with_labels[0][1] else 0.0
    
    return {
        'MC1': mc1, 
        'MC2': mc2, 
        'MC3': mc3,
        'predicted_answer': predicted_answer,
        'predicted_is_true': predicted_is_true,
        'predicted_score': predicted_score
    }
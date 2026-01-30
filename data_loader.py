import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datasets import load_dataset as hf_load_dataset
import logging

logger = logging.getLogger(__name__)

def load_truthfulqa_dataset(data_path: str) -> List[Dict]:
    if os.path.isdir(data_path):
        possible_files = [
            os.path.join(data_path, "TruthfulQA.csv"),
            os.path.join(data_path, "truthfulqa.csv"),
            os.path.join(data_path, "TruthfulQA", "TruthfulQA.csv"),
        ]
        filepath = None
        for pf in possible_files:
            if os.path.exists(pf):
                filepath = pf
                break
        if filepath is None:
            logger.error(f"Could not find TruthfulQA.csv in {data_path}")
            return []
    else:
        filepath = data_path
    
    logger.info(f"[TruthfulQA] Loading from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"[TruthfulQA] Failed to load: {e}")
        return []
    
    dataset = []
    for idx, row in df.iterrows():
        try:
            sample = {
                'question': row['Question'],
                'answer_best': row.get('Best Answer', ''),
                'answer_true': row.get('Correct Answers', ''),
                'answer_false': row.get('Incorrect Answers', '')
            }
            
            if not sample['answer_best'] or pd.isna(sample['answer_best']):
                true_answers = split_multi_answer(sample['answer_true'])
                if true_answers:
                    sample['answer_best'] = true_answers[0].strip()
            
            if sample['answer_true'] and sample['answer_false']:
                dataset.append(sample)
        except Exception:
            continue
    
    logger.info(f"[TruthfulQA] Loaded {len(dataset)} valid samples")
    return dataset

def load_benchmark_dataset(name: str, max_samples: Optional[int] = None, data_dir: Optional[str] = None) -> List[Dict]:

    try:
        if name == 'hotpotqa':

            if data_dir:
                local_path = os.path.join(data_dir, "hotpotqa")
                if os.path.exists(local_path):
                    ds = hf_load_dataset(local_path)
                else:
                    ds = hf_load_dataset("hotpotqa/hotpot_qa", "fullwiki")
            else:
                ds = hf_load_dataset("hotpotqa/hotpot_qa", "fullwiki")
            data = [ex for ex in ds["validation"]]
            
        elif name in ['sealqa', 'seal_0', 'seal_hard']:

            dataset_map = {
                'sealqa': 'vtllms/sealqa',
                'seal_0': 'vtllms/sealqa',
                'seal_hard': 'vtllms/sealqa'
            }
            
            if data_dir:
                local_path = os.path.join(data_dir, name)
                if os.path.exists(local_path):
                    ds = hf_load_dataset(local_path)
                else:
                    ds = hf_load_dataset(dataset_map[name], name=name if name != 'sealqa' else "longseal", split="test")
            else:
                ds = hf_load_dataset(dataset_map[name], name=name if name != 'sealqa' else "longseal", split="test")
            data = [ex for ex in ds]
            
        elif name == 'truthfulqa':

            if data_dir:
                data_path = os.path.join(data_dir, "TruthfulQA.csv")
                if not os.path.exists(data_path):
                    data_path = data_dir 
                data = load_truthfulqa_dataset(data_path)
            else:
                data = load_truthfulqa_dataset("./data/TruthfulQA.csv")
                
        else:
            logger.error(f"Unknown dataset: {name}")
            return []
        
        if max_samples and data:
            data = data[:max_samples]
        
        logger.info(f"[{name}] Loaded {len(data)} samples")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")

        if data_dir:
            logger.info(f"Trying to load {name} from local directory: {data_dir}")
            local_path = os.path.join(data_dir, name)
            if os.path.exists(local_path):
                try:
                    ds = hf_load_dataset(local_path)
                    data = [ex for ex in ds]
                    logger.info(f"[{name}] Loaded {len(data)} samples from local directory")
                    if max_samples:
                        data = data[:max_samples]
                    return data
                except Exception as e2:
                    logger.error(f"Failed to load from local directory: {e2}")
        
        return []
def get_dataset_specific_ablation_configs(base_params: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Generate ablation configurations for streamlined DCLED.
    Components: Confidence Gating, Contrastive Strength, Confidence Boost
    REMOVED: Entropy Weighting, JS Divergence
    """
    ablations = {}
    
    # Full DCLED (all remaining components enabled)
    ablations['full_dcled'] = base_params.copy()
    
    # Ablation 1: Remove confidence gating
    config_no_gate = base_params.copy()
    config_no_gate['gen_confidence_threshold'] = -1.0
    ablations['no_confidence_gate'] = config_no_gate
    
    # Ablation 2: Remove contrastive strength
    config_no_contrastive = base_params.copy()
    if 'contrastive_strength' in config_no_contrastive:
        config_no_contrastive['contrastive_strength'] = 0.0
    ablations['no_contrastive'] = config_no_contrastive
    
    # Ablation 3: Remove confidence boost
    config_no_boost = base_params.copy()
    if 'confidence_boost' in config_no_boost:
        config_no_boost['confidence_boost'] = 1.0
    ablations['no_confidence_boost'] = config_no_boost
    
    # Ablation 4: SLED baseline (all DC components removed)
    config_sled = base_params.copy()
    config_sled['gen_confidence_threshold'] = -1.0
    if 'contrastive_strength' in config_sled:
        config_sled['contrastive_strength'] = 0.0
    if 'confidence_boost' in config_sled:
        config_sled['confidence_boost'] = 1.0
    ablations['sled_baseline'] = config_sled
    
    return ablations
def get_dataset_specific_ablation_configs(base_params: Dict[str, Any]) -> Dict[str, Dict]:

    ablations = {}
    
    ablations['full_dcled'] = base_params.copy()
    
    config_no_gate = base_params.copy()
    config_no_gate['gen_confidence_threshold'] = -1.0
    ablations['no_confidence_gate'] = config_no_gate
    
    config_no_contrastive = base_params.copy()
    if 'contrastive_strength' in config_no_contrastive:
        config_no_contrastive['contrastive_strength'] = 0.0
    ablations['no_contrastive'] = config_no_contrastive
    
    config_no_boost = base_params.copy()
    if 'confidence_boost' in config_no_boost:
        config_no_boost['confidence_boost'] = 1.0
    ablations['no_confidence_boost'] = config_no_boost
    
    config_sled = base_params.copy()
    config_sled['gen_confidence_threshold'] = -1.0
    if 'contrastive_strength' in config_sled:
        config_sled['contrastive_strength'] = 0.0
    if 'confidence_boost' in config_sled:
        config_sled['confidence_boost'] = 1.0
    ablations['sled_baseline'] = config_sled
    
    return ablations

def check_available_datasets(data_dir: str = None) -> Dict[str, bool]:

    available = {}
    
    if data_dir:
        truthfulqa_path = os.path.join(data_dir, "TruthfulQA.csv")
        available['truthfulqa'] = os.path.exists(truthfulqa_path)
    else:
        available['truthfulqa'] = os.path.exists("TruthfulQA.csv")
    
    datasets_to_check = ['hotpotqa', 'sealqa', 'seal_0', 'seal_hard']
    for ds_name in datasets_to_check:
        if data_dir:
            local_path = os.path.join(data_dir, ds_name)
            available[ds_name] = os.path.exists(local_path)
        else:
            available[ds_name] = False
    
    return available

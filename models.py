import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from typing import Dict, List, Tuple, Optional, Any
import logging
import math

from config import EPS, LOG_EPS, PROB_CLAMP_MIN, PROB_CLAMP_MAX, LOGIT_CLIP_MAX
from utils import (
    stable_softmax, stable_log_softmax, get_relative_top_filter,
    js_divergence, get_model_size_category
)

logger = logging.getLogger(__name__)

class LLaMAQAStoppingCriteria(StoppingCriteria):
    def __init__(self, list_stop_word_ids: List[List[int]]):
        self.list_stop_word_ids = list_stop_word_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_word_ids in self.list_stop_word_ids:
            if len(stop_word_ids) > 0 and input_ids.shape[-1] >= len(stop_word_ids):
                if input_ids[0, -len(stop_word_ids):].tolist() == stop_word_ids:
                    return True
        return False

class UnifiedDCSLED:
    
    def __init__(self, model_name: str, device: str = 'cuda',
                 num_gpus: str = '1', max_gpu_memory: int = 80):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stopping_criteria = None
        self.stop_words = []
        
        self.model, self.tokenizer = self._load_model(model_name)
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        self.model_size_category = get_model_size_category(model_name)
        
        logger.info(f"[Model] Loaded {model_name}")
        logger.info(f"[Model] {self.num_layers} layers, size: {self.model_size_category}")
    
    def _load_model(self, model_name: str):

        if self.device == "cuda":
            kwargs = {
                "torch_dtype": torch.float16,
                "offload_folder": f"{model_name.replace('/', '_')}/offload"
            }
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(self.num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer_name = model_name
        if 'vicuna' in model_name.lower():
            tokenizer_name = 'huggyllama/llama-7b'
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs
        )
        
        if self.device == "cuda" and self.num_gpus == "1":
            model.cuda()
        
        model.eval()
        return model, tokenizer
    
    def set_stop_words(self, stop_words: List[str]):

        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
        
        self.stopping_criteria.append(LLaMAQAStoppingCriteria(list_stop_word_ids))
    
    def _get_model_device(self):
        if hasattr(self.model, 'device'):
            return self.model.device
        elif hasattr(self.model, 'parameters'):
            return next(self.model.parameters()).device
        else:
            return torch.device(self.device)
    
    def _get_lm_head_device(self):
        lm_head = self.model.lm_head
        if hasattr(lm_head, 'weight'):
            return lm_head.weight.device
        else:
            return next(lm_head.parameters()).device
    
    def _dola_score(
        self,
        dict_outputs: Dict[int, torch.Tensor],
        mature_layer: int,
        candidate_premature_layers: List[int],
        continue_ids: torch.Tensor,
        relative_top: float,
        relative_top_value: float,
        dola_alpha: float
    ) -> Tuple[float, Dict]:

        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
        
        available = [l for l in candidate_premature_layers if l in dict_outputs]
        
        if not available or mature_layer not in dict_outputs:
            out = stable_log_softmax(dict_outputs[mature_layer][0], dim=-1)
            seq_len = out.shape[0]
            
            if len(continue_ids) > seq_len:
                continue_ids = continue_ids[:seq_len]
            elif len(continue_ids) < seq_len:
                logger.error(f"DoLA: continue_ids too short: {len(continue_ids)} < {seq_len}")
                return -100.0, {}
            
            return out[range(seq_len), continue_ids].mean().item(), {}
        
        mature_logits = dict_outputs[mature_layer][0]
        seq_len = mature_logits.shape[0]
        
        if len(continue_ids) > seq_len:
            continue_ids = continue_ids[:seq_len]
        elif len(continue_ids) < seq_len:
            logger.error(f"DoLA: continue_ids too short: {len(continue_ids)} < {seq_len}")
            return -100.0, {}
        
        premature_logits_stack = torch.stack([dict_outputs[l][0] for l in available], dim=0)
        
        softmax_mature = stable_softmax(mature_logits, dim=-1)
        softmax_premature = stable_softmax(premature_logits_stack, dim=-1)
        
        js_divs_per_layer = []
        for layer_idx in range(softmax_premature.shape[0]):
            js_div = js_divergence(softmax_mature, softmax_premature[layer_idx])
            js_divs_per_layer.append(js_div)
        
        js_divs_tensor = torch.stack(js_divs_per_layer, dim=0)
        selected_layer_indices = torch.argmax(js_divs_tensor, dim=0)
        
        base_logits = torch.zeros_like(mature_logits)
        for pos_idx in range(seq_len):
            layer_idx = selected_layer_indices[pos_idx].item()
            selected_layer = available[layer_idx]
            premature_layer_dist[selected_layer] = premature_layer_dist.get(selected_layer, 0) + 1
            base_logits[pos_idx] = premature_logits_stack[layer_idx, pos_idx]
        
        effective_alpha = dola_alpha * min(1.0, seq_len / 3.0)
        
        diff_logits = mature_logits - effective_alpha * base_logits
        
        if relative_top > 0.0:
            relative_top_mask = get_relative_top_filter(mature_logits, relative_top)
            diff_logits = torch.where(
                relative_top_mask,
                torch.tensor(relative_top_value, device=diff_logits.device, dtype=diff_logits.dtype),
                diff_logits
            )
        
        diff_log_probs = stable_log_softmax(diff_logits, dim=-1)
        log_probs = diff_log_probs[range(seq_len), continue_ids].mean().item()
        
        return log_probs, premature_layer_dist
    
    
    def lm_score(
        self,
        input_text1: str,
        input_text2: str,
        mode: str = 'DCLED',
        mature_layer: Optional[int] = None,
        candidate_premature_layers: Optional[List[int]] = None,
        relative_top: float = 0.1,
        relative_top_value: float = -1000.0,
        post_softmax: bool = True,
        dataset_type: str = 'truthfulqa',
        max_seq_length: int = 4096,
        temperature: float = 1.0,
        custom_params: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[float, Optional[Dict]]:
        
        config = custom_params.copy() if custom_params else {}
        for key, value in kwargs.items():
            if value is not None:
                config[key] = value
        
        combined = input_text1 + input_text2
        tokens = self.tokenizer(combined, return_tensors="pt", truncation=False)
        
        if tokens.input_ids.shape[1] > max_seq_length:
            prefix_tokens = self.tokenizer(input_text1, return_tensors="pt", truncation=False)
            suffix_tokens = self.tokenizer(input_text2, return_tensors="pt", truncation=False)
            suffix_len = suffix_tokens.input_ids.shape[1]
            max_prefix_len = max_seq_length - suffix_len - 10
            if max_prefix_len > 0:
                prefix_tokens_truncated = prefix_tokens.input_ids[0, -max_prefix_len:]
                input_text1 = self.tokenizer.decode(prefix_tokens_truncated, skip_special_tokens=True)
        
        with torch.no_grad():
            input_ids = self.tokenizer(
                input_text1 + input_text2, return_tensors="pt"
            ).input_ids.to(self._get_model_device())
            
            prefix_ids = self.tokenizer(
                input_text1, return_tensors="pt"
            ).input_ids.to(self._get_model_device())
            
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            
            if len(continue_ids) == 0:
                return -100.0, None
            
            target_range = slice(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1)
            
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states
            lm_head = self.model.lm_head
            lm_head_device = self._get_lm_head_device()
            
            mature_layer = mature_layer or self.num_layers
            
            if candidate_premature_layers is None:
                start = 0
                end = int(self.num_layers * 0.8)
                candidate_premature_layers = list(range(start, end))
            
            early_exit_layers = list(set(candidate_premature_layers + [mature_layer]))
            
            dict_outputs = {}
            for l in early_exit_layers:
                if l < len(hidden_states):
                    target_hidden = hidden_states[l][:, target_range, :].to(lm_head_device)
                    dict_outputs[l] = lm_head(target_hidden)
            
            if mode == 'VanillaGreedy':
                logits = dict_outputs[mature_layer][0]
                log_probs = stable_log_softmax(logits, dim=-1) if post_softmax else logits
                
                seq_len = log_probs.shape[0]
                if len(continue_ids) > seq_len:
                    continue_ids = continue_ids[:seq_len]
                elif len(continue_ids) < seq_len:
                    return -100.0, None
                
                return log_probs[range(len(continue_ids)), continue_ids].mean().item(), None
            
            if mode == 'dola':
                dola_alpha = config.get('dola_alpha', 1.0)
                return self._dola_score(
                    dict_outputs=dict_outputs,
                    mature_layer=mature_layer,
                    candidate_premature_layers=candidate_premature_layers,
                    continue_ids=continue_ids,
                    relative_top=relative_top,
                    relative_top_value=relative_top_value,
                    dola_alpha=dola_alpha
                )
        
            
            elif mode in ['SLED', 'DCLED']:
                use_dc = mode == 'DCLED'
                
                evolution_rate = config.get('evolution_rate', 2.5)
                evolution_scale = config.get('evolution_scale', 100)
                op_T = config.get('op_T', 12)
                trajectory_data = []
                

                confidence_boost = config.get('confidence_boost', 1.8) if use_dc else 1.0
                signal_strength = config.get('signal_strength', 0.85) if use_dc else 1.0
                contrastive_strength = config.get('contrastive_strength', 0.25) if use_dc else 0.0
                gen_confidence_threshold = config.get('gen_confidence_threshold', 0.88) if use_dc else -1.0
                
                confidence_boost = max(1.0, min(5.0, confidence_boost))
                signal_strength = max(0.1, min(2.0, signal_strength))
                contrastive_strength = max(0.0, min(1.0, contrastive_strength))
                
                mature_logits = dict_outputs[mature_layer][0]
                seq_len = mature_logits.shape[0]
                
                if len(continue_ids) > seq_len:
                    continue_ids = continue_ids[:seq_len]
                elif len(continue_ids) < seq_len:
                    return -100.0, None
                
                available_layers = [l for l in candidate_premature_layers if l in dict_outputs]
                if not available_layers:
                    log_probs = stable_log_softmax(mature_logits, dim=-1)
                    return log_probs[range(len(continue_ids)), continue_ids].sum().item(), None
                
                premature_logits = torch.stack([dict_outputs[l][0] for l in available_layers], dim=0)
                
                softmax_premature = stable_softmax(premature_logits, dim=-1, temperature=temperature)
                P_latent = softmax_premature.mean(dim=0)  
                P_latent = P_latent.clamp(min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)
                P_latent = P_latent / P_latent.sum(dim=-1, keepdim=True)  
                
                mature_probs = stable_softmax(mature_logits, dim=-1, temperature=temperature)
                
                gating_enabled = (use_dc and gen_confidence_threshold > 0)  
                
                if gating_enabled:
                    max_probs, _ = mature_probs.max(dim=-1)
                    gate_mask = max_probs < gen_confidence_threshold
                    num_gated = gate_mask.sum().item()
                else:
                    gate_mask = torch.ones(seq_len, dtype=torch.bool, device=mature_logits.device)
                    num_gated = seq_len
                                
                contrastive_direction = torch.zeros_like(mature_logits)
                
                if use_dc and contrastive_strength > 0 and num_gated > 0:
                   
                    prob_diff = mature_probs - P_latent
                    prob_diff = prob_diff.clamp(min=-1.0, max=1.0)
                    
                    
                    contrastive_direction = prob_diff * 2.0
                    contrastive_direction = contrastive_direction.clamp(min=-5.0, max=5.0)
                
                confidence_multiplier = torch.ones(seq_len, 1, device=mature_logits.device)
                
                if use_dc and confidence_boost > 1.0 and num_gated > 0:

                    top_k = min(5, mature_probs.shape[-1])
                    topk_probs, _ = mature_probs.topk(top_k, dim=-1)
                    
                    top1_prob = topk_probs[:, 0]
                    top5_prob_sum = topk_probs.sum(dim=-1).clamp(min=EPS)
                    
                    confidence_score = (top1_prob / top5_prob_sum).clamp(0, 1)
                    confidence_score = torch.nan_to_num(confidence_score, nan=0.5)
                    
                    boost_factor = min(3.0, confidence_boost)
                    confidence_multiplier = 1.0 + (boost_factor - 1.0) * confidence_score
                    confidence_multiplier = confidence_multiplier.clamp(min=0.5, max=3.0)
                    confidence_multiplier = confidence_multiplier.unsqueeze(-1)
                
                hidden = mature_logits.clone()
                
                n_iterations = min(evolution_scale if 'evolution_scale' in config else op_T, 200)
                n_iterations = max(1, n_iterations)
                
                for t in range(n_iterations):

                    lr_t = evolution_rate * (1.0 - t / n_iterations)
                    lr_t = max(0.01, min(5.0, lr_t))
                    
                    P_current = stable_softmax(hidden, dim=-1, temperature=temperature)

                    if use_dc and t % 2 == 0:  
                        top2_probs, top2_indices = P_current[0].topk(2)
                        correct_token_prob = P_current[0, continue_ids[0]].item() if len(continue_ids) > 0 else 0.0
                        
                        is_top1_correct = (top2_indices[0] == continue_ids[0]) if len(continue_ids) > 0 else False
                        
                        trajectory_data.append({
                            'iteration': t,
                            'top1_prob': top2_probs[0].item(),
                            'top2_prob': top2_probs[1].item() if len(top2_probs) > 1 else 0.0,
                            'top1_token': top2_indices[0].item(),
                            'top2_token': top2_indices[1].item() if len(top2_indices) > 1 else -1,
                            'correct_token_prob': correct_token_prob,
                            'is_correct': is_top1_correct
                        })

                    base_gradient = P_current - P_latent
                    base_gradient = base_gradient.clamp(min=-10.0, max=10.0)
                    

                    final_gradient = base_gradient
                    

                    if use_dc and num_gated > 0:

                        if contrastive_strength > 0:
                            contrastive_contribution = contrastive_strength * contrastive_direction
                            contrastive_contribution = contrastive_contribution.clamp(min=-2.0, max=2.0)
                            final_gradient = final_gradient + contrastive_contribution
                        
                        if confidence_boost > 1.0:
                            final_gradient = final_gradient * confidence_multiplier
                        
                        final_gradient = final_gradient * signal_strength
                    
                    final_gradient = final_gradient.clamp(min=-20.0, max=20.0)
                    final_gradient = torch.nan_to_num(final_gradient, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    hidden = hidden - lr_t * final_gradient
                    hidden = hidden.clamp(min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
                    hidden = torch.nan_to_num(hidden, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
                
                final_logits = torch.where(
                    gate_mask.unsqueeze(-1),
                    hidden,
                    mature_logits
                )
                
                final_logits = torch.nan_to_num(final_logits, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
                
                if relative_top > 0.0:
                    top_mask = get_relative_top_filter(final_logits, relative_top)
                    final_logits = torch.where(
                        top_mask,
                        torch.tensor(relative_top_value, device=final_logits.device, dtype=final_logits.dtype),
                        final_logits
                    )
                

                log_output = stable_log_softmax(final_logits, dim=-1) if post_softmax else final_logits
                log_output = torch.nan_to_num(log_output, nan=-100.0, posinf=0.0, neginf=-100.0)
                
                log_probs = log_output[range(seq_len), continue_ids].sum().item()
                

                if math.isnan(log_probs) or math.isinf(log_probs):
                    logger.warning(f"NaN/Inf detected in {mode}, returning fallback score")
                    vanilla_log_probs = stable_log_softmax(mature_logits, dim=-1)
                    log_probs = vanilla_log_probs[range(len(continue_ids)), continue_ids].sum().item()
                

                metadata = None
                if use_dc:
                    final_probs = stable_softmax(final_logits, dim=-1, temperature=temperature)
                    final_max_prob = final_probs.max(dim=-1).values.max().item() if seq_len > 0 else 0.0
                    final_top1_token = final_probs.argmax(dim=-1)[0].item() if seq_len > 0 else -1
                    
                    initial_probs = stable_softmax(mature_logits, dim=-1, temperature=temperature)
                    initial_max_prob = initial_probs.max(dim=-1).values.max().item() if seq_len > 0 else 0.0
                    initial_top1_token = initial_probs.argmax(dim=-1)[0].item() if seq_len > 0 else -1
                    
                    metadata = {
                        'trajectory': trajectory_data if len(trajectory_data) > 0 else None,  
                        'initial_confidence': initial_max_prob,
                        'final_confidence': final_max_prob,
                        'initial_top1_token': initial_top1_token,
                        'final_top1_token': final_top1_token,
                        'gating_triggered': num_gated > 0, 
                        'num_tokens_gated': num_gated,
                        'total_tokens': seq_len,
                        'n_iterations': n_iterations,
                        'continue_ids': continue_ids.cpu().tolist() if len(continue_ids) > 0 else []
                    }
                
                return log_probs, metadata
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
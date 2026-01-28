import argparse
import os

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streamlined DCLED with Ablation and Statistical Testing"
    )
    
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_names', type=str, 
                            default='meta-llama/Llama-3.1-8B',
                            help='Comma-separated list of models')
    model_group.add_argument('--num_gpus', type=str, default='1')
    model_group.add_argument('--max_gpu_memory', type=int, default=80)
    model_group.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--datasets', type=str, default='all',
                           help='Comma-separated: truthfulqa,hotpotqa,seal_0,seal_hard,sealqa,all')
    data_group.add_argument('--max_samples', type=int, default=None)
    data_group.add_argument('--max_samples_hparam', type=int, default=100,
                           help='Samples for hyperparameter search')
    data_group.add_argument('--truthfulqa_path', type=str, default='./Truthfulqa')
    
    search_group = parser.add_argument_group('Hyperparameter Search')
    search_group.add_argument('--run_hparam_search', action='store_true',
                             help='Run hyperparameter search before evaluation')
    search_group.add_argument('--hparam_trials', type=int, default=20,
                             help='Number of random search trials per method')
    search_group.add_argument('--use_best_hparams', action='store_true',
                             help='Use previously saved best hyperparameters')
    search_group.add_argument('--hparam_file', type=str, 
                             default='best_hyperparameters.json')
    
    ablation_group = parser.add_argument_group('Ablation Study')
    ablation_group.add_argument('--run_ablation', action='store_true',
                               help='Run comprehensive dataset-specific ablation study for DCLED')
    
    method_group = parser.add_argument_group('Methods to Evaluate')
    method_group.add_argument('--methods', type=str, 
                             default='DCLED,SLED,dola,VanillaGreedy',
                             help='Comma-separated list of methods')
    method_group.add_argument('--temperature', type=float, default=1.0)
    method_group.add_argument('--relative_top', type=float, default=0.1)
    
    stats_group = parser.add_argument_group('Statistical Analysis')
    stats_group.add_argument('--n_bootstrap', type=int, default=200)
    stats_group.add_argument('--confidence_level', type=float, default=0.95)
    stats_group.add_argument('--run_significance_tests', action='store_true', default=True)
    
    qual_group = parser.add_argument_group('Qualitative Experiments')
    qual_group.add_argument('--run_qualitative', action='store_true', default=True,
                           help='Run qualitative experiments and collect data')
    qual_group.add_argument('--generate_figures', action='store_true', default=True,
                           help='Generate publication-ready figures')
    
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', type=str, default='./results_ICML_27Jan_8B')
    output_group.add_argument('--verbose', action='store_true')
    output_group.add_argument('--seed', type=int, default=42)
    
    return parser


EPS = 1e-9
LOG_EPS = 1e-12
PROB_CLAMP_MIN = 1e-8
PROB_CLAMP_MAX = 1.0 - 1e-8
LOGIT_CLIP_MAX = 88.0

METHOD_COLORS = {
    'VanillaGreedy': '#95a5a6',
    'dola': '#3498db',
    'SLED': '#e67e22',
    'DCLED': '#27ae60'
}

COMPONENT_COLORS = {
    'Full_DCLED': '#2ecc71',
    'No_Conf_Gate': '#e74c3c',
    'No_Contrastive': '#9b59b6',
    'No_Conf_Boost': '#f39c12',
    'SLED_baseline': '#34495e',
    'DoLA_baseline': '#5dade2',
    'Vanilla_baseline': '#bdc3c7'
}

TRAJECTORY_COLORS = [
    ('#00C853', '#E53935'),
    ('#2196F3', '#FF6F00'),
    ('#9C27B0', '#FFD600'),
    ('#00BCD4', '#D81B60'),
    ('#455A64', '#F57C00'),
]
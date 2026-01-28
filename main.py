import os
import sys
import torch
import numpy as np
import json
import random
import logging
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import create_argument_parser
from utils import (
    get_device, aggressive_memory_cleanup,
    get_model_size, convert_to_serializable
)
from models import UnifiedDCSLED
from evaluation import (
    run_hyperparameter_search, evaluate_truthfulqa_full, evaluate_benchmark_full
)

from data_loader import (
    load_truthfulqa_dataset, load_benchmark_dataset,
    get_dataset_specific_ablation_configs
)
from results import (
    compare_methods_statistically, analyze_ablation_results,
    QualitativeExperimentCollector, generate_all_qualitative_figures
)

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

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    def get_model_size(model_name):
        if '1B' in model_name or '1b' in model_name:
            return '1B'
        elif '3B' in model_name or '3b' in model_name:
            return '3B'
        elif '8B' in model_name or '8b' in model_name:
            return '8B'
        return 'Unknown'    

    model_names = [m.strip() for m in args.model_names.split(',')]
    
    if args.datasets == 'all':
        dataset_names = ['truthfulqa','hotpotqa', 'seal_0', 'seal_hard', 'sealqa']
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
    
    methods = [m.strip() for m in args.methods.split(',')]
    
    best_hparams = {}
    if args.use_best_hparams and os.path.exists(args.hparam_file):
        with open(args.hparam_file, 'r') as f:
            best_hparams = json.load(f)
        logger.info(f"Loaded best hyperparameters from {args.hparam_file}")
    
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING MODEL: {model_name}")
        logger.info(f"{'='*80}\n")

        qual_collector = QualitativeExperimentCollector(args.output_dir)
        logger.info(" Qualitative experiment collector initialized (will collect from all datasets)")
                
        llm = UnifiedDCSLED(
            model_name,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory
        )
        
        model_key = model_name.replace('/', '_')
        model_results = {}
        
        if args.run_hparam_search:
            logger.info(f"\n{'='*80}")
            logger.info(f"HYPERPARAMETER SEARCH PHASE")
            logger.info(f"{'='*80}\n")
            
            hparam_results = {}
            
            for dataset_name in dataset_names:

                if dataset_name == 'truthfulqa':
                    dataset = load_truthfulqa_dataset(args.truthfulqa_path)
                else:
                    dataset = load_benchmark_dataset(dataset_name, args.max_samples_hparam)
                
                if not dataset:
                    continue
                
                dataset_hparams = {}
                
                for method in methods:
                    if method == 'VanillaGreedy':
                        continue
                    
                    hparam_result = run_hyperparameter_search(
                        llm, dataset, dataset_name, method, args
                    )
                    dataset_hparams[method] = hparam_result
                
                hparam_results[dataset_name] = dataset_hparams
            
            hparam_file = os.path.join(
                args.output_dir,
                f"hparam_search_{model_key}.json"
            )
            with open(hparam_file, 'w') as f:
                json.dump(convert_to_serializable(hparam_results), f, indent=2)
            
            logger.info(f"Hyperparameter search results saved to {hparam_file}")
            
            if model_key not in best_hparams:
                best_hparams[model_key] = {}
            
            for dataset_name, dataset_hparams in hparam_results.items():
                if dataset_name not in best_hparams[model_key]:
                    best_hparams[model_key][dataset_name] = {}
                
                for method, hparam_result in dataset_hparams.items():
                    best_hparams[model_key][dataset_name][method] = hparam_result['best_params']
            
            with open(args.hparam_file, 'w') as f:
                json.dump(convert_to_serializable(best_hparams), f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"FULL EVALUATION PHASE")
        logger.info(f"{'='*80}\n")
        
        for method in methods:
            logger.info(f"\n{'='*70}")
            logger.info(f"METHOD: {method}")
            logger.info(f"{'='*70}\n")
            
            method_results = {}
            
            for dataset_name in dataset_names:

                if dataset_name == 'truthfulqa':
                    dataset = load_truthfulqa_dataset(args.truthfulqa_path)
                else:
                    dataset = load_benchmark_dataset(dataset_name, args.max_samples)
                
                if not dataset:
                    logger.warning(f"Dataset {dataset_name} not found or empty")
                    continue
                
                custom_params = None
                if (model_key in best_hparams and
                    dataset_name in best_hparams[model_key] and
                    method in best_hparams[model_key][dataset_name]):
                    custom_params = best_hparams[model_key][dataset_name][method]
                    logger.info(f"Using best hyperparameters: {custom_params}")
                else:
                    logger.info(f"Using default parameters for {method}")
                
                if dataset_name == 'truthfulqa':
                    results = evaluate_truthfulqa_full(
                        llm, dataset, method, args, custom_params,
                        qual_collector=qual_collector 
                    )
                else:
                    results = evaluate_benchmark_full(
                        llm, dataset, dataset_name, method, args, custom_params,
                        qual_collector=qual_collector  
                    )
                
                method_results[dataset_name] = results
                
                if qual_collector and 'latency_avg' in results:
          
                    if 'total_mc2' in results:
                        acc = results['total_mc2']
                    elif 'ranking_accuracy' in results:
                        acc = results['ranking_accuracy']
                    else:
                        acc = 0.0
                    
                    qual_collector.add_latency_measurement(
                        method=method,
                        dataset=dataset_name,
                        latency=results['latency_avg'],
                        accuracy=acc,
                        model_size=get_model_size(model_name)  
                    )
                    
                    logger.info(f"   Latency tracked: {results['latency_avg']:.2f}s, Accuracy: {acc:.4f}")                
               
                intermediate_file = os.path.join(
                    args.output_dir,
                    f"results_{model_key}_{method}_{dataset_name}.json"
                )
                with open(intermediate_file, 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=2)
                
                aggressive_memory_cleanup()
            
            model_results[method] = method_results

            latency_data = {}
            for model_name, model_results in all_results.items():
                latency_data[model_name] = {}
                for method, method_results in model_results.items():
                  
                    all_latencies = []
                    for dataset_name, results in method_results.items():
                        if 'latency_avg' in results:
                            all_latencies.append(results['latency_avg'])
                    
                    if all_latencies:
                        latency_data[model_name][method] = float(np.mean(all_latencies))
                    else:
                        
                        if method == 'VanillaGreedy':
                            latency_data[model_name][method] = 0.5
                        elif method == 'dola':
                            latency_data[model_name][method] = 1.2
                        elif method == 'SLED':
                            latency_data[model_name][method] = 2.0
                        elif method == 'DCLED':
                            latency_data[model_name][method] = 2.5        

        if args.run_ablation and 'DCLED' in methods:
            logger.info(f"\n{'='*80}")
            logger.info(f"DATASET-SPECIFIC ABLATION STUDY: DCLED")
            logger.info(f"{'='*80}\n")
            
            ablation_results = {}
            
            for dataset_name in dataset_names:
                logger.info(f"\n{'='*70}")
                logger.info(f"ABLATION ON DATASET: {dataset_name}")
                logger.info(f"{'='*70}\n")
                
                if dataset_name == 'truthfulqa':
                    dataset = load_truthfulqa_dataset(args.truthfulqa_path)
                else:
                    dataset = load_benchmark_dataset(dataset_name, args.max_samples)
                
                if not dataset:
                    continue
                
                base_params = {}
                if (model_key in best_hparams and
                    dataset_name in best_hparams[model_key] and
                    'DCLED' in best_hparams[model_key][dataset_name]):
                    base_params = best_hparams[model_key][dataset_name]['DCLED']
                    logger.info(f"Using dataset-specific best hyperparameters:")
                    logger.info(f"{json.dumps(base_params, indent=2)}")
                else:
                    logger.warning(f"No best hyperparameters found for DCLED on {dataset_name}, using defaults")
                
                ablation_configs = get_dataset_specific_ablation_configs(base_params)
                
                dataset_ablation = {}
                
                for ablation_name, ablation_config in ablation_configs.items():
                    logger.info(f"\n  Ablation: {ablation_name}")
                    logger.info(f"  Config: {ablation_config}")
                    
                    if dataset_name == 'truthfulqa':
                        results = evaluate_truthfulqa_full(
                            llm, dataset, 'DCLED', args, ablation_config
                        )
                    else:
                        results = evaluate_benchmark_full(
                            llm, dataset, dataset_name, 'DCLED', args, ablation_config
                        )
                    
                    dataset_ablation[ablation_name] = results
                    
                    aggressive_memory_cleanup()
                
                ablation_results[dataset_name] = dataset_ablation
            

            ablation_file = os.path.join(
                args.output_dir,
                f"ablation_{model_key}.json"
            )
            with open(ablation_file, 'w') as f:
                json.dump(convert_to_serializable(ablation_results), f, indent=2)
            
            logger.info(f"Ablation results saved to {ablation_file}")
            
            ablation_analysis = analyze_ablation_results(ablation_results)
            
            ablation_analysis_file = os.path.join(
                args.output_dir,
                f"ablation_analysis_{model_key}.json"
            )
            with open(ablation_analysis_file, 'w') as f:
                json.dump(convert_to_serializable(ablation_analysis), f, indent=2)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ABLATION ANALYSIS SUMMARY")
            logger.info(f"{'='*80}\n")
            
            for dataset_name, analysis in ablation_analysis.items():
                logger.info(f"\nDataset: {dataset_name}")
                logger.info(f"{'-'*70}")
                logger.info(f"\nComponent Ranking (by importance):")
                for rank_info in analysis['component_ranking']:
                    logger.info(f"  {rank_info['rank']}. {rank_info['component']}: "
                              f"-{rank_info['performance_drop']:.4f} "
                              f"({rank_info['relative_drop_pct']:.2f}% drop)")
                
                logger.info(f"\nDetailed Component Contributions:")
                for comp_name, comp_data in analysis['component_contributions'].items():
                    sig = "***" if comp_data['p_value'] < 0.001 else ("**" if comp_data['p_value'] < 0.01 else ("*" if comp_data['significant'] else ""))
                    logger.info(f"  {comp_name}:")
                    logger.info(f"    Performance drop: {comp_data['performance_drop']:.4f} ({comp_data['relative_drop_pct']:.2f}%)")
                    logger.info(f"    p-value: {comp_data['p_value']:.4f} {sig}")
                    logger.info(f"    Cohen's d: {comp_data['cohens_d']:.4f}")
        
        if args.run_significance_tests and len(methods) > 1:
            logger.info(f"\n{'='*80}")
            logger.info(f"STATISTICAL SIGNIFICANCE TESTING")
            logger.info(f"{'='*80}\n")
            
            comparisons = compare_methods_statistically(model_results, 'VanillaGreedy', 'DCLED', args)
            
            comparison_file = os.path.join(
                args.output_dir,
                f"statistical_comparisons_{model_key}.json"
            )
            with open(comparison_file, 'w') as f:
                json.dump(convert_to_serializable(comparisons), f, indent=2)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"DCLED vs ALL OTHER METHODS")
            logger.info(f"{'='*70}\n")
            
            for dataset_name, dataset_comparisons in comparisons['dcled_vs_all'].items():
                logger.info(f"\nDataset: {dataset_name}")
                logger.info(f"{'-'*70}")
                for method, stats in dataset_comparisons.items():
                    sig = "***" if stats['significant_at_0.001'] else ("**" if stats['significant_at_0.01'] else ("*" if stats['significant_at_0.05'] else ""))
                    logger.info(f"  DCLED vs {method}:")
                    logger.info(f"    Winner: {stats['winner']}")
                    logger.info(f"    Improvement: {stats['absolute_improvement']:.4f} ({stats['relative_improvement_pct']:.2f}%)")
                    logger.info(f"    p-value: {stats['p_value']:.4f} {sig}")
                    logger.info(f"    Cohen's d: {stats['cohens_d']:.4f}")
                    logger.info(f"    DCLED: {stats['dcled_mean']:.4f} | {method}: {stats['method_mean']:.4f}")
            
            logger.info(f"\n{'='*70}")
            logger.info(f"ALL METHODS vs BASELINE (VanillaGreedy)")
            logger.info(f"{'='*70}\n")
            
            for dataset_name, dataset_comparisons in comparisons['methods_vs_baseline'].items():
                logger.info(f"\nDataset: {dataset_name}")
                logger.info(f"{'-'*70}")
                for method, stats in dataset_comparisons.items():
                    sig = "***" if stats['significant_at_0.001'] else ("**" if stats['significant_at_0.01'] else ("*" if stats['significant_at_0.05'] else ""))
                    logger.info(f"  {method} vs VanillaGreedy:")
                    logger.info(f"    Improvement: {stats['absolute_improvement']:.4f} ({stats['relative_improvement_pct']:.2f}%)")
                    logger.info(f"    p-value: {stats['p_value']:.4f} {sig}")
                    logger.info(f"    Cohen's d: {stats['cohens_d']:.4f}")
        
        all_results[model_name] = model_results
        
        if qual_collector:
            logger.info(f"\n{'='*80}")
            logger.info("SAVING QUALITATIVE EXPERIMENT DATA")
            logger.info(f"{'='*80}\n")
            
            qual_collector.save_all_experiments()
            
            logger.info("Qualitative data collection summary:")
            logger.info(f"   Trajectories: {len(qual_collector.evolution_trajectories)} collected")
            logger.info(f"     Dataset distribution: {qual_collector.trajectory_dataset_counts}")
            logger.info(f"   Gating decisions: {len(qual_collector.confidence_scatter_data)} collected")
            logger.info(f"   Failure cases: {len(qual_collector.failure_cases)} collected")
            
            if qual_collector.failure_cases:
                from collections import Counter
                failure_datasets = Counter(f.get('dataset', 'unknown') for f in qual_collector.failure_cases)
                logger.info(f"     Failure case distribution: {dict(failure_datasets)}")
            
            failure_json = os.path.join(args.output_dir, 'experiment4_failure_cases.json')
            if os.path.exists(failure_json):
                with open(failure_json) as f:
                    cases = json.load(f)
                
                logger.info(f"\n{'='*80}")
                logger.info(f"FAILURE CASES VERIFICATION")
                logger.info(f"{'='*80}\n")
                
                consistent_count = 0
                warning_count = 0
                
                for i, case in enumerate(cases, 1):
                    dcled_ans = case.get('dcled_answer', '').strip().lower()
                    correct_ans = case.get('correct_answer', '').strip().lower()
                    dcled_correct = case.get('dcled_correct', False)
                    
                    answers_match = (dcled_ans == correct_ans or 
                                   dcled_ans in correct_ans or 
                                   correct_ans in dcled_ans)
                    
                    if answers_match and not dcled_correct:
                        logger.warning(f"    Case {i}: Answers match but marked wrong!")
                        logger.warning(f"      DCLED: {case['dcled_answer'][:60]}...")
                        logger.warning(f"      Correct: {case['correct_answer'][:60]}...")
                        warning_count += 1
                    elif not answers_match and dcled_correct:
                        logger.warning(f"    Case {i}: Answers differ but marked correct!")
                        warning_count += 1
                    else:
                        consistent_count += 1
                
                if warning_count == 0:
                    logger.info(f"   All {len(cases)} failure cases are logically consistent!")
                else:
                    logger.warning(f" Found {warning_count} inconsistent cases out of {len(cases)}")
                    logger.warning(f" Please check the failure case collection logic")
        
        del llm
        aggressive_memory_cleanup()

    final_file = os.path.join(args.output_dir, 'final_results.json')
    with open(final_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL EVALUATIONS COMPLETE")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"{'='*80}\n")

    if qual_collector:
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING QUALITATIVE EXPERIMENT FIGURES")
        logger.info(f"{'='*80}\n")
        
        try:
            generate_all_qualitative_figures(
                qual_collector=qual_collector,
                results_dict=all_results,
                ablation_analysis=ablation_analysis,
                output_dir=args.output_dir
            )
            logger.info(" All qualitative figures generated successfully!")
            logger.info(f"  Check {args.output_dir} for:")
            logger.info(f"     - 7 JSON experiment data files")
            logger.info(f"     - 6 PNG publication-ready figures")
        except Exception as e:
            logger.error(f" Error generating qualitative figures: {e}")
            import traceback
            traceback.print_exc()    

    logger.info("\nFINAL SUMMARY TABLE:")
    logger.info("="*80)
    
    for model_name, model_results in all_results.items():
        logger.info(f"\nModel: {model_name}")
        logger.info("-" * 80)
        
        for method, method_results in model_results.items():
            logger.info(f"\n  Method: {method}")
            
            for dataset_name, results in method_results.items():
                if 'total_mc2' in results:
                    ci_range = results.get('mc2_ci_upper', 0) - results['total_mc2']
                    logger.info(f"    {dataset_name}: MC2 = {results['total_mc2']:.4f} +/- {ci_range:.4f}")
                elif 'ranking_accuracy' in results:
                    ci_range = results.get('acc_ci_upper', 0) - results['ranking_accuracy']
                    logger.info(f"    {dataset_name}: Acc = {results['ranking_accuracy']:.4f} +/- {ci_range:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
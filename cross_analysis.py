"""
Cross-dataset analysis functions
Compares results across ImageNet and CIFAR-10
"""

import numpy as np
import json
from pathlib import Path

def create_cross_dataset_analysis(all_dataset_results):
    """Create comprehensive cross-dataset comparison"""
    
    print(f"\n{'='*80}")
    print("CROSS-DATASET ANALYSIS")
    print(f"{'='*80}")
    
    # Extract common models across datasets
    dataset_names = list(all_dataset_results.keys())
    common_models = set(all_dataset_results[dataset_names[0]]['results'].keys())
    
    for dataset_name in dataset_names[1:]:
        common_models &= set(all_dataset_results[dataset_name]['results'].keys())
    
    print(f"Common models across datasets: {len(common_models)}")
    print(f"Datasets compared: {', '.join(dataset_names)}")
    
    # Create side-by-side comparison table
    print(f"\nCROSS-DATASET COMPARISON TABLE:")
    print(f"{'Model':<20}", end='')
    for dataset_name in dataset_names:
        print(f" {dataset_name:<25}", end='')
    print()
    
    print(f"{'':20}", end='')
    for _ in dataset_names:
        print(f" {'MSE':<6} {'Div':<6} {'Sev':<6} {'Lit':<6}", end='')
    print()
    print("-" * (20 + len(dataset_names) * 25))
    
    for model_name in sorted(common_models):
        print(f"{model_name:<20}", end='')
        for dataset_name in dataset_names:
            results = all_dataset_results[dataset_name]['results'][model_name]
            mse = f"{results.get('reconstruction_mse', 0):.3f}"
            div = f"{results.get('condition_diversity', 0):.3f}"
            sev = f"{results.get('severity_scaling', 0):.3f}"
            lit = f"{results.get('literature_consistency', 0):.3f}"
            print(f" {mse:<6} {div:<6} {sev:<6} {lit:<6}", end='')
        print()
    
    # Statistical analysis
    cross_dataset_stats = {}
    
    for model_name in common_models:
        model_stats = {'datasets': {}}
        
        for dataset_name in dataset_names:
            results = all_dataset_results[dataset_name]['results'][model_name]
            model_stats['datasets'][dataset_name] = results
        
        # Calculate cross-dataset consistency (coefficient of variation)
        metrics = ['reconstruction_mse', 'condition_diversity', 'severity_scaling', 'literature_consistency']
        consistency_scores = {}
        
        for metric in metrics:
            values = [model_stats['datasets'][ds].get(metric, 0) for ds in dataset_names]
            if all(v > 0 for v in values):  # Avoid division by zero
                mean_val = np.mean(values)
                std_val = np.std(values)
                consistency_scores[metric] = std_val / mean_val if mean_val > 0 else 1.0
            else:
                consistency_scores[metric] = 1.0
        
        model_stats['consistency'] = consistency_scores
        cross_dataset_stats[model_name] = model_stats
    
    # Find most consistent models
    print(f"\nMODEL CONSISTENCY ANALYSIS:")
    print("(Lower coefficient of variation = more consistent across datasets)")
    
    for model_name in sorted(common_models):
        avg_consistency = np.mean(list(cross_dataset_stats[model_name]['consistency'].values()))
        print(f"{model_name:<20}: {avg_consistency:.3f}")
    
    # Save cross-dataset analysis
    cross_analysis = {
        'datasets_compared': dataset_names,
        'common_models': list(common_models),
        'detailed_comparison': cross_dataset_stats,
        'summary_tables': {
            'by_model': {},
            'by_dataset': {}
        }
    }
    
    # Organize by model
    for model_name in common_models:
        cross_analysis['summary_tables']['by_model'][model_name] = {}
        for dataset_name in dataset_names:
            cross_analysis['summary_tables']['by_model'][model_name][dataset_name] = \
                all_dataset_results[dataset_name]['results'][model_name]
    
    # Organize by dataset
    for dataset_name in dataset_names:
        cross_analysis['summary_tables']['by_dataset'][dataset_name] = \
            all_dataset_results[dataset_name]['results']
    
    Path('outputs/cross_dataset_analysis').mkdir(parents=True, exist_ok=True)
    with open('outputs/cross_dataset_analysis/comprehensive_comparison.json', 'w') as f:
        json.dump(cross_analysis, f, indent=2)
    
    # Create publication-ready tables
    create_publication_tables(cross_analysis)
    
    print(f"\n✓ Cross-dataset analysis saved to 'outputs/cross_dataset_analysis/'")
    
    return cross_analysis

def create_publication_tables(cross_analysis):
    """Create LaTeX tables for publication"""
    
    dataset_names = cross_analysis['datasets_compared']
    common_models = cross_analysis['common_models']
    
    # Table 1: Full comparison table
    latex_table1 = "\\begin{table}[h!]\n\\centering\n"
    latex_table1 += "\\caption{Model Performance Comparison Across Datasets}\n"
    latex_table1 += "\\begin{tabular}{l" + "cccc" * len(dataset_names) + "}\n"
    latex_table1 += "\\hline\n"
    latex_table1 += "Model & " + " & ".join([f"\\multicolumn{{4}}{{c}}{{{ds}}}" for ds in dataset_names]) + " \\\\\n"
    latex_table1 += "& " + " & ".join(["MSE & Div & Sev & Lit"] * len(dataset_names)) + " \\\\\n"
    latex_table1 += "\\hline\n"
    
    for model_name in sorted(common_models):
        # Fix: Extract the underscore replacement outside f-string
        model_display = model_name.replace('_', '\\_')
        row = model_display
        for dataset_name in dataset_names:
            results = cross_analysis['summary_tables']['by_model'][model_name][dataset_name]
            mse = f"{results.get('reconstruction_mse', 0):.3f}"
            div = f"{results.get('condition_diversity', 0):.3f}"
            sev = f"{results.get('severity_scaling', 0):.3f}"
            lit = f"{results.get('literature_consistency', 0):.3f}"
            row += f" & {mse} & {div} & {sev} & {lit}"
        latex_table1 += row + " \\\\\n"
    
    latex_table1 += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    # Table 2: Best models by metric
    latex_table2 = "\\begin{table}[h!]\n\\centering\n"
    latex_table2 += "\\caption{Best Performing Models by Metric and Dataset}\n"
    latex_table2 += "\\begin{tabular}{l" + "c" * len(dataset_names) + "}\n"
    latex_table2 += "\\hline\n"
    latex_table2 += "Metric & " + " & ".join(dataset_names) + " \\\\\n"
    latex_table2 += "\\hline\n"
    
    metrics_info = [
        ('Reconstruction MSE', 'reconstruction_mse', min),
        ('Condition Diversity', 'condition_diversity', max),
        ('Severity Scaling', 'severity_scaling', max),
        ('Literature Consistency', 'literature_consistency', max)
    ]
    
    for metric_name, metric_key, best_func in metrics_info:
        row = metric_name
        for dataset_name in dataset_names:
            dataset_results = cross_analysis['summary_tables']['by_dataset'][dataset_name]
            best_model = best_func(dataset_results.items(), 
                                 key=lambda x: x[1].get(metric_key, 0 if best_func == max else float('inf')))
            # Fix: Extract the underscore replacement outside f-string
            best_model_display = best_model[0].replace('_', '\\_')
            row += f" & {best_model_display}"
        latex_table2 += row + " \\\\\n"
    
    latex_table2 += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    # Save LaTeX tables
    with open('outputs/cross_dataset_analysis/publication_table_full.tex', 'w') as f:
        f.write(latex_table1)
    
    with open('outputs/cross_dataset_analysis/publication_table_best.tex', 'w') as f:
        f.write(latex_table2)
    
    print("✓ LaTeX tables created for publication")
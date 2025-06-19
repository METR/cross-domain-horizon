
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
from dataclasses import dataclass
from typing import Optional

from plotting_aliases import benchmark_aliases, plotting_aliases, benchmark_colors
from wrangle import normalize_model_name

import toml

@dataclass
class BenchmarkGroupSpec:
    benchmark: str
    models: list[str]
    num_splits: Optional[int] = None
    dividers: Optional[list[float]] = None


def plot_splits(df: pd.DataFrame, output_path: pathlib.Path):
    # Hardcoded benchmark -> models dictionary (non-group benchmarks)
    benchmark_models = {
        'livecodebench_2411_2505': ['o4 mini', 'claude 3.7 sonnet', 'claude 3 haiku'],
        'video_mme': ['google_gemini_1_5_pro_002', 'openai_gpt_4_vision']
    }

    # Configuration using BenchmarkGroupSpec for group benchmarks
    benchmark_specs = [
        BenchmarkGroupSpec(
            benchmark='gpqa_diamond', 
            models=['google/gemini-2.5-pro-exp-03-25', 'anthropic/claude-3-7-sonnet-20250219', 'openai/gpt-4-1106-preview'],
            num_splits=5
        ),
        BenchmarkGroupSpec(
            benchmark='swe_bench_verified', 
            models=['anthropic/claude-sonnet-4-20250514'],
            dividers=[4, 15, 60]
        )
    ]
    
    # Load data for each benchmark
    data_points = []
    
    # Process regular benchmarks
    for benchmark, models in benchmark_models.items():
        # Load scores
        scores_path = f'data/scores/{benchmark}.toml'
        with open(scores_path, 'r') as f:
            scores_data = toml.load(f)
        
        # Load task lengths
        benchmarks_path = f'data/benchmarks/{benchmark}.toml'
        with open(benchmarks_path, 'r') as f:
            benchmark_data = toml.load(f)
        
        for split_name, split_scores in scores_data['splits'].items():
            if split_name in benchmark_data['splits']:
                task_lengths = benchmark_data['splits'][split_name]['lengths']
                geom_mean_length = stats.gmean(task_lengths)
                
                for model in models:
                    if model in split_scores:
                        score = split_scores[model]
                        data_points.append({
                            'benchmark': benchmark,
                            'model': model,
                            'split': split_name,
                            'score': score,
                            'geom_mean_length': geom_mean_length
                        })
    
    # Process grouped benchmarks using BenchmarkGroupSpec
    for spec in benchmark_specs:
        benchmark = spec.benchmark
        models = spec.models
        
        # Load scores
        scores_path = f'data/scores/{benchmark}.toml'
        with open(scores_path, 'r') as f:
            scores_data = toml.load(f)
        
        # Load task lengths
        benchmarks_path = f'data/benchmarks/{benchmark}.toml'
        with open(benchmarks_path, 'r') as f:
            benchmark_data = toml.load(f)
        
        # Collect all splits with their lengths and question counts
        splits_info = []
        for split_name, split_scores in scores_data['splits'].items():
            if split_name in benchmark_data['splits']:
                task_lengths = benchmark_data['splits'][split_name]['lengths']
                geom_mean_length = stats.gmean(task_lengths)
                num_questions = len(task_lengths)
                splits_info.append({
                    'split_name': split_name,
                    'geom_mean_length': geom_mean_length,
                    'num_questions': num_questions,
                    'scores': split_scores
                })
        
        # Sort by length
        splits_info.sort(key=lambda x: x['geom_mean_length'])
        n_splits = len(splits_info)
        
        # Determine grouping method based on spec
        if spec.num_splits is not None:
            # Use quantiles
            num_groups = spec.num_splits
            group_size = n_splits // num_groups
            
            for group_idx in range(num_groups):
                start_idx = group_idx * group_size
                if group_idx == num_groups - 1:  # Last group gets any remainder
                    end_idx = n_splits
                else:
                    end_idx = (group_idx + 1) * group_size
                
                group_splits = splits_info[start_idx:end_idx]
                
                # Calculate group representative length (use median of the group for consistent ordering)
                group_lengths = [s['geom_mean_length'] for s in group_splits]
                group_geom_mean_length = np.median(group_lengths)
                
                # Calculate weighted average score for each model
                for model in models:
                    total_weighted_score = 0
                    total_questions = 0
                    
                    for split_info in group_splits:
                        if model in split_info['scores']:
                            score = split_info['scores'][model]
                            num_questions = split_info['num_questions']
                            total_weighted_score += score * num_questions
                            total_questions += num_questions
                    
                    if total_questions > 0:
                        weighted_avg_score = total_weighted_score / total_questions
                        display_model = normalize_model_name(model)
                        print(f"Model: {display_model}, Group: {group_idx + 1}, Score: {weighted_avg_score}, Length: {group_geom_mean_length}")
                        data_points.append({
                            'benchmark': benchmark,
                            'model': display_model,
                            'split': f'group_{group_idx + 1}',
                            'score': weighted_avg_score,
                            'geom_mean_length': group_geom_mean_length
                        })
                        
        elif spec.dividers is not None:
            # Use specified dividers
            dividers = spec.dividers
            
            # Assert dividers are increasing
            assert all(dividers[i] < dividers[i+1] for i in range(len(dividers)-1)), \
                f"Dividers must be increasing: {dividers}"
            
            # Create groups based on dividers
            groups = []
            current_group = []
            divider_idx = 0
            
            for split_info in splits_info:
                length = split_info['geom_mean_length']
                
                # Check if we need to move to the next group
                while divider_idx < len(dividers) and length >= dividers[divider_idx]:
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                    divider_idx += 1
                
                current_group.append(split_info)
            
            # Add the last group
            if current_group:
                groups.append(current_group)
            
            # Assert each group is nonempty
            assert all(len(group) > 0 for group in groups), \
                f"All groups must be nonempty, but got groups with sizes: {[len(g) for g in groups]}"
            
            # Process each group
            for group_idx, group_splits in enumerate(groups):
                group_lengths = [s['geom_mean_length'] for s in group_splits]
                group_geom_mean_length = np.median(group_lengths)
                
                # Calculate weighted average score for each model
                for model in models:
                    total_weighted_score = 0
                    total_questions = 0
                    
                    for split_info in group_splits:
                        if model in split_info['scores']:
                            score = split_info['scores'][model]
                            num_questions = split_info['num_questions']
                            total_weighted_score += score * num_questions
                            total_questions += num_questions
                    
                    if total_questions > 0:
                        weighted_avg_score = total_weighted_score / total_questions
                        display_model = normalize_model_name(model)
                        
                        # Create group label based on dividers
                        if group_idx == 0:
                            group_label = f"<{dividers[0]}min"
                        elif group_idx == len(groups) - 1:
                            group_label = f">{dividers[-1]}min"
                        else:
                            group_label = f"{dividers[group_idx-1]}-{dividers[group_idx]}min"
                        
                        print(f"Model: {display_model}, Group: {group_label}, Score: {weighted_avg_score}, Length: {group_geom_mean_length}")
                        data_points.append({
                            'benchmark': benchmark,
                            'model': display_model,
                            'split': group_label,
                            'score': weighted_avg_score,
                            'geom_mean_length': group_geom_mean_length
                        })
        else:
            raise ValueError(f"BenchmarkGroupSpec for {benchmark} must specify either num_splits or dividers")
    
    # Manually add GPT-4o HCAST/RS data
    hcast_buckets = [
        ('1-4 min', 2.0, 0.73569),      # geomean of 1-4 = 2
        ('4-16 min', 8.0, 0.614769),    # geomean of 4-16 = 8  
        ('16-64 min', 32.0, 0.237548),  # geomean of 16-64 = 32
        ('64-256 min', 128.0, 0.131602), # geomean of 64-256 = 128
        ('256-960 min', 496.0, 0.0)     # geomean of 256-960 = 496
    ]
    
    for bucket_name, geom_mean_length, score in hcast_buckets:
        data_points.append({
            'benchmark': 'hcast_r_s',
            'model': 'GPT-4o',
            'split': bucket_name,
            'score': score * 100,  # Convert to percentage
            'geom_mean_length': geom_mean_length
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique benchmarks - colors already defined in benchmark_colors
    benchmarks = df['benchmark'].unique()
    
    # Plot each (benchmark, model) group and track legend entries
    legend_entries = {}
    
    # Create set of grouped benchmark names
    grouped_benchmarks = {spec.benchmark for spec in benchmark_specs}
    
    for (benchmark, model), group in df.groupby(['benchmark', 'model']):
        # Sort by appropriate column for line connection
        if benchmark in grouped_benchmarks:
            # For grouped benchmarks, sort by x-axis (length) to ensure proper ordering
            group_sorted = group.sort_values('geom_mean_length')
        else:
            # For regular benchmarks, sort by score
            group_sorted = group.sort_values('score')
        
        color = benchmark_colors[benchmark]
        
        # Create legend entry for this benchmark if not already created
        if benchmark not in legend_entries:
            # Collect all models for this benchmark
            benchmark_df = df[df['benchmark'] == benchmark]
            models_in_benchmark = benchmark_df['model'].unique()
            model_names = [plotting_aliases.get(m, m) for m in models_in_benchmark]
            model_list = '\n '.join(model_names)
            legend_label = f'{benchmark_aliases.get(benchmark, benchmark)}:\n {model_list}'
            legend_entries[benchmark] = legend_label
            plot_label = legend_label
        else:
            # Don't add to legend again
            plot_label = None
        
        # Plot points and connect with lines
        plt.plot(group_sorted['geom_mean_length'], group_sorted['score'], 
                'o-', color=color, label=plot_label, alpha=0.7, linewidth=2, markersize=6)
    

    plt.xscale('log')

    plt.xlabel('Task Length for Humans (minutes, geometric mean of split)')
    plt.ylabel('Score (%)')
    plt.title('Model Success Rate vs Task Length by Benchmark and Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pathlib
from scipy import stats

from plotting_aliases import benchmark_aliases, plotting_aliases
from wrangle import normalize_model_name

import toml

def plot_splits(df: pd.DataFrame, output_path: pathlib.Path):
    # Configuration
    NUM_GROUPS = 5
    
    # Hardcoded benchmark -> models dictionary
    benchmark_models = {
        'livecodebench_2411_2505': ['o4 mini', 'claude 3.7 sonnet', 'claude 3 haiku'],
        'video_mme': ['google_gemini_1_5_pro_002', 'openai_gpt_4_vision']
    }

    group_benchmark_models = {
        'gpqa_diamond': ['openai/gpt-4o-2024-05-13', 'anthropic/claude-3-7-sonnet-20250219', 'anthropic/claude-3-haiku-20240307', 'google/gemini-2.5-pro-exp-03-25']
    }
    
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
    
    # Process grouped benchmarks
    for benchmark, models in group_benchmark_models.items():
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
        
        # Sort by length and create groups
        splits_info.sort(key=lambda x: x['geom_mean_length'])
        n_splits = len(splits_info)
        group_size = n_splits // NUM_GROUPS
        
        for group_idx in range(NUM_GROUPS):
            start_idx = group_idx * group_size
            if group_idx == NUM_GROUPS - 1:  # Last group gets any remainder
                end_idx = n_splits
            else:
                end_idx = (group_idx + 1) * group_size
            
            group_splits = splits_info[start_idx:end_idx]
            
            # Calculate group representative length (use median of the group for consistent ordering)
            group_lengths = [s['geom_mean_length'] for s in group_splits]
            # Use median to ensure quintiles remain in order
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
                    # Use normalized name for display
                    display_model = normalize_model_name(model)
                    print(f"Model: {display_model}, Group: {group_idx + 1}, Score: {weighted_avg_score}, Length: {group_geom_mean_length}")
                    data_points.append({
                        'benchmark': benchmark,
                        'model': display_model,  # Use normalized name for display
                        'split': f'group_{group_idx + 1}',
                        'score': weighted_avg_score,
                        'geom_mean_length': group_geom_mean_length
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique benchmarks and assign colors
    benchmarks = df['benchmark'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(benchmarks)))
    benchmark_colors = dict(zip(benchmarks, colors))
    
    # Plot each (benchmark, model) group
    for (benchmark, model), group in df.groupby(['benchmark', 'model']):
        # Sort by appropriate column for line connection
        if benchmark in group_benchmark_models:
            # For grouped benchmarks, sort by x-axis (length) to ensure proper ordering
            group_sorted = group.sort_values('geom_mean_length')
        else:
            # For regular benchmarks, sort by score
            group_sorted = group.sort_values('score')
        
        color = benchmark_colors[benchmark]
        label = f'{benchmark_aliases[benchmark]} \n- {plotting_aliases.get(model, model)}'
        
        # Plot points and connect with lines
        plt.plot(group_sorted['geom_mean_length'], group_sorted['score'], 
                'o-', color=color, label=label, alpha=0.7, linewidth=2, markersize=6)
    

    plt.xscale('log')

    plt.xlabel('Task Length (minutes, geometric mean of split)')
    plt.ylabel('Score (%)')
    plt.title('Performance vs Task Length by Benchmark and Model')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")
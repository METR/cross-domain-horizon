
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pathlib
from scipy import stats

from plotting_aliases import benchmark_aliases, plotting_aliases

import toml

def plot_splits(df: pd.DataFrame, output_path: pathlib.Path):
    # Hardcoded benchmark -> models dictionary
    benchmark_models = {
        'livecodebench_2411_2505': ['o4 mini', 'claude 3.7 sonnet', 'claude 3 haiku'],
        'video_mme': ['google_gemini_1_5_pro_002', 'openai_gpt_4_vision']
    }
    
    # Load data for each benchmark
    data_points = []
    
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
        # Sort by score for line connection
        group_sorted = group.sort_values('score')
        
        color = benchmark_colors[benchmark]
        label = f'{benchmark_aliases[benchmark]} - {plotting_aliases.get(model, model)}'
        
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
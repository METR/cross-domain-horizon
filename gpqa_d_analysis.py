import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from pathlib import Path

def load_and_analyze_gpqa_diamond():
    # Load scores data
    with open('data/scores/gpqa_diamond.toml', 'r') as f:
        scores_data = toml.load(f)
    
    # Load benchmark data (question lengths)
    with open('data/benchmarks/gpqa_diamond.toml', 'r') as f:
        benchmark_data = toml.load(f)
    
    # Prepare data for analysis
    analysis_data = []
    
    for split_name, split_scores in scores_data['splits'].items():
        if split_name in benchmark_data['splits']:
            # Get question length (assuming single length per split)
            question_length = benchmark_data['splits'][split_name]['lengths'][0]
            
            # Calculate average success rate across all models for this split
            scores = list(split_scores.values())
            avg_success_rate = np.mean(scores)
            
            analysis_data.append({
                'split': split_name,
                'question_length': question_length,
                'avg_success_rate': avg_success_rate,
                'num_models': len(scores)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Create scatterplot with log-transformed question length
    plt.figure(figsize=(10, 6))
    log_question_length = np.log(df['question_length'])
    plt.scatter(log_question_length, df['avg_success_rate'], alpha=0.6, s=60)
    
    plt.xlabel('Log(Question Length) (minutes)')
    plt.ylabel('Average Model Success Rate (%)')
    plt.title('GPQA Diamond: Average Success Rate vs Log(Question Length)')
    plt.grid(True, alpha=0.3)
    
    # Add trend line and correlation
    z = np.polyfit(log_question_length, df['avg_success_rate'], 1)
    p = np.poly1d(z)
    correlation = np.corrcoef(log_question_length, df['avg_success_rate'])[0,1]
    plt.plot(log_question_length, p(log_question_length), "r--", alpha=0.8, 
             label=f'Trend line (slope: {z[0]:.2f}, r: {correlation:.3f})')
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'gpqa_diamond_analysis.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print(f"Analysis of {len(df)} GPQA Diamond questions")
    print(f"Question length range: {df['question_length'].min():.1f} - {df['question_length'].max():.1f} minutes")
    print(f"Average success rate range: {df['avg_success_rate'].min():.1f}% - {df['avg_success_rate'].max():.1f}%")
    print(f"Correlation coefficient (log length vs success rate): {np.corrcoef(np.log(df['question_length']), df['avg_success_rate'])[0,1]:.3f}")
    print(f"Plot saved to plots/gpqa_diamond_analysis.png")
    
    return df

if __name__ == "__main__":
    df = load_and_analyze_gpqa_diamond()
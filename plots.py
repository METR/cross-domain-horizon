import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import gmean
import numpy as np
import adjustText # Import adjustText

DATA_DIR = 'data'
BAR_PLOT_OUTPUT_FILE = 'plots/all.png'
SCATTER_PLOT_OUTPUT_FILE = 'plots/scatter.png' # New output file
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes
Y_AXIS_MIN_SECONDS = 60  # 1 minute

def load_data(data_dir):
    """Loads horizon data from all benchmark subdirectories."""
    all_data = []
    benchmark_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]

    for bench_dir in benchmark_dirs:
        benchmark_name = os.path.basename(bench_dir)
        csv_path = os.path.join(bench_dir, 'horizons.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Assuming columns are 'model', 'horizon' (in minutes), 'score'
                if 'model' in df.columns and 'horizon' in df.columns:
                    # Fill missing horizons with 0 before converting
                    df['horizon'] = df['horizon'].fillna(0)
                    # Convert horizon from minutes to seconds
                    df['horizon'] = df['horizon'] * 60
                    df['benchmark'] = benchmark_name
                    all_data.append(df[['model', 'benchmark', 'horizon']])
                else:
                    print(f"Warning: Skipping {csv_path}. Missing 'model' or 'horizon' column.")
            except Exception as e:
                print(f"Warning: Could not read {csv_path}. Error: {e}")
        else:
            print(f"Warning: No horizons.csv found in {bench_dir}")

    if not all_data:
        return pd.DataFrame(columns=['model', 'benchmark', 'horizon'])

    return pd.concat(all_data, ignore_index=True)

def filter_and_sort_models(df):
    """Filters models based on horizon threshold and sorts by geometric mean."""
    if df.empty:
        return df, []

    # Filter models: must have >= 10 min horizon on at least one benchmark
    max_horizons = df.groupby('model')['horizon'].max()
    models_to_keep = max_horizons[max_horizons >= MIN_HORIZON_THRESHOLD_SECONDS].index
    filtered_df = df[df['model'].isin(models_to_keep)].copy()

    if filtered_df.empty:
        return filtered_df, []

    # Calculate geometric mean horizon for sorting
    # Pivot to get models as index, benchmarks as columns, horizon as values
    pivot_df = filtered_df.pivot(index='model', columns='benchmark', values='horizon')
    # Replace NaN/0 with a small value (e.g., 1 second) for gmean calculation
    pivot_df = pivot_df.fillna(1).clip(lower=1)

    # Calculate geometric mean across benchmarks for each model
    model_gmean = pivot_df.apply(gmean, axis=1)

    # Sort models by geometric mean ascending
    sorted_models = model_gmean.sort_values().index.tolist()

    return filtered_df, sorted_models

def plot_horizons(df, sorted_models):
    """Generates and saves the grouped bar plot."""
    if df.empty or not sorted_models:
        print("No data to plot after filtering for bar plot.")
        return

    # Get unique benchmarks for hue order
    benchmarks = df['benchmark'].unique()
    benchmarks.sort()

    plt.figure(figsize=(15, 8)) # Increased figure size for potentially many models

    # Create the bar plot - SWAPPED x and hue
    ax = sns.barplot(
        data=df,
        x='model',
        y='horizon',
        hue='benchmark',
        order=sorted_models,  # Use the sorted model order for the x-axis
        hue_order=benchmarks # Use sorted benchmarks for hue
    )

    # Set y-axis to log scale starting at 1 minute
    ax.set_yscale('log')
    ax.set_ylim(bottom=Y_AXIS_MIN_SECONDS)

    # Add labels and title - UPDATED x-label
    ax.set_xlabel("Model (Sorted by GeoMean Horizon)")
    ax.set_ylabel("Horizon (seconds, log scale)")
    ax.set_title("Model Horizon across Benchmarks")

    # Improve layout and legend - UPDATED legend title and x-ticks
    plt.xticks(rotation=75, ha='right') # Rotate more if many models
    plt.legend(title='Benchmark', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Ensure output directory exists
    os.makedirs(os.path.dirname(BAR_PLOT_OUTPUT_FILE), exist_ok=True)

    # Save the plot
    plt.savefig(BAR_PLOT_OUTPUT_FILE)
    print(f"Bar plot saved to {BAR_PLOT_OUTPUT_FILE}")
    plt.close() # Close the plot to free memory

def plot_scatter(df):
    """Generates and saves a scatter plot comparing AIME and GPQA horizons."""
    if df.empty:
        print("No data loaded for scatter plot.")
        return

    # Pivot data: models as index, benchmarks as columns
    pivot_df = df.pivot(index='model', columns='benchmark', values='horizon')

    # Check if 'aime' and 'gpqa' columns exist
    required_cols = ['aime', 'gpqa']
    if not all(col in pivot_df.columns for col in required_cols):
        print(f"Warning: Missing required benchmarks for scatter plot ({required_cols}). Skipping scatter plot.")
        return

    # Filter for models present in both and with non-zero horizon
    scatter_data = pivot_df[required_cols].dropna()
    scatter_data = scatter_data[(scatter_data['aime'] > 0) & (scatter_data['gpqa'] > 0)]

    if scatter_data.empty:
        print("No models found with valid horizons for both AIME and GPQA. Skipping scatter plot.")
        return

    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(data=scatter_data, x='aime', y='gpqa')

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel("AIME Horizon (seconds, log scale)")
    ax.set_ylabel("GPQA Horizon (seconds, log scale)")
    ax.set_title("AIME vs GPQA Horizon")

    # Optional: Add text labels for points (can get crowded)
    texts = []
    for i, point in scatter_data.iterrows():
        texts.append(plt.text(point['aime'], point['gpqa'], i, fontsize=8))
    
    # Use adjustText to prevent labels overlapping
    adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SCATTER_PLOT_OUTPUT_FILE), exist_ok=True)

    # Save the plot
    plt.savefig(SCATTER_PLOT_OUTPUT_FILE)
    print(f"Scatter plot saved to {SCATTER_PLOT_OUTPUT_FILE}")
    plt.close()

def main():
    # Load all data initially
    all_df = load_data(DATA_DIR)
    if all_df.empty:
        print("No data loaded. Exiting.")
        return

    # --- Bar Plot --- 
    # Filter and sort for the bar plot
    filtered_df, sorted_models = filter_and_sort_models(all_df.copy()) # Use copy to avoid modifying original
    # Generate and save the bar plot
    plot_horizons(filtered_df, sorted_models)

    # --- Scatter Plot --- 
    # Generate and save the scatter plot using the original data
    plot_scatter(all_df)

if __name__ == "__main__":
    main()

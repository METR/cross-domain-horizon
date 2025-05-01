import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import adjustText # Import adjustText
import matplotlib.ticker as mticker # Import ticker
import matplotlib.dates as mdates # Add date formatting import

# Import wrangle functions and constants
import wrangle

BAR_PLOT_OUTPUT_FILE = 'plots/all_bar.png'
SCATTER_PLOT_OUTPUT_FILE = 'plots/scatter.png' # New output file
LINES_PLOT_OUTPUT_FILE = 'plots/lines_over_time.png' # New output file for lines plot
Y_AXIS_MIN_SECONDS = 60  # 1 minute

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
    scatter_data = scatter_data[(scatter_data['aime'] > 0) & (scatter_data['gpqa'] > 0)].copy()

    if scatter_data.empty:
        print("No models found with valid horizons for both AIME and GPQA. Skipping scatter plot.")
        return
        
    # Apply log transformation manually
    scatter_data['aime_log'] = np.log10(scatter_data['aime'])
    scatter_data['gpqa_log'] = np.log10(scatter_data['gpqa'])

    plt.figure(figsize=(9, 8))

    # Use regplot on the log-transformed data
    ax = sns.regplot(
        data=scatter_data, 
        x='aime_log', 
        y='gpqa_log', 
        scatter_kws={'s': 50, 'alpha': 0.7},
        line_kws={'color': 'orange', 'lw': 2, 'label': 'Trendline (log-log)'}
    )

    # Determine limits for y=x line and axes in log space
    min_log = min(scatter_data['aime_log'].min(), scatter_data['gpqa_log'].min()) - 0.1
    max_log = max(scatter_data['aime_log'].max(), scatter_data['gpqa_log'].max()) + 0.1
    ax.plot([min_log, max_log], [min_log, max_log], color='red', linestyle='--', lw=1, label='y=x')

    # Set limits in log space
    ax.set_xlim(min_log, max_log)
    ax.set_ylim(min_log, max_log)

    # Add labels and title for log axes
    ax.set_xlabel("AIME Horizon (seconds, log scale)") # Clarify log scale
    ax.set_ylabel("GPQA Horizon (seconds, log scale)") # Clarify log scale
    ax.set_title("AIME vs GPQA Horizon (Log-Log Scale)")
    
    # --- Manual Tick Formatting --- 
    # Define potential ticks in original scale (seconds)
    potential_ticks_sec = np.array([1, 10, 30, 60, 180, 300, 600, 1800, 3600, 7200, 10800])
    potential_ticks_labels = ['1s', '10s', '30s', '1m', '3m', '5m', '10m', '30m', '1h', '2h', '3h']
    
    # Filter ticks to be within the plotted range (in log10 space)
    actual_ticks_log = np.log10(potential_ticks_sec)
    valid_ticks_mask = (actual_ticks_log >= min_log) & (actual_ticks_log <= max_log)
    
    tick_positions = actual_ticks_log[valid_ticks_mask]
    tick_labels = [potential_ticks_labels[i] for i, valid in enumerate(valid_ticks_mask) if valid]

    # Set ticks and labels manually
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.minorticks_off() # Turn off minor ticks which might look weird with manual major ticks
    # --- End Manual Tick Formatting --- 

    # Add text labels for points using log coordinates
    texts = []
    for i, point in scatter_data.iterrows():
        texts.append(plt.text(point['aime_log'], point['gpqa_log'], i, fontsize=8))
    
    adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(SCATTER_PLOT_OUTPUT_FILE), exist_ok=True)
    plt.savefig(SCATTER_PLOT_OUTPUT_FILE)
    print(f"Scatter plot saved to {SCATTER_PLOT_OUTPUT_FILE}")
    plt.close()

def plot_lines_over_time(df):
    """Generates and saves a scatter plot of horizon vs. release date with trendlines fitted in log space to frontier models."""
    if df.empty:
        print("No data loaded for lines over time plot.")
        return

    # Ensure 'release_date' exists and has valid dates
    if 'release_date' not in df.columns or df['release_date'].isnull().all():
        print("Warning: No valid 'release_date' data found. Skipping lines over time plot.")
        return

    plot_df = df.dropna(subset=['release_date']).copy()
    plot_df = plot_df[plot_df['horizon'] > 0] # Log scale needs positive values

    if plot_df.empty:
        print("No valid data points (with release date and positive horizon) found for lines over time plot.")
        return

    # Convert horizon to minutes and add log horizon
    plot_df['horizon_minutes'] = plot_df['horizon'] / 60.0
    plot_df['log_horizon_minutes'] = np.log10(plot_df['horizon_minutes'])

    plot_df['release_date_num'] = mdates.date2num(plot_df['release_date'])

    # Identify frontier models for each benchmark
    plot_df['is_frontier'] = False
    benchmarks = plot_df['benchmark'].unique()
    for bench in benchmarks:
        bench_df = plot_df[plot_df['benchmark'] == bench].sort_values(by=['release_date_num', 'horizon_minutes'], ascending=[True, False])
        max_horizon_so_far = -np.inf
        frontier_indices = []
        for index, row in bench_df.iterrows():
            # A model is on the frontier if its horizon is greater than all previous models' horizons
            if row['horizon_minutes'] > max_horizon_so_far:
                 frontier_indices.append(index)
                 max_horizon_so_far = row['horizon_minutes']
        if frontier_indices:
            plot_df.loc[frontier_indices, 'is_frontier'] = True

    fig, ax = plt.subplots(figsize=(14, 8))

    palette = sns.color_palette(n_colors=len(benchmarks))
    benchmark_colors = {bench: color for bench, color in zip(benchmarks, palette)}

    for bench in benchmarks:
        bench_data = plot_df[plot_df['benchmark'] == bench]
        color = benchmark_colors[bench]
        frontier_data = bench_data[bench_data['is_frontier']]
        non_frontier_data = bench_data[~bench_data['is_frontier']]

        # Plot non-frontier points (circles)
        ax.scatter(
            non_frontier_data['release_date'], # Use datetime objects for x-axis plotting
            non_frontier_data['horizon_minutes'],
            color=color,
            marker='o',
            label=f"{bench}", # Main label for legend
            alpha=0.6,
            s=50,
            edgecolor='k', # Add edge color for better visibility
            linewidth=0.5
        )

        # Plot frontier points (diamonds)
        ax.scatter(
            frontier_data['release_date'],
            frontier_data['horizon_minutes'],
            color=color,
            marker='D',
            label=f"_{bench}_frontier", # Hidden label for legend
            alpha=0.9,
            s=70,
            edgecolor='k',
            linewidth=0.5
        )


        # Fit and plot regression line using only frontier points
        if len(frontier_data) >= 2:
            # Perform linear regression on numerical date vs log horizon
            X = frontier_data['release_date_num'].values
            Y_log = frontier_data['log_horizon_minutes'].values

            coeffs = np.polyfit(X, Y_log, 1)
            poly = np.poly1d(coeffs)

            x_line_num = np.linspace(X.min(), X.max(), 100)
            y_line_log = poly(x_line_num)

            x_line_date = mdates.num2date(x_line_num)
            y_line = 10**y_line_log

            ax.plot(x_line_date, y_line, color=color, linestyle='--', linewidth=2, label=f"_{bench}_trend")

    ax.set_yscale('log')

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("Horizon (minutes, log scale)")
    ax.set_title("Model Horizon vs. Release Date (Log Scale, Trend on Frontier)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45, ha='right') # Auto-format dates (includes rotation)

    # Create a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Benchmark', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    os.makedirs(os.path.dirname(LINES_PLOT_OUTPUT_FILE), exist_ok=True)

    plt.savefig(LINES_PLOT_OUTPUT_FILE)
    print(f"Lines over time plot saved to {LINES_PLOT_OUTPUT_FILE}")
    plt.close(fig)

def main():
    # Load all data initially using wrangle
    all_df = wrangle.load_data(wrangle.DATA_DIR)
    if all_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Write raw data to CSV
    pivot_df = all_df.pivot(index='model', columns='benchmark', values='horizon')
    pivot_df.reset_index(inplace=True)
    os.makedirs(os.path.dirname('data/all_data.csv'), exist_ok=True)
    pivot_df.to_csv('data/all_data.csv', index=False)
    print("Raw data written to data/all_data.csv")

    # --- Bar Plot --- 
    # Filter and sort for the bar plot using wrangle
    filtered_df, sorted_models = wrangle.filter_and_sort_models(all_df.copy()) # Use copy to avoid modifying original
    # Generate and save the bar plot
    plot_horizons(filtered_df, sorted_models)

    # --- Scatter Plot --- 
    # Generate and save the scatter plot using the original loaded data
    plot_scatter(all_df.copy()) # Use copy

    # --- Lines Over Time Plot ---
    # Generate and save the lines over time plot using the original loaded data
    plot_lines_over_time(all_df.copy()) # Use copy

if __name__ == "__main__":
    main()

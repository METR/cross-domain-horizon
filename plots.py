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

    texts = [] # Initialize list to store text objects for adjustText
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
            alpha=0.2,
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

            # Calculate doubling rate (slope in doublings per year)
            # The slope of log10(y) vs. x gives us log10(2) per unit x
            # To convert to doublings per year, divide by log10(2)
            doubling_rate = coeffs[0] * 365 / np.log10(2)  # Convert from per day to per year
            
            # Add text with doubling rate near the middle of the line
            mid_point_idx = len(x_line_num) // 2
            mid_x_num = x_line_num[mid_point_idx]
            mid_y = 10**y_line_log[mid_point_idx]
            
            # Format the doubling rate with appropriate sign
            if doubling_rate > 0:
                rate_text = f"+{doubling_rate:.1f} doublings/year"
            else:
                rate_text = f"{doubling_rate:.1f} doublings/year"
                
            # Convert numerical date to datetime for text positioning
            mid_x = mdates.num2date(mid_x_num)
            
            ax.text(mid_x, mid_y * 1.1, rate_text, fontsize=8, color=color, 
                   ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, pad=2))

            x_line_date = mdates.num2date(x_line_num)
            y_line = 10**y_line_log

            ax.plot(x_line_date, y_line, color=color, linestyle='--', linewidth=2, label=f"_{bench}_trend")

            # Add text labels for first and last frontier points
            if not frontier_data.empty:
                # Sort frontier_data by release date to ensure first and last are correct
                frontier_data_sorted = frontier_data.sort_values('release_date')
                first_frontier_point = frontier_data_sorted.iloc[0]
                last_frontier_point = frontier_data_sorted.iloc[-1]

                texts.append(ax.text(first_frontier_point['release_date'],
                                     first_frontier_point['horizon_minutes'],
                                     first_frontier_point['model'],
                                     fontsize=8, color=color))
                # Only add last point label if it's different from the first
                if len(frontier_data_sorted) > 1:
                    texts.append(ax.text(last_frontier_point['release_date'],
                                         last_frontier_point['horizon_minutes'],
                                         last_frontier_point['model'],
                                         fontsize=8, color=color))

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

    # Adjust text to prevent overlap
    if texts:
        adjustText.adjust_text(texts, ax=ax,
                               arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

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
    # Get release dates for each model (assuming one release date per model)
    release_dates = all_df[['model', 'release_date']].drop_duplicates().set_index('model')
    
    # Create pivot table for horizons
    pivot_df = all_df.pivot(index='model', columns='benchmark', values='horizon')
    
    # Join with release dates
    pivot_df = pivot_df.join(release_dates)
    
    pivot_df.reset_index(inplace=True)
    os.makedirs(os.path.dirname('data/all_data.csv'), exist_ok=True)
    pivot_df.to_csv('data/all_data.csv', index=False)
    print("Raw data written to data/all_data.csv")

    # --- Bar Plot --- 
    # Filter and sort for the bar plot using wrangle
    filtered_df, sorted_models = wrangle.filter_and_sort_models(all_df.copy()) # Use copy to avoid modifying original
    # Generate and save the bar plot
    plot_horizons(filtered_df, sorted_models)

    # --- Lines Over Time Plot ---
    # Generate and save the lines over time plot using the original loaded data
    plot_lines_over_time(all_df.copy()) # Use copy

if __name__ == "__main__":
    main()

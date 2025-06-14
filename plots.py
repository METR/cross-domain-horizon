import adjustText
import argparse
import glob
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import matplotlib.patches as mpatches
import pathlib
import matplotlib.patheffects as patheffects
import seaborn as sns
import toml
from scipy.interpolate import make_splrep
from enum import Enum
from dataclasses import dataclass, field
from functools import total_ordering

from plotting_aliases import benchmark_aliases, plotting_aliases
from plot_splits import plot_splits
from plot_speculation import plot_speculation

BENCHMARKS_PATH = 'data/benchmarks'

SCATTER_PLOT_OUTPUT_FILE = 'plots/scatter.png'
LINES_PLOT_OUTPUT_FILE = 'plots/lines_over_time.png'
LINES_SUBPLOTS_OUTPUT_FILE = 'plots/lines_over_time_subplots.png'
BENCHMARK_TASK_LENGTHS_OUTPUT_FILE = 'plots/benchmark_task_lengths.png'
LENGTH_DEPENDENCE_OUTPUT_FILE = 'plots/length_dependence.png'
SPLITS_OUTPUT_FILE = 'plots/splits_plot.png'
SPECULATION_OUTPUT_FILE = pathlib.Path('plots/speculation.png')
Y_AXIS_MIN_SECONDS = 60  # 1 minute


@total_ordering
class ShowPointsLevel(Enum):
    NONE = 0
    FIRST_AND_LAST = 1
    FRONTIER = 2
    ALL = 3
    def __lt__(self, other):
        return self.value < other.value

@dataclass
class LinesPlotParams:
    show_points_level: ShowPointsLevel
    show_benchmarks: list[str] = field(default_factory=list)
    hide_benchmarks: list[str] = field(default_factory=list)
    show_doubling_rate: bool = False
    subplots: bool = False
    title: str = "Time Horizon vs. Release Date (Log Scale, Trend on Frontier)"
    verbose: bool = False

def add_watermark(ax=None, text="DRAFT\nDO NOT HYPE", alpha=0.25):
    """Add a watermark to the current plot or specified axes."""
    if ax is None:
        ax = plt.gca()
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes, 
            fontsize=80, color='gray', alpha=alpha,
            ha='center', va='center', rotation=45, zorder=0)

def get_benchmark_data(benchmarks_path) -> dict[str, list[float]]:
    """
    Loads benchmark data from a folder of TOML files, reads the "lengths" key from each file, and returns a dictionary of benchmark name to list of lengths.
    """
    benchmark_data = {}
    for file in os.listdir(benchmarks_path):
        if file.endswith('.toml'):
            with open(os.path.join(benchmarks_path, file), 'r') as f:
                benchmark_data[file.replace('.toml', '')] = toml.load(f)


    result = pd.DataFrame([
        {'length': length, 'benchmark': benchmark, 'length_type': data.get('length_type', "default")}
        for benchmark, data in benchmark_data.items()
        for split_name, split_data in data['splits'].items() if split_name != "all" or len(data['splits']) == 1
        for length in split_data['lengths']
    ])
    return result


def plot_lines_over_time(df, output_file,
                         benchmark_data: pd.DataFrame,
                         params: LinesPlotParams):
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
    plot_df['log2_horizon_minutes'] = np.log2(plot_df['horizon_minutes'])

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

    if params.subplots:
        # last subplot contains legend
        fig, axs = plt.subplots(figsize=(12, 8), nrows=(len(benchmarks) + 1) // 3, ncols=3, sharex=True, sharey=True)
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    palette = sns.color_palette(n_colors=len(benchmarks))
    benchmark_colors = {bench: color for bench, color in zip(benchmarks, palette)}

    texts = [] # Initialize list to store text objects for adjustText

    if params.hide_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench not in params.hide_benchmarks]
    if params.show_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench in params.show_benchmarks]

    densely_dotted = (0, (1, 1))
    for bench in benchmarks:
        if params.subplots:
            ax = axs[benchmarks.index(bench)]
            ax.set_title(benchmark_aliases[bench], fontsize=10)

        bench_data = plot_df[plot_df['benchmark'] == bench]
        color = benchmark_colors[bench]
        frontier_data = bench_data[bench_data['is_frontier']]
        non_frontier_data = bench_data[~bench_data['is_frontier']]

        length_data = benchmark_data[benchmark_data['benchmark'] == bench]['length']
        p2 = length_data.quantile(0.02)
        p98 = length_data.quantile(0.98)
        if pd.isna(p2) or pd.isna(p98):
            p2 = 0
            p98 = 10**10

        def scatter_points(data, label, **kwargs):
            if 'slope' in data.columns and len(data) > 0:
                # Plot each point individually with its own marker based on slope
                for idx, row in data.iterrows():
                    slope_val = row['slope']
                    if pd.isna(slope_val):
                        marker = 'o'  # Keep as circle if slope is NaN
                    else:
                        marker = 'x' if slope_val < 0.3 else 'o'
                    
                    # Only add label to the first point to avoid duplicate legend entries
                    point_label = label if idx == data.index[0] else None
                    
                    ax.scatter(
                        row['release_date'],
                        row['horizon_minutes'],
                        color=color,
                        label=point_label,
                        marker=marker,
                        **kwargs
                    )
            else:
                # Default to circles for all points
                ax.scatter(
                    data['release_date'],
                    data['horizon_minutes'],
                    color=color,
                    label=label,
                    marker='o',
                    **kwargs
                )

        # Plot non-frontier points
        if params.show_points_level >= ShowPointsLevel.ALL:
            scatter_points(non_frontier_data, f"_{bench}_nonfrontier", alpha=0.2, s=20, edgecolor='k', linewidth=0.5)

        if params.verbose:
            print(f"Frontier models for {bench}: {', '.join(frontier_data['model'].unique())}")
        df_within = frontier_data[(frontier_data['horizon_minutes'] > p2) & (frontier_data['horizon_minutes'] < p98)]
        df_outside = frontier_data[(frontier_data['horizon_minutes'] > p98) | (frontier_data['horizon_minutes'] < p2)]

        # Plot frontier points (slope-based markers for both within and outside range)
        if params.show_points_level >= ShowPointsLevel.FRONTIER:
            scatter_points(df_within, f"_{bench}", alpha=0.9, s=12, edgecolor='k', linewidth=0.5)
            scatter_points(df_outside, f"_{bench}_outside", alpha=0.9, s=15, linewidth=0.5)
        else:
            frontier_data = frontier_data.sort_values('release_date_num')
            selected_data = frontier_data.iloc[[0, -1]] if params.show_points_level == ShowPointsLevel.FIRST_AND_LAST else frontier_data.iloc[[]]
            scatter_points(selected_data, f"_{bench}", alpha=0.9, s=30, edgecolor='k', linewidth=0.5)

        # Fit and plot smoothing spline using only frontier points
        if len(frontier_data) >= 3:  # Need at least 3 points for spline
            # Sort frontier data by release date for spline fitting
            frontier_sorted = frontier_data.sort_values('release_date_num')
            X = frontier_sorted['release_date_num'].values
            Y_log = frontier_sorted['log2_horizon_minutes'].values

            # Keep linear regression for doubling rate calculation
            coeffs = np.polyfit(X, Y_log, 1)
            doubling_rate = coeffs[0] * 365  # Convert from per day to per year
            slope = coeffs[0]  # Store raw slope for marker selection

            # degree-1 spline to avoid negative slopes
            spline = make_splrep(X, Y_log, s=0.2, k=1)

            x_line_num = np.linspace(X.min(), X.max(), 100)
            y_line_log = spline(x_line_num)
            
            # Add text with doubling rate near the middle of the line
            mid_point_idx = len(x_line_num) // 2
            mid_x_num = x_line_num[mid_point_idx]
            mid_y = 2**y_line_log[mid_point_idx]
            mid_x = mdates.num2date(mid_x_num)
            
            if params.show_doubling_rate:
                rate_text = f"{doubling_rate:.1f} dbl./yr"
                ax.text(mid_x, mid_y * 1.1, rate_text, fontsize=10, color=color, 
                        ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, pad=2))

            x_line_date = np.array(mdates.num2date(x_line_num))
            y_line = 2.0**y_line_log

            # Split the line into three segments based on p2 and p98
            mask_within = (y_line >= p2) & (y_line <= p98)
            mask_above = y_line > p98
            mask_below = y_line < p2

            thick = (bench == "hcast_r_s") and not params.subplots

            ax.plot(x_line_date[mask_within], y_line[mask_within], color=color, linestyle='-', linewidth=5 if thick else 2.5, label=f"{benchmark_aliases[bench]}", zorder=100 if thick else None)
            ax.plot(x_line_date[mask_above], y_line[mask_above], color=color, alpha=0.3, linestyle=densely_dotted, linewidth=2)
            ax.plot(x_line_date[mask_below], y_line[mask_below], color=color, alpha=0.3, linestyle=densely_dotted, linewidth=2)

            # Add text labels for first and last frontier points
            if not frontier_data.empty and params.show_points_level >= ShowPointsLevel.FIRST_AND_LAST:
                # Sort frontier_data by release date to ensure first and last are correct
                frontier_data_sorted = frontier_data.sort_values('release_date')
                first_frontier_point = frontier_data_sorted.iloc[0]
                last_frontier_point = frontier_data_sorted.iloc[-1]

                def text_label(point):
                    model_name = point['model']
                    if model_name in plotting_aliases:
                        model_name = plotting_aliases[model_name]
                    else:
                        model_name = point['model']
                    texts.append(ax.text(point['release_date'],
                                         point['horizon_minutes'],
                                         model_name,
                                         fontsize=10, color=color))

                text_label(first_frontier_point)
                text_label(last_frontier_point)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x}'))
    plt.ylim(0.05, 10000)

    if params.subplots:
        fig.suptitle(params.title)
        fig.supxlabel("Model Release Date")
        fig.supylabel("Time Horizon (minutes)")
    else:
        plt.xlabel("Model Release Date")
        plt.ylabel("Time Horizon (minutes)")
        plt.title(params.title)
        ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)



    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45, ha='right') # Auto-format dates (includes rotation)


    # Create a legend
    handles, labels = ax.get_legend_handles_labels()

    trend_legend_handles = []
    handle1, = ax.plot([], [], color='black', linestyle='-', linewidth=2, label="Inside range")
    trend_legend_handles.append(handle1)
    handle2, = ax.plot([], [], color='black', linestyle=densely_dotted, alpha=0.4, linewidth=2, label="Outside range")
    trend_legend_handles.append(handle2)

    trend_legend_ax = axs[-1] if params.subplots else ax

    trend_legend = trend_legend_ax.legend(handles=trend_legend_handles, title="Range of task lengths\n in benchmark", bbox_to_anchor=(0.02, 0.5), loc='center left')

    
    if not params.subplots:
        bench_legend = ax.legend(handles=handles, labels=labels, title='Benchmark', bbox_to_anchor=(0.02, 1), loc='upper left')
        ax.add_artist(trend_legend)

        if texts:
            adjustText.adjust_text(texts, ax=ax,
                                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    add_watermark(ax)

    if params.subplots:
        for i in range(len(benchmarks), len(axs)):
            axs[i].set_axis_off()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.savefig(output_file)
    print(f"Lines over time plot saved to {output_file}")
    plt.close(fig)


def plot_benchmarks(df: pd.DataFrame, benchmark_data: dict[str, list[float]], output_file: pathlib.Path):
    """
    df is a dataframe holding horizon data for all models on all benchmarks.

    TODO this should have multiple boxplots "within" each scatter plot, one for each split.
    """

    
    length_to_color_map = {
        "baseline": "royalblue",
        "estimate": "darkred",
        "default": "grey"
    }

    # Create a DataFrame for seaborn
    lengths_df = benchmark_data.copy()
    lengths_df['length_type'] = pd.Categorical(lengths_df['length_type'], categories=length_to_color_map.keys(), ordered=True)

    benchmarks = lengths_df['benchmark'].unique().tolist()
    lengths_df.sort_values(by='benchmark', inplace=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=lengths_df, y='length', x='benchmark', whis=(10, 90),
                showfliers=False, width=0.3, fill=False, color='black', zorder=2, linewidth=2)
    sns.stripplot(data=lengths_df, y='length', x='benchmark', size=3, hue='length_type', zorder=1, alpha=0.3,
                  palette=length_to_color_map.values(), legend=False)

    # plot a diamond for the frontier (max horizon) model on each benchmark
    s_frontier = df.groupby('benchmark', as_index=True)["horizon"].max()
    s_frontier /= 60 # Convert to minutes

    s_frontier = s_frontier[s_frontier.index.isin(benchmarks)]
    
    kwargs = {"label": f"Frontier (max horizon)"}
    for benchmark, horizon in s_frontier.items():
        plt.scatter(benchmark, horizon, color='white', edgecolor='black', marker='*', s=130, zorder=3, **kwargs)
        kwargs = {}

    # Legend
    plt.plot([], [], color='black', linewidth=2, label="Quantiles (10/25/50/75/90%)")
    plt.scatter([], [], color='royalblue', marker='o', s=20, label="Individual task")
    plt.scatter([], [], color='darkred', marker='o', s=20, label="Individual task (estimated)")
    plt.legend()

    plt.grid(True, which="major", axis="y",ls="--", linewidth=0.5, alpha=0.4)

    plt.yscale('log')
    plt.ylabel('Length (minutes)')
    plt.xlabel('Benchmark')
    plt.title('Task Lengths By Benchmark')

    
    # Replace x-axis tick labels with benchmark aliases
    ax = plt.gca()
    tick_labels = ax.get_xticklabels()
    new_labels = [benchmark_aliases.get(label.get_text(), label.get_text()) for label in tick_labels]
    ax.set_xticklabels(new_labels)

    
    
    add_watermark()
    
    plt.savefig(output_file)
    print(f"Benchmark lengths plot saved to {output_file}")


def plot_length_dependence(df: pd.DataFrame, output_file: pathlib.Path):
    """
    Plots the length dependence of the horizon.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    add_watermark(ax)

    benchmarks_to_use = ["hcast_r_s_full_method", "video_mme", "livecodebench_2411_2505", "gpqa_diamond", ]

    df_to_use = df[df['benchmark'].isin(benchmarks_to_use)]

    arr = mpatches.FancyArrowPatch((0.2, 0.05), (0.8, 0.05),
                               arrowstyle='->,head_width=.15', mutation_scale=20,
                               transform=ax.transAxes)
    ax.add_patch(arr)
    ax.annotate("Model success depends MORE on human task length", (.5, 1), xycoords=arr, ha='center', va='bottom', fontsize=12)


    sns.scatterplot(data=df_to_use, y='horizon', x='slope', hue='benchmark', ax=ax)

    ax.set_xlabel("Slope of logistic curve")
    ax.set_ylabel("Model horizon (minutes)")

    ax.set_xlim(0.08, 4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.1)
    ax.set_title("Length-dependence of each benchmark (each point is a model)")
    ax.legend()

    plt.savefig(output_file)
    print(f"Length dependence plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate various plots for model analysis')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--lines', action='store_true', help='Generate lines over time plot')
    parser.add_argument('--hcast', action='store_true', help='Generate hcast comparison plot')
    parser.add_argument('--lengths', action='store_true', help='Generate benchmark lengths plot')
    parser.add_argument('--length-dependence', action='store_true', help='Generate length dependence plot')
    parser.add_argument('--speculation', action='store_true', help='Generate speculation plot')
    parser.add_argument('--splits', action='store_true', help='Generate splits plot')
    args = parser.parse_args()

    plots_to_make = []
    if args.all or not any(vars(args).values()):
        plots_to_make = ["lines", "hcast", "lengths", "length_dependence"]
    elif args.lines:
        plots_to_make += ["lines"]
    elif args.hcast:
        plots_to_make += ["hcast"]
    elif args.lengths:
        plots_to_make += ["lengths"]
    elif args.length_dependence:
        plots_to_make += ["length_dependence"]
    elif args.speculation:
        plots_to_make += ["speculation"]
    elif args.splits:
        plots_to_make += ["splits"]

    # If no arguments provided, default to --all
    if not any(vars(args).values()):
        args.all = True

    # Load all data from CSV file written by wrangle.py
    data_file = 'data/processed/all_data.csv'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run wrangle.py first.")
        return
    
    all_df = pd.read_csv(data_file)
    # Convert release_date back to datetime
    all_df['release_date'] = pd.to_datetime(all_df['release_date']).dt.date
    
    if all_df.empty:
        print("No data loaded. Exiting.")
        return

    benchmark_data = get_benchmark_data(BENCHMARKS_PATH)


    # --- Lines Over Time Plot ---
    if "lines" in plots_to_make:
        # Generate and save the lines over time plot using the original loaded data
        plot_lines_over_time(all_df.copy(), LINES_PLOT_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s_full_method"], show_points_level=ShowPointsLevel.FRONTIER, verbose=True))

        plot_lines_over_time(all_df.copy(), "plots/hcast_comparison.png", benchmark_data, LinesPlotParams(
            title="HCAST/RS Time Horizons (full method vs average-scores-only)",
            show_benchmarks=["hcast_r_s", "hcast_r_s_full_method"], show_points_level=ShowPointsLevel.FRONTIER,)
        )
        plot_lines_over_time(all_df.copy(), LINES_SUBPLOTS_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s_full_method"], show_points_level=ShowPointsLevel.ALL, subplots=True, show_doubling_rate=True))


    # --- Benchmark Task Lengths Plot ---
    if "lengths" in plots_to_make:
        plot_benchmarks(all_df, benchmark_data,BENCHMARK_TASK_LENGTHS_OUTPUT_FILE)

    # --- Length Dependence Plot ---
    if "length_dependence" in plots_to_make:
        plot_length_dependence(all_df, LENGTH_DEPENDENCE_OUTPUT_FILE)

    if "splits" in plots_to_make:
        plot_splits(all_df, SPLITS_OUTPUT_FILE)

    if "speculation" in plots_to_make:
        plot_speculation(all_df, SPECULATION_OUTPUT_FILE)

if __name__ == "__main__":
    main()

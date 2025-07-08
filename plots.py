import adjustText
import argparse
import glob
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
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
import dataclasses
from datetime import date

from plotting_aliases import benchmark_aliases, plotting_aliases, benchmark_colors
from plot_splits import plot_splits
from plot_speculation import plot_speculation

BENCHMARKS_PATH = 'data/benchmarks'

SCATTER_PLOT_OUTPUT_FILE = 'plots/scatter.png'
HEADLINE_PLOT_OUTPUT_FILE = 'plots/headline.png'
LINES_PLOT_OUTPUT_FILE = 'plots/lines_over_time.png'
LINES_SUBPLOTS_OUTPUT_FILE = 'plots/lines_over_time_subplots.png'
BENCHMARK_TASK_LENGTHS_OUTPUT_FILE = 'plots/benchmark_task_lengths.png'
LENGTH_DEPENDENCE_OUTPUT_FILE = 'plots/length_dependence.png'
SPLITS_OUTPUT_FILE = 'plots/splits_plot.png'
SPECULATION_OUTPUT_FILE = pathlib.Path('plots/speculation.png')
PERCENT_OVER_TIME_OUTPUT_FILE = 'plots/percent_over_time.png'
BETA_SWARMPLOT_OUTPUT_FILE = 'plots/beta_swarmplot.png'
Y_AXIS_MIN_SECONDS = 60  # 1 minute
WATERMARK = False
ROBUSTNESS_SUBPLOTS_OUTPUT_FILE = 'plots/robustness_subplots.png'


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
    show_dotted_lines: bool = True
    show_doubling_rate: bool = False
    subplots: bool = False
    title: str = "Time Horizon vs. Release Date (Log Scale, Trend on Frontier)"
    verbose: bool = False
    xbound: tuple[str, str] | None = None
    ybound: tuple[float, float] | None = None
    date_cutoff: date | None = date(2023, 1, 1)
    skip_labels: bool = False  # New field to skip title and axis labels
    skip_legend: bool = False  # New field to skip legend creation
    rotate_dates: bool = True  # New field to control date rotation

def add_watermark(ax=None, text="DRAFT\nDO NOT HYPE", alpha=0.35):
    """Add a watermark to the current plot or specified axes."""
    if not WATERMARK:
        return
    if ax is None:
        ax = plt.gca()
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes, 
            fontsize=60, color='gray', alpha=alpha,
            ha='center', va='center', rotation=45, zorder=1000)

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
                         params: LinesPlotParams,
                         ax: plt.Axes | None = None):
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
    plot_df = plot_df[(plot_df['slope'].isna()) | (plot_df['slope'] > 0.25)]

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
        nrows = (len(benchmarks) + 1) // 4
        fig, axs = plt.subplots(figsize=(12, nrows * 4), nrows=nrows, ncols=4, sharex=True, sharey=True)
        nrows = (len(benchmarks) + 1) // 4
        fig, axs = plt.subplots(figsize=(12, nrows * 4), nrows=nrows, ncols=4, sharex=True, sharey=True)
        axs = axs.flatten()
        texts_per_ax = {ax: [] for ax in axs}  # Track texts per axis
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.figure

    texts = [] # Initialize list to store text objects for adjustText
    points = [] # Track actual data point coordinates for adjustText
    subplot_texts = {} # Store texts for each subplot separately
    subplot_points = {} # Store points for each subplot separately

    if params.hide_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench not in params.hide_benchmarks]
    if params.show_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench in params.show_benchmarks]

    # Sort benchmarks by has_slope_06 for legend purposes
    # Calculate has_slope_06 for each benchmark first
    benchmark_slope_info = []
    for bench in benchmarks:
        bench_data = plot_df[plot_df['benchmark'] == bench]
        has_placeholder_slope = bench_data['slope'].eq(0.6).any()
        has_placeholder_slope |= bench_data['slope'].isna().all()
        benchmark_slope_info.append((bench, has_placeholder_slope))
    
    # Sort by has_slope_06 (False first)
    benchmark_slope_info.sort(key=lambda x: (x[1], x[0]))  # not x[1] puts True first
    benchmarks = [bench for bench, _ in benchmark_slope_info]

    densely_dotted = (0, (1, 1))
    BASE_LINE_WIDTH = 2
    THICK_LINE_WIDTH = BASE_LINE_WIDTH + 1.0
    for bench in benchmarks:
        if params.subplots:
            ax = axs[benchmarks.index(bench)]
            ax.set_title(benchmark_aliases[bench], fontsize=10)
            subplot_texts[bench] = []  # Initialize text list for this subplot
            subplot_points[bench] = []  # Initialize points list for this subplot

        bench_data = plot_df[plot_df['benchmark'] == bench]
        color = benchmark_colors[bench]
        frontier_data = bench_data[bench_data['is_frontier']]
        non_frontier_data = bench_data[~bench_data['is_frontier']]

        length_data = benchmark_data[benchmark_data['benchmark'] == bench]['length']
        has_placeholder_slope = benchmark_slope_info[benchmarks.index(bench)][1]
        p2 = length_data.quantile(0.02)
        p98 = length_data.quantile(0.98)
        if pd.isna(p2) or pd.isna(p98):
            p2 = 0
            p98 = 10**10

        def scatter_points(data, label, **kwargs):
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
            scatter_points(non_frontier_data, f"_{bench}_nonfrontier", alpha=0.2, s=20,  linewidth=0.5)

        if params.verbose:
            print(f"Frontier models for {bench}: {', '.join(frontier_data['model'].unique())}")
        df_within = frontier_data[(frontier_data['horizon_minutes'] > p2) & (frontier_data['horizon_minutes'] < p98)]
        df_outside = frontier_data[(frontier_data['horizon_minutes'] > p98) | (frontier_data['horizon_minutes'] < p2)]

        # Plot frontier points (slope-based markers for both within and outside range)
        if params.show_points_level >= ShowPointsLevel.FRONTIER:
            scatter_points(df_within, f"_{bench}", alpha=0.9, s=17, linewidth=0.5)
            scatter_points(df_outside, f"_{bench}_outside", alpha=0.9, s=17, linewidth=0.5)
        else:
            frontier_data = frontier_data.sort_values('release_date_num')
            selected_data = frontier_data.iloc[[0, -1]] if params.show_points_level == ShowPointsLevel.FIRST_AND_LAST else frontier_data.iloc[[]]
            scatter_points(selected_data, f"_{bench}", alpha=0.9, s=30, linewidth=0.5)

        if len(frontier_data) >= 3:  # Need at least 3 points for spline
            frontier_sorted = frontier_data.sort_values('release_date_num')
            X = frontier_sorted['release_date_num'].values
            Y_log = frontier_sorted['log2_horizon_minutes'].values

            # Keep linear regression for doubling rate calculation
            coeffs = np.polyfit(X, Y_log, 1)
            doubling_rate_per_year = coeffs[0] * 365  # Convert from per day to per year
            months_per_doubling = 12 / doubling_rate_per_year
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
                rate_text = f"Doubling time:\n{months_per_doubling:.1f} months"
                ax.text(0.02, 0.99, rate_text, fontsize=14, color=color, 
                        ha='left', va='top', transform=ax.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))

            x_line_date = np.array(mdates.num2date(x_line_num))
            y_line = 2.0**y_line_log

            # Split the line into three segments based on p2 and p98
            mask_within = (y_line >= p2) & (y_line <= p98)
            mask_above = y_line > p98
            mask_below = y_line < p2

            is_hrs = (bench == "hcast_r_s_full_method") and not params.subplots

            # Render all main trend lines as solid. Increase thickness slightly when
            # the benchmark *does not* include any model with β = 0.6 (previously
            # indicated by "triangles").
            line_width = THICK_LINE_WIDTH if not has_placeholder_slope else BASE_LINE_WIDTH

            ax.plot(
                x_line_date[mask_within],
                y_line[mask_within],
                color=color,
                linestyle='-',
                linewidth=line_width,
                label=f"{benchmark_aliases[bench]}",
                zorder=100 if is_hrs else None
            )
            if params.show_dotted_lines:
                ax.plot(x_line_date[mask_above], y_line[mask_above], color=color, alpha=0.3, linestyle=densely_dotted, linewidth=2)
                ax.plot(x_line_date[mask_below], y_line[mask_below], color=color, alpha=0.3, linestyle=densely_dotted, linewidth=2)

            # Add text labels for first and last frontier points
            if not frontier_data.empty and params.show_points_level >= ShowPointsLevel.FIRST_AND_LAST:
                # Sort frontier_data by release date to ensure first and last are correct
                frontier_data_sorted = frontier_data.sort_values('release_date')
                first_frontier_point = frontier_data_sorted.iloc[0]
                last_frontier_point = frontier_data_sorted.iloc[-1]

                def text_label(point, current_ax=None):
                    model_name = point['model']
                    if model_name in plotting_aliases:
                        model_name = plotting_aliases[model_name]
                    else:
                        model_name = point['model']
                    if point['horizon_minutes'] < 0.1:  
                        y_offset = max(point['horizon_minutes'] * 1.5, 0.06)
                        va = 'bottom'
                    else:
                        # More offset for robustness subplots to reduce overlap
                        is_robustness = ax is not None and not params.subplots
                        offset_factor = 1.3 if is_robustness else 1.1
                        y_offset = point['horizon_minutes'] * offset_factor
                        va = 'bottom'
                    
                    # Store actual data point coordinates
                    point_x = point['release_date']
                    point_y = point['horizon_minutes']
                    
                    # Smaller font and no bbox for robustness subplots
                    is_robustness = ax is not None and not params.subplots
                    text_obj = ax.text(point['release_date'],
                                      y_offset,
                                      model_name,
                                      fontsize=8 if is_robustness else 10,
                                      color=color,
                                      ha='center',
                                      va=va,
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                    if params.subplots:
                        subplot_texts[bench].append(text_obj)
                        subplot_points[bench].append((mdates.date2num(point_x), point_y))
                    else:
                        texts.append(text_obj)
                        points.append((mdates.date2num(point_x), point_y))

                # Hide the *first* frontier label if it is for a model released *after* the
                # specified cutoff (helps reduce clutter for very recent models). Allow callers
                # to disable the cutoff by passing `date_cutoff=None`.
                DATE_CUTOFF = params.date_cutoff
                if params.subplots or DATE_CUTOFF is None or first_frontier_point['release_date'] <= DATE_CUTOFF:
                    text_label(first_frontier_point, current_ax=ax)
                text_label(last_frontier_point, current_ax=ax)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x}'))


    if params.subplots:
        fig.suptitle(params.title)
        fig.supxlabel("Model Release Date")
        fig.supylabel("50% Time Horizon (minutes)")
    elif not params.skip_labels:
        plt.xlabel("Model Release Date")
        plt.ylabel("50% Time Horizon (minutes)")
        plt.title(params.title)
        ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)
    else:
        ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)



    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    if params.rotate_dates:
        fig.autofmt_xdate(rotation=45, ha='right') # Auto-format dates (includes rotation)
    
    # Set x-axis bounds if specified
    if params.xbound is not None:
        start_date, end_date = params.xbound
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    if params.ybound is not None:
        ax.set_ylim(params.ybound)
    else:
        plt.ylim(0.05, 3000)


    # Create a legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Add section headings for known and unknown beta
    if not params.subplots:
        # Count how many benchmarks have thick lines (not has_slope_06)
        num_unknown_beta = sum(1 for _, has_slope_06 in benchmark_slope_info if not has_slope_06)
        num_known_beta = sum(1 for _, has_slope_06 in benchmark_slope_info if has_slope_06)
        
        # Add headings and reorganize
        if num_unknown_beta > 0 and num_known_beta > 0:
            # Insert "Known β" heading at the beginning
            heading_handle = Line2D([0], [0], color='none')
            handles.insert(0, heading_handle)
            labels.insert(0, '**Known β**')

            # Add divider between known and unknown beta
            divider_handle = Line2D([0], [0], color='none')
            # Insert "Unknown β" heading after known beta section
            transition_index = num_known_beta + 1  # +1 for the "Known β" heading
            heading_handle2 = Line2D([0], [0], color='none')
            handles.insert(transition_index, divider_handle)
            labels.insert(transition_index, '─────────────')
            handles.insert(transition_index + 1, heading_handle2)
            labels.insert(transition_index + 1, '**Unknown β**')
        elif num_unknown_beta > 0:
            # Only unknown beta benchmarks
            heading_handle = Line2D([0], [0], color='none')
            handles.insert(0, heading_handle)
            labels.insert(0, '**Unknown β**')
        elif num_known_beta > 0:
            # Only known beta benchmarks
            heading_handle = Line2D([0], [0], color='none')
            handles.insert(0, heading_handle)
            labels.insert(0, '**Known β**')

    # Add divider before extrapolated line
    divider_handle = Line2D([0], [0], color='none')
    handles.append(divider_handle)
    labels.append('─────────────')

    # Add extrapolated line
    handle4, = ax.plot([], [], color='black', linestyle=densely_dotted, alpha=0.4, linewidth=BASE_LINE_WIDTH)
    handles.append(handle4)
    labels.append("Extrapolated outside \nbenchmark range")

    
    if not params.subplots and not params.skip_legend:
        bench_legend = ax.legend(handles=handles, labels=labels, title='Benchmark', bbox_to_anchor=(0.02, 1), loc='upper left')
        
        # Make section headings bold and left-aligned
        for text in bench_legend.get_texts():
            if text.get_text().startswith('**') and text.get_text().endswith('**'):
                x, y = text.get_position()
                plt.setp(text, text=text.get_text().strip('*'), weight='bold', 
                        position=(x - 0.05, y), ha='left')

    if not params.subplots:
        if texts:
            adjustText.adjust_text(texts, ax=ax,
                                points=points)

        add_watermark(ax)

    if params.subplots:
        # Apply adjustText to each subplot individually
        for bench in benchmarks:
            if bench in subplot_texts and subplot_texts[bench]:
                ax = axs[benchmarks.index(bench)]
                adjustText.adjust_text(subplot_texts[bench], ax=ax,
                                     points=subplot_points[bench],
                                     only_move={'points':'y', 'texts':'y'},
                                     force_text=(0.3, 0.3),
                                     force_points=(0.2, 0.2),
                                     expand_text=(1.2, 1.5),
                                     expand_points=(1.2, 1.2),
                                     avoid_self=True,
                                     avoid_points=True,
                                     avoid_text=True)
        
        for i in range(len(benchmarks), len(axs)):
            axs[i].set_axis_off()

    # After all plotting is done, adjust texts
    if params.subplots:
        # Adjust texts for each subplot
        for ax in axs[:len(benchmarks)]:
            if texts_per_ax[ax]:
                adjustText.adjust_text(texts_per_ax[ax], ax=ax)
    else:
        # Existing code for non-subplot case
        if texts:
            adjustText.adjust_text(texts, ax=ax)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file)
        print(f"Lines plot saved to {output_file}")
        plt.close(fig)



def plot_benchmarks(df: pd.DataFrame, benchmark_data: dict[str, list[float]], output_file: pathlib.Path):
    """
    df is a dataframe holding horizon data for all models on all benchmarks.

    TODO this should have multiple boxplots "within" each scatter plot, one for each split.
    """

    
    length_to_color_map = {
        "baseline": "royalblue",
        "estimate": "indianred",
        "default": "grey"
    }

    benchmarks_to_use = ["hcast_r_s_full_method", "video_mme", "gpqa_diamond", "livecodebench_2411_2505", "mock_aime", "hendrycks_math", "osworld", "rlbench", "swe_bench_verified"]

    # Create a DataFrame for seaborn
    lengths_df = benchmark_data.copy()
    lengths_df['length_type'] = pd.Categorical(lengths_df['length_type'], categories=length_to_color_map.keys(), ordered=True)

    benchmarks = lengths_df['benchmark'].unique().tolist()
    benchmarks = [b for b in benchmarks_to_use if b in benchmarks]
    lengths_df = lengths_df[lengths_df['benchmark'].isin(benchmarks)]
    
    # Sort benchmarks by median length (lowest to highest)
    median_lengths = lengths_df.groupby('benchmark')['length'].median().sort_values(ascending=True)
    benchmarks = median_lengths.index.tolist()
    
    # Set benchmark order for plotting
    lengths_df['benchmark'] = pd.Categorical(lengths_df['benchmark'], categories=benchmarks, ordered=True)

    plt.figure(figsize=(10, 6))
    # Create I-beam error bars showing 10th and 90th percentiles
    for i, benchmark in enumerate(benchmarks):
        bench_data = lengths_df[lengths_df['benchmark'] == benchmark]['length']
        p10 = bench_data.quantile(0.1)
        p90 = bench_data.quantile(0.9)
        median = bench_data.median()
        
        # Draw error bar with caps (I-beam)
        plt.errorbar(i, median, yerr=[[median - p10], [p90 - median]], 
                    fmt='none', color='black', linewidth=2.5, capsize=8, capthick=2.5, zorder=2)
    # Set random seed for reproducible jitter
    np.random.seed(42)
    sns.stripplot(data=lengths_df, y='length', x='benchmark', size=3, hue='length_type', zorder=1, alpha=0.3,
                  palette=length_to_color_map.values(), legend=False, jitter=0.375)

    # plot a diamond for the frontier (max horizon) model on each benchmark
    s_frontier = df.groupby('benchmark', as_index=True)["horizon"].max()
    s_frontier /= 60 # Convert to minutes

    s_frontier = s_frontier[s_frontier.index.isin(benchmarks)]
    
    kwargs = {"label": f"Best performance\nof tested models"}
    for benchmark, horizon in s_frontier.items():
        plt.scatter(benchmark, horizon, color='gold', edgecolor='darkorange', marker='*', s=130, zorder=3, **kwargs)
        kwargs = {}

    # Legend
    plt.plot([], [], color='black', linewidth=2, label="Quantiles\n(10/25/50/75/90%)")
    plt.scatter([], [], color='royalblue', marker='o', s=20, label="Individual\ntask")
    plt.scatter([], [], color='darkred', marker='o', s=20, label="Individual task\n(estimated)")
    plt.legend()

    plt.grid(True, which="major", axis="y",ls="--", linewidth=0.5, alpha=0.4)

    plt.yscale('log')
    plt.ylabel('Length (minutes)')
    plt.xlabel('Benchmark')
    plt.title('Task Lengths By Benchmark')

    
    # Replace x-axis tick labels with benchmark aliases
    ax = plt.gca()
    tick_labels = ax.get_xticklabels()
    new_labels = [benchmark_aliases.get(label.get_text(), label.get_text()).replace(' ', '\n') for label in tick_labels]
    ax.set_xticklabels(new_labels)

    plt.tight_layout()
    add_watermark()
    
    plt.savefig(output_file)
    print(f"Benchmark lengths plot saved to {output_file}")


def setup_beta_dual_axis(ax, y_min=0.08, y_max=4):
    """
    Sets up dual y-axis for beta plots with beta values on left and odds ratio on right
    """
    from matplotlib.ticker import FuncFormatter, FixedLocator
    
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')
    
    # Format left y-axis to show odds ratios
    def odds_ratio_formatter(x, _):
        return f'{np.exp(x):.2f}x'
    
    # Set specific tick locations for denser labeling
    tick_locations = np.logspace(np.log10(y_min), np.log10(y_max), 10)
    ax.yaxis.set_major_locator(FixedLocator(tick_locations))
    ax.yaxis.set_minor_locator(FixedLocator([]))  # Remove minor ticks
    ax.yaxis.set_major_formatter(FuncFormatter(odds_ratio_formatter))
    ax.set_ylabel("Failure rate increase per 2x task length\n(odds ratio)", ha='center')
    
    # Add secondary y-axis on the right showing β values
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yscale('log')
    ax2.yaxis.set_major_locator(FixedLocator(tick_locations))
    ax2.yaxis.set_minor_locator(FixedLocator([]))  # Remove minor ticks
    
    def beta_formatter(x, _):
        return f'{x:.2f}'
    ax2.yaxis.set_major_formatter(FuncFormatter(beta_formatter))
    ax2.set_ylabel(r'$\beta$ (log scale)')
    
    # Ensure the right spine is visible
    ax2.spines['right'].set_visible(True)
    
    # Add text annotations for strongly/weakly related
    ax.annotate("Model success\n$\\mathbf{strongly}$ related\nto task length", 
                (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12)
    ax.annotate("Model success\n$\\mathbf{weakly}$ related\nto task length", 
                (0.05, 0.05), xycoords='axes fraction', ha='left', va='bottom', fontsize=12)
    
    return ax2

def plot_length_dependence(df: pd.DataFrame, output_file: pathlib.Path):
    """
    Plots the relationship between task length and difficulty
    """
    _, ax = plt.subplots(figsize=(10, 6))
    add_watermark(ax)

    benchmarks_to_use = ["hcast_r_s_full_method", "video_mme", "gpqa_diamond", "livecodebench_2411_2505"]

    df_to_use = df[df['benchmark'].isin(benchmarks_to_use)]

    # Calculate HCAST average slope
    hcast_data = df[df['benchmark'] == 'hcast_r_s']
    hcast_avg_slope = hcast_data['slope'].mean() if not hcast_data.empty else 0.6

    # Draw the HRS average β line using the same styling rule:
    #   – standard width when β≈0.6
    #   – slightly thicker when β differs from 0.6.
    base_width = 2
    hrs_line_width = base_width if np.isclose(hcast_avg_slope, 0.6) else base_width + 1.0
    ax.axhline(
        y=hcast_avg_slope,
        color='gray',
        linestyle='-',
        linewidth=hrs_line_width,
        alpha=0.7,
        label=f'HRS average (β={hcast_avg_slope:.2f})'
    )


    # Create color palette using consistent benchmark colors
    palette_dict = {bench: benchmark_colors[bench] for bench in benchmarks_to_use}
    sns.scatterplot(data=df_to_use, x='score', y='slope', hue='benchmark', ax=ax, palette=palette_dict, alpha=0.7)

    # Add bounding ellipses for specific benchmarks using consistent colors
    benchmark_to_ellipse_color = {
        "hcast_r_s_full_method": benchmark_colors["hcast_r_s_full_method"],
        "video_mme": benchmark_colors["video_mme"],
        "gpqa_diamond": benchmark_colors["gpqa_diamond"],
        "livecodebench_2411_2505": benchmark_colors["livecodebench_2411_2505"],
    }
    
    for bench in benchmark_to_ellipse_color.keys():
        bench_data = df_to_use[df_to_use['benchmark'] == bench]
        if len(bench_data) >= 2:  # Need at least 2 points for an ellipse
            # Work in log space for y-coordinates to fit ellipse
            x = bench_data['score'].values
            y_log = np.log(bench_data['slope'].values)
            
            # Calculate mean and covariance in log space
            mean_x = np.mean(x)
            mean_y_log = np.mean(y_log)
            
            # Calculate covariance matrix
            cov = np.cov(x, y_log)
            
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Calculate ellipse parameters (2 standard deviations)
            a = 2 * np.sqrt(eigenvals[0])  # semi-major axis
            b = 2 * np.sqrt(eigenvals[1])  # semi-minor axis
            angle_rad = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            
            # Generate ellipse boundary points in log space
            t = np.linspace(0, 2*np.pi, 100)
            ellipse_x = a * np.cos(t)
            ellipse_y = b * np.sin(t)
            
            # Rotate ellipse
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle
            y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle
            
            # Translate to center and convert y back to linear scale
            ellipse_x_final = x_rot + mean_x
            ellipse_y_final = np.exp(y_rot + mean_y_log)  # Convert back from log space
            
            # Plot ellipse boundary with matching point color
            color = benchmark_colors[bench]
            zorder = 0 if bench == "gpqa_diamond" else 1  # Send gpqa_diamond ellipse to back
            ax.fill(ellipse_x_final, ellipse_y_final, color=color, alpha=0.06 if bench == "gpqa_diamond" else 0.1, edgecolor=color, linewidth=1, zorder=zorder)

    ax.set_xlabel("Model overall score on benchmark")
    ax.set_xlim(0, 1)
    
    # Setup dual y-axis
    setup_beta_dual_axis(ax, y_min=0.08, y_max=4)
    
    ax.set_title("As models get better on GPQA and Video-MME, their performance is\nless correlated with task length")
    
    # Update legend to use benchmark aliases
    handles, labels = ax.get_legend_handles_labels()
    aliased_labels = [benchmark_aliases.get(label, label) for label in labels]
    ax.legend(handles, aliased_labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Length dependence plot saved to {output_file}")


def plot_beta_swarmplot(df: pd.DataFrame, output_file: pathlib.Path):
    """
    Creates a vertical swarmplot showing beta (slope) values for each benchmark
    """
    _, ax = plt.subplots(figsize=(9.4, 6))
    add_watermark(ax)
    
    benchmarks_to_use = ["hcast_r_s_full_method", "swe_bench_verified", "video_mme", "gpqa_diamond", 
                         "livecodebench_2411_2505", "mock_aime"]
    
    df_to_use = df[df['benchmark'].isin(benchmarks_to_use)].copy()
    
    # Sort benchmarks to put hcast_r_s_full_method first
    df_to_use['benchmark'] = pd.Categorical(df_to_use['benchmark'], 
                                           categories=benchmarks_to_use, 
                                           ordered=True)
    
    # Filter out NaN slopes and placeholder values
    df_to_use = df_to_use.dropna(subset=['slope'])
    df_to_use = df_to_use[df_to_use['slope'] > 0]  # Remove any invalid slopes
    
    if df_to_use.empty:
        print("No valid slope data found for beta swarmplot")
        return
    
    # Create color palette using consistent benchmark colors
    palette_dict = {bench: benchmark_colors[bench] for bench in benchmarks_to_use if bench in df_to_use['benchmark'].unique()}
    
    # Create the swarmplot (use stripplot for better visibility when many points overlap)
    sns.stripplot(data=df_to_use, x='benchmark', y='slope', hue='benchmark', 
                  palette=palette_dict, size=6, ax=ax, legend=False, jitter=True, alpha=0.5)
    
    # Setup dual y-axis for beta values
    setup_beta_dual_axis(ax, y_min=0.08, y_max=4)
    
    # Add faint grid lines for major y-ticks
    ax.grid(True, which="major", axis="y", ls="--", linewidth=0.5, alpha=0.4)
    
    ax.set_title("Benchmarks vary in how strongly model performance correlates to task length\n(each point is a model)")
    ax.set_xlabel("")  # Remove x-axis label
    
    # Replace x-axis tick labels with benchmark aliases
    current_labels = [t.get_text() for t in ax.get_xticklabels()]
    new_labels = [benchmark_aliases.get(label, label).replace(' ', '\n') for label in current_labels]
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Beta swarmplot saved to {output_file}")


def plot_percent_over_time(df, output_file, benchmarks_to_skip=None):
    """Generates and saves a plot of score (percentage) over time for frontier models only."""
    
    assert not df.empty, "No data loaded for percent over time plot."
    assert 'score' in df.columns and not df['score'].isnull().all(), "No valid 'score' data found."
    assert 'release_date' in df.columns and not df['release_date'].isnull().all(), "No valid 'release_date' data found."

    plot_df = df.dropna(subset=['release_date', 'score']).copy()
    assert not plot_df.empty, "No valid data points (with release date and score) found."

    plot_df['release_date_num'] = mdates.date2num(plot_df['release_date'])

    # Identify frontier models for each benchmark (models with highest score at each time point)
    plot_df['is_frontier'] = False
    benchmarks = plot_df['benchmark'].unique()

    if benchmarks_to_skip is not None:
        benchmarks = [bench for bench in benchmarks if bench not in benchmarks_to_skip]
    
    for bench in benchmarks:
        bench_df = plot_df[plot_df['benchmark'] == bench].sort_values(by=['release_date_num', 'score'], ascending=[True, False])
        max_score_so_far = -np.inf
        frontier_indices = []
        for index, row in bench_df.iterrows():
            # A model is on the frontier if its score is greater than all previous models' scores
            if row['score'] > max_score_so_far:
                frontier_indices.append(index)
                max_score_so_far = row['score']
        if frontier_indices:
            plot_df.loc[frontier_indices, 'is_frontier'] = True

    # Filter to only frontier models
    frontier_df = plot_df[plot_df['is_frontier']].copy()
    assert not frontier_df.empty, "No frontier models found for percent over time plot."

    fig, ax = plt.subplots(figsize=(12, 8))

    for bench in benchmarks:
        bench_frontier = frontier_df[frontier_df['benchmark'] == bench]
        if bench_frontier.empty:
            continue
            
        color = benchmark_colors[bench]
        
        # Sort by release date for proper line plotting
        bench_frontier = bench_frontier.sort_values('release_date')
        
        # Plot the line connecting frontier points
        ax.plot(bench_frontier['release_date'], bench_frontier['score'] * 100, 
                color=color, linewidth=2.5, marker='o', markersize=6,
                label=benchmark_aliases.get(bench, bench))

    ax.set_xlabel("Model Release Date")
    ax.set_ylabel("Score (%)")
    ax.set_title("Performance Over Time (Frontier Models Only)")
    ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45, ha='right')

    # Set y-axis to show percentages nicely
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Create legend
    ax.legend(title='Benchmark', bbox_to_anchor=(0.02, 1), loc='upper left')
    
    add_watermark(ax)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Percent over time plot saved to {output_file}")
    plt.close(fig)

def _plot_bench_group(ax, df, benchmark_data, benches, params, date_cutoff, x_bound=None):
    # Draw directly on the provided axes
    plot_lines_over_time(
        df.copy(),
        output_file=None,  # Don't save to file
        benchmark_data=benchmark_data,
        params=dataclasses.replace(
            params,
            show_benchmarks=benches,
            hide_benchmarks=[],
            subplots=False,
            date_cutoff=date_cutoff,
            xbound=x_bound,
            skip_labels=True,  # Skip individual labels
            skip_legend=True  # Skip individual legends
        ),
        ax=ax  # Pass the axes directly
    )        


def plot_robustness_subplots(df: pd.DataFrame,
                            benchmark_data: pd.DataFrame,
                            output_file: str):
    """
    2x2 version of the big subplot figure.
    """
    # Create a 2×2 grid and keep the overall footprint compact
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=True)
    # Flatten for easier indexing
    axs_flat = axs.flatten()

    common_params = LinesPlotParams(
        show_points_level=ShowPointsLevel.ALL,
        show_dotted_lines=False,
        verbose=False,
        xbound=("2022-01-01", "2025-12-31"),
        date_cutoff=None,  # show first-label even for post-cutoff models
        rotate_dates=False,  # Don't rotate dates for robustness subplots
    )

    # Top-left – AIME family
    _plot_bench_group(
        ax=axs_flat[0],
        df=df,
        benchmark_data=benchmark_data,
        benches=["aime", "mock_aime"],
        params=common_params,
        date_cutoff=date(2025, 1, 1),
        x_bound=("2023-05-01", "2025-06-01")
    )
    axs_flat[0].set_title("AIME", fontsize=10)

    # Top-right – HRS family
    _plot_bench_group(
        ax=axs_flat[1],
        df=df,
        benchmark_data=benchmark_data,
        benches=["hcast_r_s", "hcast_r_s_full_method"],
        params=common_params,
        date_cutoff=date(2025, 1, 1)
    )
    axs_flat[1].set_title("HRS", fontsize=10)

    # Bottom-left – GPQA family
    _plot_bench_group(
        ax=axs_flat[2],
        df=df,
        benchmark_data=benchmark_data,
        benches=["gpqa", "gpqa_diamond"],
        params=common_params,
        date_cutoff=date(2023, 6, 1),
        x_bound=("2023-05-01", "2025-06-01")
    )
    axs_flat[2].set_title("GPQA", fontsize=10)

    _plot_bench_group(
        ax=axs_flat[3],
        df=df,
        benchmark_data=benchmark_data,
        benches=["livecodebench_2411_2505","livecodebench_2411_2505_approx"],
        params=common_params,
        date_cutoff=date(2024, 1, 1),
        x_bound=("2023-05-01", "2025-06-01")
    )
    axs_flat[3].set_title("LiveCodeBench", fontsize=10)


    # Create a structured legend with two columns
    from matplotlib.lines import Line2D
    
    # Collect unique legend entries from all axes
    all_handles = []
    all_labels = []
    seen_labels = set()
    
    for ax in axs_flat[:4]:  # Only the 4 subplots we're using
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in seen_labels and not label.startswith('_'):
                all_handles.append(handle)
                all_labels.append(label)
                seen_labels.add(label)
    
    # Separate benchmarks into β known and approximate categories
    # Based on the thick line logic from the main plotting function
    beta_known = ["Mock AIME", "HRS (Original)", "GPQA Diamond", "LiveCodeBench"]  # These have thick lines (not has_placeholder_slope)
    approximate = []  # These have thin lines (has_placeholder_slope)
    
    # Categorize based on actual benchmark names that appear in the legend
    for handle, label in zip(all_handles, all_labels):
        if label not in beta_known:
            approximate.append(label)
    
    # Create structured legend
    structured_handles = []
    structured_labels = []
    
    # Add β known section
    heading_handle = Line2D([0], [0], color='none')
    structured_handles.append(heading_handle)
    structured_labels.append('Known β')
    
    for handle, label in zip(all_handles, all_labels):
        if label in beta_known:
            structured_handles.append(handle)
            structured_labels.append(label)
    
    # Add approximate section
    heading_handle2 = Line2D([0], [0], color='none')
    structured_handles.append(heading_handle2)
    structured_labels.append('Approximate (Unknown β)')
    
    for handle, label in zip(all_handles, all_labels):
        if label in approximate:
            structured_handles.append(handle)
            structured_labels.append(label)
    
    # Add global title and axis labels
    fig.suptitle("Robustness of Approximate Time Horizon Methods", fontsize=14, y=0.99)
    fig.supxlabel("Model Release Date", fontsize=14, y=0.01)
    fig.supylabel("50% Time Horizon (minutes)", fontsize=14, x=0.01)
    
    # Create shared legend at the bottom with 2 columns
    legend = fig.legend(structured_handles, structured_labels, loc='center', bbox_to_anchor=(0.5, -0.1), 
                       ncol=2, frameon=True, fontsize=9)
    
    # Make section headings bold
    for text in legend.get_texts():
        if text.get_text() in ['Known β', 'Approximate (Unknown β)']:
            text.set_weight('bold')
    
    # Adjust layout with more space for legend and labels
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(wspace=0.08, hspace=0.25, top=0.92, bottom=0.10, left=0.12, right=0.98)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Robustness subplots saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate various plots for model analysis')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--lines', action='store_true', help='Generate lines over time plot')
    parser.add_argument('--hcast', action='store_true', help='Generate hcast comparison plot')
    parser.add_argument('--lengths', action='store_true', help='Generate benchmark lengths plot')
    parser.add_argument('--length-dependence', action='store_true', help='Generate length dependence plot')
    parser.add_argument('--speculation', action='store_true', help='Generate speculation plot')
    parser.add_argument('--splits', action='store_true', help='Generate splits plot')
    parser.add_argument('--percent', action='store_true', help='Generate percent over time plot')
    parser.add_argument('--robustness', action='store_true',
                        help='Generate robustness subplots')
    parser.add_argument('--beta-swarm', action='store_true',
                        help='Generate beta swarmplot')
    args = parser.parse_args()

    plots_to_make = []
    if args.all or not any(vars(args).values()):
        plots_to_make = ["lines", "hcast", "lengths", "length_dependence", "robustness"]
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
    elif args.percent:
        plots_to_make += ["percent"]
    elif args.robustness:
        plots_to_make += ["robustness"]
    elif args.beta_swarm:
        plots_to_make += ["beta_swarm"]

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
        plot_lines_over_time(all_df.copy(), HEADLINE_PLOT_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s", "video_mme", "gpqa", "aime", "livecodebench_2411_2505_approx"], show_points_level=ShowPointsLevel.NONE, verbose=False, show_dotted_lines=False, ybound=(0.05, 400)))

        # Generate and save the lines over time plot using the original loaded data
        plot_lines_over_time(all_df.copy(), LINES_PLOT_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s", "video_mme", "gpqa", "aime","livecodebench_2411_2505_approx"], show_points_level=ShowPointsLevel.FRONTIER, verbose=False))
        
        plot_lines_over_time(all_df.copy(), LINES_SUBPLOTS_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s", "video_mme","livecodebench_2411_2505_approx"], show_points_level=ShowPointsLevel.ALL, subplots=True, show_doubling_rate=True, xbound=("2021-01-01", "2025-08-01"), ybound=(0.05, 400)))


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

    if "percent" in plots_to_make:
        plot_percent_over_time(all_df.copy(), PERCENT_OVER_TIME_OUTPUT_FILE, benchmarks_to_skip=["hcast_r_s", "gpqa", "livecodebench_2411_2505_approx"])

    if "robustness" in plots_to_make:
        plot_robustness_subplots(all_df.copy(),
                                benchmark_data,
                                ROBUSTNESS_SUBPLOTS_OUTPUT_FILE)

    if "beta_swarm" in plots_to_make:
        plot_beta_swarmplot(all_df.copy(), BETA_SWARMPLOT_OUTPUT_FILE)

if __name__ == "__main__":
    main()

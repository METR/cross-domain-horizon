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
Y_AXIS_MIN_SECONDS = 60  # 1 minute
WATERMARK = False


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
    overlay: bool = False

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

    if params.overlay:
        # Create figure normally - we'll add the background image later
        fig, ax = plt.subplots(figsize=(12, 8))
        # Store the background image path for later use
        fig._overlay_image_path = 'plots/original_time_horizon_plot.png'
    elif params.subplots:
        # last subplot contains legend
        nrows = (len(benchmarks) + 1) // 4
        fig, axs = plt.subplots(figsize=(12, nrows * 4), nrows=nrows, ncols=4, sharex=True, sharey=True)
        nrows = (len(benchmarks) + 1) // 4
        fig, axs = plt.subplots(figsize=(12, nrows * 4), nrows=nrows, ncols=4, sharex=True, sharey=True)
        axs = axs.flatten()
        texts_per_ax = {ax: [] for ax in axs}  # Track texts per axis
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    texts = [] # Initialize list to store text objects for adjustText
    subplot_texts = {} # Store texts for each subplot separately

    if params.hide_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench not in params.hide_benchmarks]
    if params.show_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench in params.show_benchmarks]

    densely_dotted = (0, (1, 1))
    for bench in benchmarks:
        if params.subplots:
            ax = axs[benchmarks.index(bench)]
            ax.set_title(benchmark_aliases[bench], fontsize=10)
            subplot_texts[bench] = []  # Initialize text list for this subplot

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
                    elif row['benchmark'] == 'hcast_r_s':
                        marker = '^'
                    else:
                        marker = 'x' if slope_val < 0.25 else 'o' if slope_val == 0.6 else '^'
                    
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

        # Fit and plot smoothing spline using only frontier points
        if len(frontier_data) >= 3:  # Need at least 3 points for spline
            # Sort frontier data by release date for spline fitting
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

                rate_text = f"{doubling_rate_per_year:.1f} dbl./yr"
                # Place text in top left corner of each subplot
                ax.text(0.02, 0.99, rate_text, fontsize=10, color=color, 
                        ha='left', va='top', transform=ax.transAxes)

            x_line_date = np.array(mdates.num2date(x_line_num))
            y_line = 2.0**y_line_log

            # Split the line into three segments based on p2 and p98
            mask_within = (y_line >= p2) & (y_line <= p98)
            mask_above = y_line > p98
            mask_below = y_line < p2

            thick = (bench == "hcast_r_s") and not params.subplots

            ax.plot(x_line_date[mask_within], y_line[mask_within], color=color, linestyle='-', linewidth=5 if thick else 2.5, label=f"{benchmark_aliases[bench]}", zorder=100 if thick else None)
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
                        y_offset = 0.20  # fixed position
                        va = 'bottom'
                    else:
                        y_offset = point['horizon_minutes'] * 1.1  # 10% above the point
                        va = 'bottom'
                    
                    text_obj = ax.text(point['release_date'],
                                      y_offset,
                                      model_name,
                                      fontsize=8 if params.subplots else 10, 
                                      color=color,
                                      ha='center',
                                      va=va,
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                    if params.subplots:
                        subplot_texts[bench].append(text_obj)
                    else:
                        texts.append(text_obj)

                text_label(first_frontier_point, current_ax=ax)
                text_label(last_frontier_point, current_ax=ax)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x}'))
    plt.ylim(0.05, 3000)

    if params.subplots:
        fig.suptitle(params.title)
        fig.supxlabel("Model Release Date")
        fig.supylabel("50% Time Horizon (minutes)")
    else:
        plt.xlabel("Model Release Date")
        plt.ylabel("50% Time Horizon (minutes)")
        plt.title(params.title)
        ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)



    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45, ha='right') # Auto-format dates (includes rotation)
    
    # Set x-axis bounds if specified
    if params.xbound is not None:
        start_date, end_date = params.xbound
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    if params.ybound is not None:
        ax.set_ylim(params.ybound)


    # Create a legend
    handles, labels = ax.get_legend_handles_labels()
    trend_legend_handles = []
    if params.show_dotted_lines:

        handle1, = ax.plot([], [], color='black', linestyle='-', linewidth=2, label="Inside range")
        trend_legend_handles.append(handle1)
        handle2, = ax.plot([], [], color='black', linestyle=densely_dotted, alpha=0.4, linewidth=2, label="Outside range")
        trend_legend_handles.append(handle2)

    # Create fit quality legend
    fit_quality_handles = []
    handle3 = ax.scatter([], [], color='black', marker='^', s=30, label="Good (β > 0.25)")
    fit_quality_handles.append(handle3)
    handle4 = ax.scatter([], [], color='black', marker='x', s=30, label="Poor (β < 0.25)")
    fit_quality_handles.append(handle4)
    handle5 = ax.scatter([], [], color='black', marker='o', s=30, label="Unknown")
    fit_quality_handles.append(handle5)

    trend_legend_ax = axs[-1] if params.subplots else ax

    trend_legend = trend_legend_ax.legend(handles=trend_legend_handles, title="Range of task lengths\n in benchmark", bbox_to_anchor=(0.02, 0.5), loc='center left')
    
    fit_quality_legend = trend_legend_ax.legend(handles=fit_quality_handles, title="Logistic fit quality", bbox_to_anchor=(0.02, 0.3), loc='center left')

    
    if not params.subplots:
        bench_legend = ax.legend(handles=handles, labels=labels, title='Benchmark', bbox_to_anchor=(0.02, 1), loc='upper left')
        if params.show_dotted_lines:
            ax.add_artist(trend_legend)
            ax.add_artist(fit_quality_legend)

        if texts:
            adjustText.adjust_text(texts, ax=ax,
                                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

        add_watermark(ax)

    if params.subplots:
        # Apply adjustText to each subplot individually
        for bench in benchmarks:
            if bench in subplot_texts and subplot_texts[bench]:
                ax = axs[benchmarks.index(bench)]
                adjustText.adjust_text(subplot_texts[bench], ax=ax,
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
        for ax_idx, ax in enumerate(axs[:len(benchmarks)]):
            if texts_per_ax[ax]:
                adjustText.adjust_text(texts_per_ax[ax], ax=ax,
                                     arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    else:
        # Existing code for non-subplot case
        if texts:
            adjustText.adjust_text(texts, ax=ax,
                                 arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    plt.tight_layout()

    # Add background image if this is an overlay plot
    if hasattr(fig, '_overlay_image_path'):
        # Hide all axes elements for clean overlay
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Hide legend if it exists
        legend = ax.get_legend()
        if legend:
            legend.set_visible(False)
        # Hide grid
        ax.grid(False)
        import matplotlib.image as mpimg
        from PIL import Image
        try:
            background_img = mpimg.imread(fig._overlay_image_path)
            
            # Get figure dimensions in pixels
            fig_width_px = fig.get_figwidth() * fig.dpi / 2
            fig_height_px = fig.get_figheight() * fig.dpi / 2
            
            # Resize background image to match figure size
            pil_img = Image.fromarray((background_img * 255).astype('uint8'))
            resized_img = pil_img.resize((int(fig_width_px), int(fig_height_px)), Image.Resampling.LANCZOS)
            resized_array = np.array(resized_img) / 255.0
            
            # Add the background image covering the entire figure
            fig.figimage(resized_array, xo=0, yo=0, alpha=0.4, zorder=100)
        except FileNotFoundError:
            print(f"Warning: Background image {fig._overlay_image_path} not found.")
        except ImportError:
            print("Warning: PIL/Pillow not available for image resizing. Using original size.")

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

    plt.tight_layout()
    add_watermark()
    
    plt.savefig(output_file)
    print(f"Benchmark lengths plot saved to {output_file}")


def plot_length_dependence(df: pd.DataFrame, output_file: pathlib.Path):
    """
    Plots the relationship between task length and difficulty
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    add_watermark(ax)

    benchmarks_to_use = ["hcast_r_s_full_method", "video_mme", "gpqa_diamond", "livecodebench_2411_2505", "mock_aime"]

    df_to_use = df[df['benchmark'].isin(benchmarks_to_use)]

    # Calculate HCAST average slope
    hcast_data = df[df['benchmark'] == 'hcast_r_s']
    hcast_avg_slope = hcast_data['slope'].mean() if not hcast_data.empty else 0.6

    ax.axhline(y=hcast_avg_slope, color='gray', linestyle='--', alpha=0.7, label=f'HRS average (β={hcast_avg_slope:.2f})')

    ax.annotate("Model success\n$\\mathbf{strongly}$ related\nto task length", (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12)
    ax.annotate("Model success\n$\\mathbf{weakly}$ related\nto task length", (0.05, 0.05), xycoords='axes fraction', ha='left', va='bottom', fontsize=12)


    # Create color palette using consistent benchmark colors
    palette_dict = {bench: benchmark_colors[bench] for bench in benchmarks_to_use}
    sns.scatterplot(data=df_to_use, x='score', y='slope', hue='benchmark', ax=ax, palette=palette_dict)

    # Add bounding ellipses for specific benchmarks using consistent colors
    benchmark_to_ellipse_color = {
        "hcast_r_s_full_method": benchmark_colors["hcast_r_s_full_method"],
        "video_mme": benchmark_colors["video_mme"],
        "gpqa_diamond": benchmark_colors["gpqa_diamond"],
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
            ax.fill(ellipse_x_final, ellipse_y_final, color=color, alpha=0.1, edgecolor=color, linewidth=1)

    ax.set_ylabel("Failure odds ratio per task length doubling")
    ax.set_xlabel("Model overall score on benchmark")

    ax.set_ylim(0.08, 4)
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    
    # Format y-axis ticks to show odds ratios (exp of slope values)
    from matplotlib.ticker import FuncFormatter, FixedLocator
    def odds_ratio_formatter(x, pos):
        return f'{np.exp(x):.2f}x'
    
    # Set specific tick locations for denser labeling
    tick_locations = np.logspace(np.log10(0.08), np.log10(4), 10)
    ax.yaxis.set_major_locator(FixedLocator(tick_locations))
    ax.yaxis.set_minor_locator(FixedLocator([]))  # Remove minor ticks
    ax.yaxis.set_major_formatter(FuncFormatter(odds_ratio_formatter))
    
    # Add secondary y-axis on the right showing original β values
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yscale('log')
    ax2.yaxis.set_major_locator(FixedLocator(tick_locations))
    ax2.yaxis.set_minor_locator(FixedLocator([]))  # Remove minor ticks
    
    def beta_formatter(x, pos):
        return f'{x:.2f}'
    ax2.yaxis.set_major_formatter(FuncFormatter(beta_formatter))
    ax2.set_ylabel(r'$\beta$')
    
    # Ensure the right spine is visible
    ax2.spines['right'].set_visible(True)
    
    ax.set_title("Benchmarks have varying relationships between task length and difficulty\n(each point is a model)")
    
    # Update legend to use benchmark aliases
    handles, labels = ax.get_legend_handles_labels()
    aliased_labels = [benchmark_aliases.get(label, label) for label in labels]
    ax.legend(handles, aliased_labels)

    plt.savefig(output_file)
    print(f"Length dependence plot saved to {output_file}")


def plot_percent_over_time(df, output_file):
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
    elif args.percent:
        plots_to_make += ["percent"]

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
        plot_lines_over_time(all_df.copy(), HEADLINE_PLOT_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s_full_method", "video_mme", "gpqa", "aime"], show_points_level=ShowPointsLevel.NONE, verbose=False, show_dotted_lines=False, ybound=(0.05, 400)))

        # Generate and save the lines over time plot using the original loaded data
        plot_lines_over_time(all_df.copy(), LINES_PLOT_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s_full_method", "video_mme", "gpqa", "aime"], show_points_level=ShowPointsLevel.FRONTIER, verbose=False))

        plot_lines_over_time(all_df.copy(), "plots/hcast_comparison.png", benchmark_data, LinesPlotParams(
            title="HRS Time Horizons (full method vs average-scores-only)",
            show_benchmarks=["hcast_r_s", "hcast_r_s_full_method"], show_points_level=ShowPointsLevel.FRONTIER,)
        )
        
        # Overlay plot - like headline but overlaid on original plot
        plot_lines_over_time(all_df.copy(), "plots/overlay.png", benchmark_data, LinesPlotParams(
            hide_benchmarks=["hcast_r_s", "hcast_r_s_full_method", "video_mme", "gpqa", "aime"], 
            show_points_level=ShowPointsLevel.NONE, 
            verbose=False, 
            show_dotted_lines=False, 
            ybound=(0.05, 400),
            overlay=True,
            title="Time Horizon vs. Release Date (Overlay)"
        ))
        
        plot_lines_over_time(all_df.copy(), LINES_SUBPLOTS_OUTPUT_FILE, benchmark_data, LinesPlotParams(hide_benchmarks=["hcast_r_s_full_method"], show_points_level=ShowPointsLevel.ALL, subplots=True, show_doubling_rate=True, xbound=("2021-01-01", "2026-01-01")))


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
        plot_percent_over_time(all_df.copy(), PERCENT_OVER_TIME_OUTPUT_FILE)

if __name__ == "__main__":
    main()

"""
Combined visualization of benchmark trends and bootstrap confidence intervals.

This module merges functionality from lines_over_time.py and bootstrap_ci.py to create
a comprehensive view of AI progress over time. The resulting plot overlays statistical
uncertainty estimates on benchmark-specific performance trends.

Components displayed:
1. Benchmark trend lines (colored lines)
   - Each benchmark gets its own color (defined in benchmark_colors)
   - Lines show frontier model performance (best at each point in time)
   - Solid lines indicate regions with sufficient task data (2nd-98th percentile)
   - Dotted lines indicate extrapolation beyond typical task lengths
   - Optional markers indicate logistic fit quality (^ = good, x = poor, o = unknown)

2. Bootstrap confidence interval (green shaded region)
   - Shows uncertainty in overall AI progress trends
   - Calculated by fitting exponential trends to bootstrap samples
   - Default 95% confidence interval (configurable)
   - Lighter green shading with semi-transparency

3. Bootstrap median trend (green dashed line)
   - Central tendency from bootstrap analysis
   - Represents median prediction across all bootstrap samples

4. Individual model points (large red dots)
   - Shows specific model performance from bootstrap data
   - Only displays frontier models (each improving on all previous)
   - Optional labels with model names (controlled by --model-names flag)
   - Uses adjustText to prevent label overlap

Data sources:
- Benchmark data: all_data.csv (created by wrangle.py)
  Contains: model, benchmark, horizon (seconds), release_date, slope
- Task lengths: benchmark TOML files in data/benchmarks/
- Bootstrap data: data/wrangled/bootstrap/headline.csv
  Contains: columns like "GPT-4_p50" with bootstrap samples
- Release dates: data/external/release_dates.yaml

Key parameters (CombinedPlotParams):
- show_points_level: Controls which benchmark points to display
- show_bootstrap_ci: Enable/disable confidence interval overlay
- show_individual_agents: Show/hide red model dots
- show_model_names: Display model labels (via --model-names flag)
- exclude_agents: List of models to exclude from bootstrap
- confidence_level: CI level (e.g., 0.95 for 95%)
- success_percent: Which percentile to plot (default 50 = median)

Usage:
    python combined.py                  # No labels
    python combined.py --model-names    # With model name labels

The plot helps visualize:
- How individual benchmarks are progressing over time
- Overall statistical trends across all models
- Uncertainty in AI progress extrapolations
- Which models represent true capability improvements (frontier)
"""

import adjustText
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import toml
from scipy.interpolate import make_splrep
from enum import Enum
from dataclasses import dataclass, field
from functools import total_ordering
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num, num2date
import yaml
import logging

from plotting_aliases import benchmark_aliases, plotting_aliases, benchmark_colors

logger = logging.getLogger(__name__)

# Configuration constants
BENCHMARKS_PATH = 'data/benchmarks'  # Path to benchmark TOML files
LINES_PLOT_OUTPUT_FILE = 'plots/combined.png'  # Default output filename
WATERMARK = False  # Whether to add watermark (not implemented)


@total_ordering
class ShowPointsLevel(Enum):
    """
    Enum to control which data points are displayed on benchmark lines.
    
    NONE: Only show trend lines, no individual points
    FIRST_AND_LAST: Show only the first and last frontier points for each benchmark
    FRONTIER: Show all frontier (best-performing) points
    ALL: Show all data points including non-frontier models
    """
    NONE = 0
    FIRST_AND_LAST = 1
    FRONTIER = 2
    ALL = 3
    def __lt__(self, other):
        return self.value < other.value

@dataclass
class CombinedPlotParams:
    """
    Configuration parameters for the combined plot.
    
    This dataclass contains all parameters needed to customize both the benchmark
    trend lines (from lines_over_time.py) and bootstrap CI overlay (from bootstrap_ci.py).
    """
    # Benchmark line parameters
    show_points_level: ShowPointsLevel  # Which points to show on benchmark lines
    show_benchmarks: list[str] = field(default_factory=list)  # Only show these benchmarks (if specified)
    hide_benchmarks: list[str] = field(default_factory=list)  # Hide these benchmarks
    show_dotted_lines: bool = True  # Show dotted lines for task length ranges
    show_doubling_rate: bool = False  # Display doubling rate annotations on lines
    title: str = "Time Horizon vs. Release Date with Bootstrap CI"
    verbose: bool = False  # Print debug information
    xbound: tuple[str, str] | None = None  # X-axis date bounds (start, end)
    ybound: tuple[float, float] | None = None  # Y-axis time bounds in minutes
    
    # Bootstrap CI parameters
    show_bootstrap_ci: bool = True  # Whether to overlay bootstrap confidence intervals
    bootstrap_data_file: str | None = None  # Path to bootstrap results CSV
    agent_summaries_file: str | None = None  # Path to agent summaries (optional)
    confidence_level: float = 0.95  # Confidence level for bootstrap CI (e.g., 0.95 for 95%)
    exclude_agents: list[str] = field(default_factory=list)  # Agents to exclude from bootstrap
    success_percent: int = 50  # Which percentile to use (50 = median)
    
    # Individual agent point parameters
    show_individual_agents: bool = False  # Show individual model points from bootstrap data
    agent_point_color: str = "#d06494"  # Color for individual agent points (black)
    agent_point_size: int = 100  # Size of individual agent points
    show_model_names: bool = False  # Show model names on points (controlled by --model-names flag)


def get_benchmark_data(benchmarks_path) -> pd.DataFrame:
    """
    Loads benchmark task length data from TOML files.
    
    This function reads benchmark configuration files that contain information about
    task lengths for each benchmark. These lengths are used to determine the dotted
    line ranges (2nd to 98th percentile) on the plot.
    
    Args:
        benchmarks_path: Directory containing benchmark TOML files
        
    Returns:
        DataFrame with columns: length, benchmark, length_type
    """
    benchmark_data = {}
    for file in os.listdir(benchmarks_path):
        if file.endswith('.toml'):
            with open(os.path.join(benchmarks_path, file), 'r') as f:
                benchmark_data[file.replace('.toml', '')] = toml.load(f)

    # Extract task lengths from each benchmark's data
    result = pd.DataFrame([
        {'length': length, 'benchmark': benchmark, 'length_type': data.get('length_type', "default")}
        for benchmark, data in benchmark_data.items()
        for split_name, split_data in data['splits'].items() if split_name != "all" or len(data['splits']) == 1
        for length in split_data['lengths']
    ])
    return result


def fit_trendline(
    p50s: pd.Series,
    release_dates: pd.Series,
    log_scale: bool = True,
) -> tuple[LinearRegression, float]:
    """
    Fit exponential trendline using linear regression in log space.
    
    This function is used by the bootstrap CI calculation to fit trend lines
    to model performance data. It performs linear regression on log-transformed
    data to achieve exponential fitting.
    
    Args:
        p50s: Series of performance values (e.g., 50th percentile times)
        release_dates: Series of model release dates
        log_scale: Whether to use log transformation (True for exponential fit)
        
    Returns:
        Tuple of (fitted LinearRegression model, R² score)
    """
    # Convert dates to numeric values for regression
    X = date2num(release_dates).reshape(-1, 1)
    
    if log_scale:
        y = np.log(p50s)  # Log transform for exponential fitting
    else:
        y = p50s
    
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    
    return reg, score


def add_bootstrap_confidence_region(
    ax: plt.Axes,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, pd.Timestamp]],
    after_date: str,
    max_date: pd.Timestamp,
    confidence_level: float,
    exclude_agents: list[str],
    success_percent: int = 50,
    benchmark_data: pd.DataFrame = None,
) -> tuple[list[float], pd.DataFrame]:
    """
    Add bootstrap confidence intervals and median trend to the plot.
    
    This function calculates and displays:
    1. A shaded confidence region showing uncertainty in AI progress trends
    2. A median trend line from bootstrap samples
    3. Collects data for individual agent points (if requested)
    
    The bootstrap results contain multiple samples (rows) with performance values
    for each agent (columns). This function fits exponential trends to each sample
    and calculates confidence bounds.
    
    Args:
        ax: Matplotlib axes to plot on
        bootstrap_results: DataFrame with columns "{agent}_p{percentile}" containing bootstrap samples
        release_dates: Dictionary with "date" key mapping agent names to release dates
        after_date: Start date for confidence region projection
        max_date: End date for confidence region projection
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        exclude_agents: List of agent names to exclude from analysis
        success_percent: Which percentile to use (e.g., 50 for median)
        benchmark_data: Optional benchmark data (not currently used)
        
    Returns:
        Tuple of:
        - List of doubling times calculated from each bootstrap sample
        - DataFrame with agent summary data (name, date, median performance)
    """
    # Extract release dates and filter agents
    dates = release_dates["date"]
    focus_agents = sorted(list(dates.keys()), key=lambda x: dates[x])
    focus_agents = [agent for agent in focus_agents if agent not in exclude_agents]
    doubling_times = []
    
    # First, collect median performance for each agent across all bootstrap samples
    # This will be used for plotting individual agent points
    agent_data = []
    for agent in focus_agents:
        col_name = f"{agent}_p{success_percent}"
        if col_name not in bootstrap_results.columns:
            continue
        
        # Calculate median value across all bootstrap samples for this agent
        agent_values = pd.to_numeric(bootstrap_results[col_name], errors="coerce")
        median_val = agent_values.median()
        
        if not pd.isna(median_val) and median_val > 0:
            agent_data.append({
                'agent': agent,
                'release_date': dates[agent],
                f'p{success_percent}': median_val
            })
    
    agent_summary_df = pd.DataFrame(agent_data)
    
    # Now calculate confidence bounds by fitting trends to each bootstrap sample
    n_bootstraps = len(bootstrap_results)
    
    # Create daily time points for smooth confidence region plotting
    time_points = pd.date_range(
        start=pd.to_datetime(after_date),
        end=max_date,
        freq="D",
    )
    predictions = np.zeros((n_bootstraps, len(time_points)))
    
    # Fit exponential trend to each bootstrap sample
    valid_samples = 0
    for sample_idx in range(n_bootstraps):
        # Collect valid p50 values and dates for this sample
        valid_p50s = []
        valid_dates = []
        
        for agent in focus_agents:
            col_name = f"{agent}_p{success_percent}"
            if col_name not in bootstrap_results.columns:
                continue
                
            p_val = pd.to_numeric(
                bootstrap_results[col_name].iloc[sample_idx], errors="coerce"
            )
            
            if pd.isna(p_val) or np.isinf(p_val) or p_val < 1e-3:
                continue
                
            # Bootstrap data is already in minutes, no conversion needed
            valid_p50s.append(p_val)
            valid_dates.append(dates[agent])
        
        if len(valid_p50s) < 2:
            continue  # Need at least 2 points to fit a line
            
        # Fit exponential trend to this bootstrap sample
        reg, _ = fit_trendline(
            pd.Series(valid_p50s),
            pd.Series(pd.to_datetime(valid_dates)),
            log_scale=True,
        )
        
        # Generate predictions for all time points
        time_x = date2num(time_points)
        predictions[valid_samples] = np.exp(reg.predict(time_x.reshape(-1, 1)))
        
        # Calculate doubling time from the slope
        slope = reg.coef_[0]
        doubling_time = np.log(2) / slope
        if doubling_time > 0:  # Only include positive doubling times
            doubling_times.append(doubling_time)
        valid_samples += 1
    
    # Trim predictions array to valid samples
    predictions = predictions[:valid_samples]
    
    if valid_samples > 0:
        # Calculate confidence bounds from all valid predictions
        low_q = (1 - confidence_level) / 2  # e.g., 0.025 for 95% CI
        high_q = 1 - low_q  # e.g., 0.975 for 95% CI
        lower_bound = np.nanpercentile(predictions, low_q * 100, axis=0)
        upper_bound = np.nanpercentile(predictions, high_q * 100, axis=0)
        
        # Bootstrap data is already in minutes (no conversion needed)
        lower_bound_minutes = lower_bound
        upper_bound_minutes = upper_bound
        
        # Plot confidence region as shaded area
        ax.fill_between(
            time_points,
            lower_bound_minutes,
            upper_bound_minutes,
            color="#d2dfd7",  # Light green color
            alpha=0.4,
            label=f"{int(confidence_level*100)}% Bootstrap CI",
            zorder=10  # Behind benchmark lines but above grid
        )
        
        # Plot median trend line from bootstrap predictions
        median_predictions = np.nanmedian(predictions, axis=0)
        ax.plot(
            time_points,
            median_predictions,
            color="#2c7c58",  # Darker green
            linestyle='--',  # Dashed line
            linewidth=2,
            label="Bootstrap Median Trend",
            zorder=60,  # Above confidence region but below main lines
            alpha=0.8
        )
    
    return doubling_times, agent_summary_df


def plot_combined(df, output_file,
                 benchmark_data: pd.DataFrame,
                 params: CombinedPlotParams):
    """
    Main function to generate the combined visualization.
    
    This function creates a plot that overlays:
    1. Multiple benchmark trend lines showing progress on specific tasks
    2. Bootstrap confidence intervals showing overall AI progress uncertainty
    3. Individual model performance points (optional)
    
    Args:
        df: DataFrame with columns: model, benchmark, horizon (seconds), release_date, slope
        output_file: Path where the plot will be saved
        benchmark_data: DataFrame with task length data for percentile calculations
        params: CombinedPlotParams object with all configuration options
    """
    
    # Input validation
    if df.empty:
        print("No data loaded for combined plot.")
        return
    
    if 'release_date' not in df.columns or df['release_date'].isnull().all():
        print("Warning: No valid 'release_date' data found. Skipping combined plot.")
        return
    
    # Prepare data for plotting
    plot_df = df.dropna(subset=['release_date']).copy()
    plot_df = plot_df[plot_df['horizon'] > 0]  # Log scale requires positive values
    
    if plot_df.empty:
        print("No valid data points (with release date and positive horizon) found for combined plot.")
        return
    
    # Convert horizon from seconds to minutes for display
    plot_df['horizon_minutes'] = plot_df['horizon'] / 60.0
    plot_df['log2_horizon_minutes'] = np.log2(plot_df['horizon_minutes'])
    
    # Convert dates to numeric for calculations
    plot_df['release_date_num'] = mdates.date2num(plot_df['release_date'])
    
    # Identify frontier models for each benchmark
    # Frontier = best performing model at each point in time
    plot_df['is_frontier'] = False
    benchmarks = plot_df['benchmark'].unique()
    for bench in benchmarks:
        bench_df = plot_df[plot_df['benchmark'] == bench].sort_values(
            by=['release_date_num', 'horizon_minutes'], 
            ascending=[True, False]
        )
        max_horizon_so_far = -np.inf
        frontier_indices = []
        for index, row in bench_df.iterrows():
            # A model is on the frontier if its horizon exceeds all previous models
            if row['horizon_minutes'] > max_horizon_so_far:
                 frontier_indices.append(index)
                 max_horizon_so_far = row['horizon_minutes']
        if frontier_indices:
            plot_df.loc[frontier_indices, 'is_frontier'] = True
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    texts = []  # Will store text labels for automatic positioning
    
    # Filter benchmarks based on parameters
    if params.hide_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench not in params.hide_benchmarks]
    if params.show_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench in params.show_benchmarks]
    
    # Add bootstrap CI overlay (plotted first so it appears behind benchmark lines)
    if params.show_bootstrap_ci and params.bootstrap_data_file:
        try:
            # Load bootstrap data
            bootstrap_results = pd.read_csv(params.bootstrap_data_file)
            
            # Load release dates
            release_dates_file = 'data/external/release_dates.yaml'
            if os.path.exists(release_dates_file):
                with open(release_dates_file, 'r') as f:
                    release_dates = yaml.safe_load(f)
                
                # Determine date range
                after_date = plot_df['release_date'].min() - pd.Timedelta(days=30)
                max_date = plot_df['release_date'].max() + pd.Timedelta(days=180)
                
                # Get focus agents for later use
                dates = release_dates["date"]
                focus_agents = sorted(list(dates.keys()), key=lambda x: dates[x])
                focus_agents = [agent for agent in focus_agents if agent not in params.exclude_agents]
                
                # Add bootstrap confidence region
                doubling_times, agent_summary_df = add_bootstrap_confidence_region(
                    ax=ax,
                    bootstrap_results=bootstrap_results,
                    release_dates=release_dates,
                    after_date=after_date.strftime('%Y-%m-%d'),
                    max_date=max_date,
                    confidence_level=params.confidence_level,
                    exclude_agents=params.exclude_agents,
                    success_percent=params.success_percent,
                    benchmark_data=benchmark_data,
                )
                
                if doubling_times:
                    lower_bound = np.percentile(doubling_times, 2.5)
                    upper_bound = np.percentile(doubling_times, 97.5)
                    median = np.median(doubling_times)
                    logger.info(
                        f"95% CI for doubling times: [{lower_bound:.0f}, {upper_bound:.0f}] days "
                        f"(+{(upper_bound - median) / median:.0%}/-{(median - lower_bound) / median:.0%})"
                    )
                
                # Optionally add individual agent points on top of CI
                if params.show_individual_agents:
                    agent_texts = []  # List to store text labels for adjustment
                    
                    # First, collect all agent data with dates and values
                    agent_data_list = []
                    for agent in focus_agents:
                        col_name = f"{agent}_p{params.success_percent}"
                        if col_name in bootstrap_results.columns:
                            # Get median value across bootstrap samples
                            agent_values = pd.to_numeric(bootstrap_results[col_name], errors="coerce")
                            median_val = agent_values.median()
                            
                            if not pd.isna(median_val) and median_val > 0:
                                agent_date = pd.to_datetime(dates[agent])
                                agent_data_list.append({
                                    'agent': agent,
                                    'date': agent_date,
                                    'value': median_val
                                })
                    
                    # Sort by date and identify frontier models
                    agent_data_list.sort(key=lambda x: x['date'])
                    max_value_so_far = -np.inf
                    frontier_agents = []
                    
                    for agent_data in agent_data_list:
                        # Only include if this model improves on all previous models
                        if agent_data['value'] > max_value_so_far:
                            frontier_agents.append(agent_data)
                            max_value_so_far = agent_data['value']
                    
                    # Plot only frontier models
                    for agent_data in frontier_agents:
                        # Plot individual model as a point
                        ax.scatter(
                            agent_data['date'],
                            agent_data['value'],
                            color=params.agent_point_color,
                            s=params.agent_point_size,
                            zorder=200,  # On top of everything
                            alpha=0.8,
                            edgecolors='white',
                            linewidth=0.5
                        )
                        # Add model name labels if requested
                        if params.show_model_names:
                            label_name = plotting_aliases.get(agent_data['agent'], agent_data['agent'])
                            text = ax.text(
                                agent_data['date'],
                                agent_data['value'],
                                label_name,
                                fontsize=9,
                                color=params.agent_point_color,
                                ha='center',
                                va='bottom'
                            )
                            agent_texts.append(text)
                    
                    # Adjust label positioning to prevent overlap
                    if agent_texts:
                        adjustText.adjust_text(
                            agent_texts, 
                            ax=ax,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.5)
                        )
                
        except Exception as e:
            print(f"Warning: Could not load bootstrap data: {e}")
    
    # Plot benchmark trend lines (main visualization from lines_over_time.py)
    densely_dotted = (0, (1, 1))  # Line style for dotted portions
    
    for bench in benchmarks:
        # Get data for this benchmark
        bench_data = plot_df[plot_df['benchmark'] == bench]
        color = benchmark_colors[bench]  # Each benchmark has a predefined color
        frontier_data = bench_data[bench_data['is_frontier']]
        non_frontier_data = bench_data[~bench_data['is_frontier']]
        
        # Calculate task length percentiles for dotted line ranges
        # Dotted lines show where we have limited task data (< 2% or > 98%)
        length_data = benchmark_data[benchmark_data['benchmark'] == bench]['length']
        p2 = length_data.quantile(0.02)
        p98 = length_data.quantile(0.98)
        if pd.isna(p2) or pd.isna(p98):
            p2 = 0
            p98 = 10**10
        
        def scatter_points(data, label, **kwargs):
            """Helper function to plot scatter points with markers based on fit quality."""
            if 'slope' in data.columns and len(data) > 0:
                # Use different markers based on logistic fit quality (slope value)
                for idx, row in data.iterrows():
                    slope_val = row['slope']
                    if pd.isna(slope_val):
                        marker = 'o'  # Circle for unknown fit quality
                    elif row['benchmark'] == 'hcast_r_s':
                        marker = '^'  # Special case for HRS benchmark
                    else:
                        # x = poor fit (slope < 0.25), o = unknown, ^ = good fit
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
                # Default to circles if no slope information
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
        
        # Fit and plot trend line using frontier points
        if len(frontier_data) >= 3:  # Need at least 3 points for spline fitting
            # Sort frontier data by release date
            frontier_sorted = frontier_data.sort_values('release_date_num')
            X = frontier_sorted['release_date_num'].values
            Y_log = frontier_sorted['log2_horizon_minutes'].values
            
            # Calculate doubling rate using linear regression in log space
            coeffs = np.polyfit(X, Y_log, 1)
            doubling_rate = coeffs[0] * 365  # Convert from per day to per year
            slope = coeffs[0]  # Store raw slope
            
            # Fit degree-1 spline for smooth visualization
            # k=1 (linear) prevents unrealistic curves
            spline = make_splrep(X, Y_log, s=0.2, k=1)
            
            # Generate smooth line points
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
            
            thick = (bench == "hcast_r_s")
            
            ax.plot(x_line_date[mask_within], y_line[mask_within], color=color, linestyle='-', 
                   linewidth=5 if thick else 2.5, label=f"{benchmark_aliases[bench]}", 
                   zorder=100 if thick else None)
            if params.show_dotted_lines:
                ax.plot(x_line_date[mask_above], y_line[mask_above], color=color, alpha=0.3, 
                       linestyle=densely_dotted, linewidth=2)
                ax.plot(x_line_date[mask_below], y_line[mask_below], color=color, alpha=0.3, 
                       linestyle=densely_dotted, linewidth=2)
            
            # Add text labels for first and last frontier points if requested
            if params.show_model_names and not frontier_data.empty and params.show_points_level >= ShowPointsLevel.FIRST_AND_LAST:
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
    
    # Configure plot appearance
    ax.set_yscale('log')  # Log scale for y-axis (time horizon)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x}'))
    plt.ylim(0.05, 3000)  # Y-axis range in minutes
    
    # Labels and title
    plt.xlabel("Model Release Date")
    plt.ylabel(f"{params.success_percent}% Time Horizon (minutes)")
    plt.title(params.title)
    
    # Grid for readability
    ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.4)
    
    # Format x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45, ha='right')  # Rotate date labels
    
    # Set x-axis bounds if specified
    if params.xbound is not None:
        start_date, end_date = params.xbound
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    if params.ybound is not None:
        ax.set_ylim(params.ybound)
    
    # Create and organize legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Filter out internal labels (those starting with underscore)
    visible_items = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
    if visible_items:
        handles, labels = zip(*visible_items)
        
        # Sort legend: benchmark lines first, then bootstrap items
        sorted_items = []
        bootstrap_items = []
        for h, l in zip(handles, labels):
            if "Bootstrap" in l:
                bootstrap_items.append((h, l))
            else:
                sorted_items.append((h, l))
        
        # Add bootstrap items at the end
        sorted_items.extend(bootstrap_items)
        
        if sorted_items:
            handles, labels = zip(*sorted_items)
            ax.legend(handles, labels, loc='upper left', fontsize=10, frameon=True)
    
    # Adjust text positions to prevent overlap
    if texts:
        adjustText.adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    plt.savefig(output_file, dpi=150)
    print(f"Combined plot saved to {output_file}")
    plt.close(fig)


def process_agent_summaries(exclude_agents: list[str], agent_summaries: pd.DataFrame, 
                          release_dates: dict) -> pd.DataFrame:
    """Process agent summaries to add release dates and filter agents."""
    # Add release dates
    dates = release_dates["date"]
    agent_summaries["release_date"] = agent_summaries["agent"].map(dates)
    
    # Filter out excluded agents
    agent_summaries = agent_summaries[~agent_summaries["agent"].isin(exclude_agents)]
    
    # Drop rows without release dates
    agent_summaries = agent_summaries.dropna(subset=["release_date"])
    
    return agent_summaries


def main():
    """
    Main entry point for generating the combined plot.
    
    This function:
    1. Loads benchmark performance data from all_data.csv
    2. Loads task length data from benchmark TOML files
    3. Checks for bootstrap CI data availability
    4. Configures plot parameters
    5. Generates the combined visualization
    
    Command-line arguments:
        --model-names: Show all model names (red dots + first/last benchmark points)
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate combined benchmark and bootstrap CI plot')
    parser.add_argument('--model-names', action='store_true', 
                        help='Show model names on the plot (for red dots and first/last benchmark points)')
    args = parser.parse_args()
    # Load benchmark performance data (created by wrangle.py)
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
    
    # Load benchmark configuration for task length percentiles
    benchmark_data = get_benchmark_data(BENCHMARKS_PATH)
    
    # Configure bootstrap CI data paths
    bootstrap_data_file = 'data/external/bootstrap/headline.csv'
    release_dates_file = 'data/external/release_dates.yaml'
    
    # Check if bootstrap files exist
    show_bootstrap = os.path.exists(bootstrap_data_file) and os.path.exists(release_dates_file)
    
    if show_bootstrap:
        print(f"Found bootstrap data at {bootstrap_data_file}")
    else:
        print("Bootstrap data not found, plotting benchmarks only")
    
    # Configure plot parameters
    params = CombinedPlotParams(
        # Benchmark settings
        hide_benchmarks=["hcast_r_s", "hcast_r_s_full_method", "video_mme", "gpqa", "aime"], 
        show_points_level=ShowPointsLevel.FRONTIER,  # Show frontier points only
        verbose=False,
        
        # Bootstrap CI settings
        show_bootstrap_ci=show_bootstrap,
        bootstrap_data_file=bootstrap_data_file if show_bootstrap else None,
        agent_summaries_file=None,  # Not needed - we extract from bootstrap data
        confidence_level=0.95,  # 95% confidence interval
        exclude_agents=[],  # Include all agents (GPT-2, GPT-3, etc.)
        success_percent=50,  # Use median (50th percentile)
        
        # Visual settings
        show_individual_agents=True,  # Show red dots for individual models
        show_model_names=args.model_names,  # Show model names based on command-line flag
        title="Time Horizon vs. Release Date (Benchmarks + Bootstrap CI)"
    )
    
    # Generate the combined plot
    plot_combined(all_df.copy(), LINES_PLOT_OUTPUT_FILE, benchmark_data, params)


if __name__ == "__main__":
    main()
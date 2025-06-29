import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
import pandas as pd
import toml
from scipy.interpolate import make_splrep, splev
from enum import Enum
from dataclasses import dataclass, field
from functools import total_ordering
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num, num2date
import yaml
import logging
import pathlib
from typing import List, Dict, Tuple, Optional
import base64
from PIL import Image
import io

from plotting_aliases import benchmark_aliases, plotting_aliases, benchmark_colors

logger = logging.getLogger(__name__)

# Configuration constants
BENCHMARKS_PATH = 'data/benchmarks'
LINES_PLOT_OUTPUT_FILE = 'plots/combined_interactive.html'
STATIC_PLOT_OUTPUT_FILE = 'plots/combined_interactive.png'


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
    trend lines and bootstrap CI overlay.
    """
    # Benchmark line parameters
    show_points_level: ShowPointsLevel
    show_benchmarks: list[str] = field(default_factory=list)
    hide_benchmarks: list[str] = field(default_factory=list)
    show_dotted_lines: bool = False 
    show_doubling_rate: bool = False
    title: str = "Time Horizon vs. Release Date with Bootstrap CI"
    verbose: bool = False
    xbound: tuple[str, str] | None = None
    ybound: tuple[float, float] | None = None
    
    # Bootstrap CI parameters
    show_bootstrap_ci: bool = True
    bootstrap_data_file: str | None = None
    agent_summaries_file: str | None = None
    confidence_level: float = 0.95
    exclude_agents: list[str] = field(default_factory=list)
    success_percent: int = 50
    
    # Individual agent point parameters
    show_individual_agents: bool = False
    agent_point_color: str = "#2c7c58"
    agent_point_size: int = 100
    show_model_names: bool = False
    
    # Output parameters
    output_html: bool = True
    output_static: bool = False


def get_benchmark_data(benchmarks_path) -> pd.DataFrame:
    """
    Loads benchmark task length data from TOML files.
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


def fit_trendline(
    p50s: pd.Series,
    release_dates: pd.Series,
    log_scale: bool = True,
) -> tuple[LinearRegression, float]:
    """
    Fit exponential trendline using linear regression in log space.
    """
    X = date2num(release_dates).reshape(-1, 1)
    
    if log_scale:
        y = np.log(p50s)
    else:
        y = p50s
    
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    
    return reg, score


def create_hover_template(benchmark_name: str, include_doubling: bool = False) -> str:
    """
    Create a custom hover template for benchmark lines.
    """
    template = f"<b>{benchmark_name}</b><br>"
    template += "Date: %{x|%Y-%m-%d}<br>"
    template += "Time: %{y:.2f} minutes<br>"
    if include_doubling:
        template += "Doubling rate: %{customdata:.1f} dbl/yr<br>"
    template += "<extra></extra>"
    return template


def create_point_hover_template(show_model_name: bool = True) -> str:
    """
    Create hover template for individual model points.
    """
    template = ""
    if show_model_name:
        template += "<b>%{customdata[0]}</b><br>"
    template += "Date: %{x|%Y-%m-%d}<br>"
    template += "Time: %{y:.2f} minutes<br>"
    template += "Benchmark: %{customdata[1]}<br>"
    if show_model_name:
        template += "Slope: %{customdata[2]:.2f}<br>"
    template += "<extra></extra>"
    return template


def add_bootstrap_confidence_region(
    fig: go.Figure,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, pd.Timestamp]],
    after_date: str,
    max_date: pd.Timestamp,
    confidence_level: float,
    exclude_agents: list[str],
    success_percent: int = 50,
    visual_start_date: str = None,
) -> tuple[list[float], pd.DataFrame]:
    """
    Add bootstrap confidence intervals and median trend to the plot.
    """
    dates = release_dates["date"]
    focus_agents = sorted(list(dates.keys()), key=lambda x: dates[x])
    focus_agents = [agent for agent in focus_agents if agent not in exclude_agents]
    doubling_times = []
    
    # Collect median performance for each agent
    agent_data = []
    for agent in focus_agents:
        col_name = f"{agent}_p{success_percent}"
        if col_name not in bootstrap_results.columns:
            continue
        
        agent_values = pd.to_numeric(bootstrap_results[col_name], errors="coerce")
        median_val = agent_values.median()
        
        if not pd.isna(median_val) and median_val > 0:
            agent_data.append({
                'agent': agent,
                'release_date': dates[agent],
                f'p{success_percent}': median_val
            })
    
    agent_summary_df = pd.DataFrame(agent_data)
    
    # Calculate confidence bounds
    n_bootstraps = len(bootstrap_results)
    
    visualization_start = pd.to_datetime(visual_start_date) if visual_start_date else pd.to_datetime(after_date)
    time_points = pd.date_range(
        start=visualization_start,
        end=max_date,
        freq="D",
    )
    predictions = np.zeros((n_bootstraps, len(time_points)))
    
    # Fit exponential trend to each bootstrap sample
    valid_samples = 0
    for sample_idx in range(n_bootstraps):
        valid_p50s = []
        valid_dates = []
        
        for agent in focus_agents:
            col_name = f"{agent}_p{success_percent}"
            if col_name not in bootstrap_results.columns:
                continue
                
            p_val = pd.to_numeric(
                bootstrap_results[col_name].iloc[sample_idx], errors="coerce"
            )
            
            if pd.isna(p_val) or np.isinf(p_val) or p_val <= 0:
                continue
                
            valid_p50s.append(p_val)
            valid_dates.append(dates[agent])
        
        if len(valid_p50s) < 2:
            continue
            
        reg, _ = fit_trendline(
            pd.Series(valid_p50s),
            pd.Series(pd.to_datetime(valid_dates)),
            log_scale=True,
        )
        
        time_x = date2num(time_points)
        predictions[valid_samples] = np.exp(reg.predict(time_x.reshape(-1, 1)))
        
        slope = reg.coef_[0]
        doubling_time = np.log(2) / slope
        if doubling_time > 0:
            doubling_times.append(doubling_time)
        valid_samples += 1
    
    predictions = predictions[:valid_samples]
    
    if valid_samples > 0:
        # Calculate confidence bounds
        low_q = (1 - confidence_level) / 2
        high_q = 1 - low_q
        lower_bound = np.nanpercentile(predictions, low_q * 100, axis=0)
        upper_bound = np.nanpercentile(predictions, high_q * 100, axis=0)
        median_predictions = np.nanmedian(predictions, axis=0)
        
        # Add confidence interval as filled area
        fig.add_trace(go.Scatter(
            x=time_points,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(210, 223, 215, 0.4)',
            fill='tonexty',
            name=f'{int(confidence_level*100)}% Bootstrap CI',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Determine solid vs dashed portions
        included_agents_dates = [pd.to_datetime(dates[agent]) for agent in focus_agents 
                                if agent in dates]
        if included_agents_dates:
            data_start = min(included_agents_dates)
            data_end = max(included_agents_dates)
            
            # Create masks
            mask_solid = (time_points >= data_start) & (time_points <= data_end)
            mask_before = time_points < data_start
            mask_after = time_points > data_end
            
            # Plot solid portion
            if np.any(mask_solid):
                fig.add_trace(go.Scatter(
                    x=time_points[mask_solid],
                    y=median_predictions[mask_solid],
                    mode='lines',
                    line=dict(color='#2c7c58', width=2),
                    name='HRS',
                    showlegend=True,
                    hovertemplate="<b>HRS Median</b><br>Date: %{x|%Y-%m-%d}<br>Time: %{y:.2f} minutes<extra></extra>"
                ))
            
            # Plot dashed portions
            if np.any(mask_before):
                fig.add_trace(go.Scatter(
                    x=time_points[mask_before],
                    y=median_predictions[mask_before],
                    mode='lines',
                    line=dict(color='#2c7c58', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            if np.any(mask_after):
                fig.add_trace(go.Scatter(
                    x=time_points[mask_after],
                    y=median_predictions[mask_after],
                    mode='lines',
                    line=dict(color='#2c7c58', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    return doubling_times, agent_summary_df


def plot_combined_plotly(df, output_file, benchmark_data: pd.DataFrame, params: CombinedPlotParams):
    """
    Main function to generate the interactive Plotly visualization.
    """
    
    # Input validation
    if df.empty:
        print("No data loaded for combined plot.")
        return
    
    if 'release_date' not in df.columns or df['release_date'].isnull().all():
        print("Warning: No valid 'release_date' data found. Skipping combined plot.")
        return
    
    # Prepare data
    plot_df = df.dropna(subset=['release_date']).copy()
    plot_df = plot_df[plot_df['horizon'] > 0]
    
    if plot_df.empty:
        print("No valid data points found for combined plot.")
        return
    
    # Convert horizon to minutes
    plot_df['horizon_minutes'] = plot_df['horizon'] / 60.0
    plot_df['log2_horizon_minutes'] = np.log2(plot_df['horizon_minutes'])
    plot_df['release_date_num'] = date2num(plot_df['release_date'])
    
    # Identify frontier models
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
            if row['horizon_minutes'] > max_horizon_so_far:
                 frontier_indices.append(index)
                 max_horizon_so_far = row['horizon_minutes']
        if frontier_indices:
            plot_df.loc[frontier_indices, 'is_frontier'] = True
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Filter benchmarks
    if params.hide_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench not in params.hide_benchmarks]
    if params.show_benchmarks:
        benchmarks = [bench for bench in benchmarks if bench in params.show_benchmarks]
    
    # Add bootstrap CI first (so it appears behind)
    if params.show_bootstrap_ci and params.bootstrap_data_file:
        try:
            bootstrap_results = pd.read_csv(params.bootstrap_data_file)
            release_dates_file = 'data/external/release_dates.yaml'
            
            if os.path.exists(release_dates_file):
                with open(release_dates_file, 'r') as f:
                    release_dates = yaml.safe_load(f)
                
                after_date = pd.Timestamp(params.xbound[0]) if params.xbound else pd.Timestamp("2019-01-01")
                max_date = pd.Timestamp(params.xbound[1]) if params.xbound else plot_df['release_date'].max() + pd.Timedelta(days=180)
                
                dates = release_dates["date"]
                focus_agents = sorted(list(dates.keys()), key=lambda x: dates[x])
                focus_agents = [agent for agent in focus_agents if agent not in params.exclude_agents]
                
                visual_start = params.xbound[0] if params.xbound else "2018-09-03"
                doubling_times, agent_summary_df = add_bootstrap_confidence_region(
                    fig=fig,
                    bootstrap_results=bootstrap_results,
                    release_dates=release_dates,
                    after_date=after_date.strftime('%Y-%m-%d'),
                    max_date=max_date,
                    confidence_level=params.confidence_level,
                    exclude_agents=params.exclude_agents,
                    success_percent=params.success_percent,
                    visual_start_date=visual_start,
                )
                
                # Add individual agent points if requested
                if params.show_individual_agents:
                    agent_data_list = []
                    for agent in focus_agents:
                        col_name = f"{agent}_p{params.success_percent}"
                        if col_name in bootstrap_results.columns:
                            agent_values = pd.to_numeric(bootstrap_results[col_name], errors="coerce")
                            median_val = agent_values.median()
                            
                            if not pd.isna(median_val) and median_val > 0:
                                agent_date = pd.to_datetime(dates[agent])
                                agent_data_list.append({
                                    'agent': agent,
                                    'date': agent_date,
                                    'value': median_val
                                })
                    
                    # Sort and identify frontier
                    agent_data_list.sort(key=lambda x: x['date'])
                    max_value_so_far = -np.inf
                    frontier_agents = []
                    
                    for agent_data in agent_data_list:
                        if agent_data['value'] > max_value_so_far:
                            frontier_agents.append(agent_data)
                            max_value_so_far = agent_data['value']
                    
                    # Plot frontier agent points
                    if frontier_agents:
                        agent_dates = [a['date'] for a in frontier_agents]
                        agent_values = [a['value'] for a in frontier_agents]
                        agent_names = [plotting_aliases.get(a['agent'], a['agent']) for a in frontier_agents]
                        
                        fig.add_trace(go.Scatter(
                            x=agent_dates,
                            y=agent_values,
                            mode='markers' + ('+text' if params.show_model_names else ''),
                            marker=dict(
                                color=params.agent_point_color,
                                size=10,
                                line=dict(color='black', width=0.5)
                            ),
                            text=agent_names if params.show_model_names else None,
                            textposition='top center',
                            customdata=[[name, 'HRS', ''] for name in agent_names],
                            hovertemplate="<b>%{customdata[0]}</b><br>Date: %{x|%Y-%m-%d}<br>Time: %{y:.2f} minutes<br>Dataset: HRS<extra></extra>",
                            name='Individual Models',
                            showlegend=False
                        ))
                
        except Exception as e:
            print(f"Warning: Could not load bootstrap data: {e}")
    
    # Plot benchmark trend lines
    for bench in benchmarks:
        bench_data = plot_df[plot_df['benchmark'] == bench]
        color = benchmark_colors[bench]
        frontier_data = bench_data[bench_data['is_frontier']]
        non_frontier_data = bench_data[~bench_data['is_frontier']]
        
        # Calculate task length percentiles
        length_data = benchmark_data[benchmark_data['benchmark'] == bench]['length']
        p2 = length_data.quantile(0.02)
        p98 = length_data.quantile(0.98)
        if pd.isna(p2) or pd.isna(p98):
            p2 = 0
            p98 = 10**10
        
        # Plot points based on show level
        if params.show_points_level >= ShowPointsLevel.ALL and len(non_frontier_data) > 0:
            # Non-frontier points
            fig.add_trace(go.Scatter(
                x=non_frontier_data['release_date'],
                y=non_frontier_data['horizon_minutes'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.2,
                    line=dict(width=0.5)
                ),
                customdata=np.column_stack((
                    non_frontier_data['model'],
                    [benchmark_aliases[bench]] * len(non_frontier_data),
                    non_frontier_data['slope'].fillna(0)
                )),
                hovertemplate=create_point_hover_template(show_model_name=True),
                showlegend=False,
                name=f'{benchmark_aliases[bench]} (non-frontier)'
            ))
        
        # Plot frontier points
        if params.show_points_level >= ShowPointsLevel.FRONTIER and len(frontier_data) > 0:
            fig.add_trace(go.Scatter(
                x=frontier_data['release_date'],
                y=frontier_data['horizon_minutes'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.9,
                    line=dict(width=0.5),
                    symbol=[
                        'x' if slope < 0.25 else 'circle' if slope == 0.6 else 'triangle-up'
                        for slope in frontier_data['slope'].fillna(0.6)
                    ] if 'slope' in frontier_data.columns else 'circle'
                ),
                customdata=np.column_stack((
                    frontier_data['model'],
                    [benchmark_aliases[bench]] * len(frontier_data),
                    frontier_data['slope'].fillna(0) if 'slope' in frontier_data.columns else [0] * len(frontier_data)
                )),
                hovertemplate=create_point_hover_template(show_model_name=True),
                showlegend=False,
                name=f'{benchmark_aliases[bench]} (frontier)'
            ))
        elif params.show_points_level == ShowPointsLevel.FIRST_AND_LAST and len(frontier_data) > 0:
            # Only first and last points
            frontier_sorted = frontier_data.sort_values('release_date_num')
            selected_data = frontier_sorted.iloc[[0, -1]]
            fig.add_trace(go.Scatter(
                x=selected_data['release_date'],
                y=selected_data['horizon_minutes'],
                mode='markers+text' if params.show_model_names else 'markers',
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.9,
                    line=dict(width=0.5)
                ),
                text=[plotting_aliases.get(m, m) for m in selected_data['model']] if params.show_model_names else None,
                textposition='top center',
                customdata=np.column_stack((
                    selected_data['model'],
                    [benchmark_aliases[bench]] * len(selected_data),
                    selected_data['slope'].fillna(0) if 'slope' in selected_data.columns else [0] * len(selected_data)
                )),
                hovertemplate=create_point_hover_template(show_model_name=True),
                showlegend=False,
                name=f'{benchmark_aliases[bench]} endpoints'
            ))
        
        # Fit and plot trend line
        if len(frontier_data) >= 3:
            frontier_sorted = frontier_data.sort_values('release_date_num')
            X = frontier_sorted['release_date_num'].values
            Y_log = frontier_sorted['log2_horizon_minutes'].values
            
            # Calculate doubling rate
            coeffs = np.polyfit(X, Y_log, 1)
            doubling_rate = coeffs[0] * 365
            
            # Create smooth line using interpolation
            x_smooth = np.linspace(X.min(), X.max(), 200)
            # Use linear spline to match combined.py
            tck = make_splrep(X, Y_log, s=0.2, k=1)
            y_smooth_log = splev(x_smooth, tck)
            y_smooth = 2.0**y_smooth_log
            x_smooth_dates = pd.to_datetime([num2date(x) for x in x_smooth])
            
            # Split line based on p2/p98 bounds
            mask_within = (y_smooth >= p2) & (y_smooth <= p98)
            
            # Main trend line
            customdata = np.full(len(x_smooth_dates[mask_within]), doubling_rate)
            
            # Skip certain benchmarks from having labels
            show_label = bench not in ['mock_aime', 'livecodebench_2411_2505', 'gpqa_diamond', 'webarena']
            
            fig.add_trace(go.Scatter(
                x=x_smooth_dates[mask_within],
                y=y_smooth[mask_within],
                mode='lines',
                line=dict(
                    color=color,
                    width=5 if bench == "hcast_r_s" else 2.5
                ),
                customdata=customdata,
                hovertemplate=create_hover_template(benchmark_aliases[bench], include_doubling=params.show_doubling_rate),
                name=benchmark_aliases[bench],
                showlegend=show_label
            ))
            
            # Dotted lines outside range
            if params.show_dotted_lines:
                mask_outside = ~mask_within
                if np.any(mask_outside):
                    fig.add_trace(go.Scatter(
                        x=x_smooth_dates[mask_outside],
                        y=y_smooth[mask_outside],
                        mode='lines',
                        line=dict(
                            color=color,
                            width=2,
                            dash='dot'
                        ),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text=params.title,
            x=0,
            xanchor='left',
            font=dict(size=20)
        ),
        xaxis=dict(
            title=dict(
                text="Model release date",
                font=dict(size=14)
            ),
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.1)',
            range=[params.xbound[0] if params.xbound else "2018-09-03",
                   params.xbound[1] if params.xbound else "2027-01-01"]
        ),
        yaxis=dict(
            title=dict(
                text="Task length (at 50% success rate)",
                font=dict(size=14)
            ),
            type="log",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.1)',
            range=[np.log10(0.5/60), np.log10(240)],  # 0.5 seconds to 240 minutes
            tickmode='array',
            tickvals=[1/60, 4/60, 15/60, 1, 4, 15, 60, 240],
            ticktext=['1 sec', '4 sec', '15 sec', '1 min', '4 min', '15 min', '1 hr', '4 hrs']
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1200,
        height=600,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    # Add watermark image if available
    logo_path = pathlib.Path("data/external/metr-logo.svg")
    if logo_path.exists():
        # For SVG, we'd need to convert it first. For now, add text watermark
        fig.add_annotation(
            text="METR",
            xref="paper",
            yref="paper",
            x=0.90,
            y=1.05,
            showarrow=False,
            font=dict(size=22, color="black"),
            opacity=0.6
        )
    
    # Add website text
    fig.add_annotation(
        text="metr.org",
        xref="paper",
        yref="paper",
        x=0.90,
        y=-0.05,
        showarrow=False,
        font=dict(size=16, color="#2c7c58"),
        opacity=0.6
    )
    
    fig.add_annotation(
        text="CC-BY",
        xref="paper",
        yref="paper",
        x=0.05,
        y=-0.05,
        showarrow=False,
        font=dict(size=16, color="#2c7c58"),
        opacity=0.6
    )
    
    # Save outputs
    if params.output_html:
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"Interactive plot saved to {output_file}")
    
    if params.output_static:
        fig.write_image(STATIC_PLOT_OUTPUT_FILE, width=1200, height=600, scale=2)
        print(f"Static plot saved to {STATIC_PLOT_OUTPUT_FILE}")


def main():
    """
    Main entry point for generating the interactive combined plot.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate interactive combined benchmark and bootstrap CI plot')
    parser.add_argument('--model-names', action='store_true', 
                        help='Show model names on the plot')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML output')
    parser.add_argument('--static', action='store_true',
                        help='Also generate static PNG output')
    args = parser.parse_args()
    
    # Load data
    data_file = 'data/processed/all_data.csv'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run wrangle.py first.")
        return
    
    all_df = pd.read_csv(data_file)
    all_df['release_date'] = pd.to_datetime(all_df['release_date']).dt.date
    
    if all_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Load benchmark configuration
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
        show_points_level=ShowPointsLevel.FRONTIER,
        verbose=False,
        
        # Bootstrap CI settings
        show_bootstrap_ci=show_bootstrap,
        bootstrap_data_file=bootstrap_data_file if show_bootstrap else None,
        confidence_level=0.95,
        exclude_agents=[],
        success_percent=50,
        
        # Visual settings
        show_individual_agents=True,
        show_model_names=args.model_names,
        title="NEW TITLE",
        xbound=('2018-09-03', '2027-01-01'),
        
        # Output settings
        output_html=not args.no_html,
        output_static=args.static
    )
    
    # Generate the plot
    plot_combined_plotly(all_df.copy(), LINES_PLOT_OUTPUT_FILE, benchmark_data, params)


if __name__ == "__main__":
    main()
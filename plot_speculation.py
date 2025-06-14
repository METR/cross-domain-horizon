import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LineSpec:
    label: str
    y: Callable[[np.ndarray], np.ndarray]
    color: str
    y_start: Optional[float] = None
    y_end: Optional[float] = None
    dotted_start: Optional[float] = None
    dotted_end: Optional[float] = None

@dataclass
class PointSpec:
    x: float
    y: float
    label: str
    color: str
    size: float = 3
    text_size: Optional[float] = 12


def plot_speculation(df: pd.DataFrame, output_file: pathlib.Path):
    """
    Plot based on speculation about other domains.

    The plot should have an x-axis of release date, extending from 1950 to 2050,
    and a y axis of time horizon that's mostly log scale but the top is infinity.

    The specific y axis transform should be:

    y = log2(h / (h + c))

    where y is the displayed y axis value, h is the horizon, and
    c is a constant equal to 1000.

    y ticks represent time horizons in minutes.

    There should be lines on the plot representing:
    - arithmetic
    - chess
    - image recognition
    - HCAST
    - All intellectual labor
    """

    X_START = 1960
    X_END = 2050
    # Create x-axis values (release dates from 1900 to 2050)
    x = np.linspace(X_START, X_END, 1000)
    
    # Define the y-axis transform function
    c = 10000
    def y_transform(h):
        return np.log2(h / (h + c))
    
    # Define the inverse transform for tick labels
    def y_inverse_transform(y):
        return c * (2**y) / (1 - 2**y)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define the lines using LineSpec dataclass
    lines = [
        LineSpec("Arithmetic", lambda x: 2**(0.2*(x - 1900)), 'blue', y_start=1, y_end=np.inf),
        LineSpec("Chess", lambda x: 2**((1/1.7)*(x - 1984)), 'purple', y_start=1, y_end=np.inf,
        dotted_start=0, dotted_end=4*60),
        LineSpec("Image Recognition", lambda x: 2**(x - 2008), 'green', y_start=0, y_end=np.inf, dotted_start=1/60, dotted_end=5),
        LineSpec("HCAST", lambda x: 2**(2*(x - 2021)), 'orange', y_start=0, y_end=np.inf, dotted_start=1, dotted_end=960),
        LineSpec("All Intellectual Labor?", lambda x: 2**(x - 2025), 'red', y_start=0, y_end=np.inf, dotted_start=np.inf, dotted_end=np.inf)
    ]

    points = [
        PointSpec(1986, 1, "Chess", 'purple', 3, 12),
    ]
    
    # Plot each line
    for line_spec in lines:
        h = line_spec.y(x)
        y = y_transform(h)
        
        # Apply y_start and y_end filtering
        visible_mask = np.ones_like(h, dtype=bool)
        if line_spec.y_start is not None:
            visible_mask &= (h >= line_spec.y_start)
        if line_spec.y_end is not None:
            visible_mask &= (h <= line_spec.y_end)
        
        # Only plot if there are visible points
        if not np.any(visible_mask):
            continue
            
        x_visible = x[visible_mask]
        y_visible = y[visible_mask]
        h_visible = h[visible_mask]
        
        if line_spec.dotted_start is not None and line_spec.dotted_end is not None:
            # Plot line with special styling: dotted except between dotted_start and dotted_end
            solid_mask = (h_visible >= line_spec.dotted_start) & (h_visible <= line_spec.dotted_end)
            
            # Plot dotted parts separately (before dotted_start and after dotted_end)
            before_mask = h_visible < line_spec.dotted_start
            after_mask = h_visible > line_spec.dotted_end
            
            if np.any(before_mask):
                plt.plot(x_visible[before_mask], y_visible[before_mask], ':', label=line_spec.label, linewidth=2, color=line_spec.color)
                label_used = True
            else:
                label_used = False
            
            if np.any(after_mask):
                after_label = None if label_used else line_spec.label
                plt.plot(x_visible[after_mask], y_visible[after_mask], ':', label=after_label, linewidth=2, color=line_spec.color)
                label_used = True
            
            # Plot solid part (between dotted_start and dotted_end)
            if np.any(solid_mask):
                solid_label = None if label_used else line_spec.label
                plt.plot(x_visible[solid_mask], y_visible[solid_mask], '-', label=solid_label, linewidth=2, color=line_spec.color)
        else:
            plt.plot(x_visible, y_visible, label=line_spec.label, linewidth=2, color=line_spec.color)
        
        # Store "All intellectual labor?" line data for shading
        if line_spec.label == "All Intellectual Labor?":
            all_labor_x = x_visible
            all_labor_y = y_visible
    
    # Set up the plot
    plt.xlabel('Year')
    plt.ylabel('Time Horizon (minutes)')
    plt.title('Speculation about Automation of Different Domains')
    plt.xlim(X_START, X_END)
    
    # Set up y-axis ticks with time units
    # Define tick values in minutes and their corresponding labels
    tick_data = [
        (1/6, "10 sec"),
        (1, "1 min"),
        (4, "4 min"),
        (15, "15 min"),
        (60, "1 hr"),
        (240, "4 hr"),
        (1440, "1 day"),
        (10080, "1 week"),
    ]
    
    tick_horizons = [h for h, _ in tick_data]
    tick_labels = [label for _, label in tick_data]
    tick_positions = [y_transform(h) for h in tick_horizons]
    
    # Add infinity at the top
    infinity_y = y_transform(1e9)  # Large value to approximate infinity
    tick_positions.append(infinity_y)
    tick_labels.append('∞')
    
    plt.yticks(tick_positions, tick_labels)
    plt.ylim(-15, 0)
    
    # Add red shading below "All intellectual labor?" line
    y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(all_labor_x, all_labor_y, y_min, color='red', alpha=0.1)
    
    
    plt.grid(True, alpha=0.3)
    
    # Add right y-axis for percent automated
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Map the y-axis range to 20%-100% linearly
    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)
    
    # Create evenly spaced percent ticks on the right axis
    percent_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    percent_positions = [y_min + (p - 20) / 80 * (y_max - y_min) for p in percent_values]
    
    ax2.set_yticks(percent_positions)
    ax2.set_yticklabels([f'{p}%' for p in percent_values])
    ax2.set_ylabel('Percent Automated')
    ax2.grid(False)  # Disable grid lines for right axis
    
    # Add grid and legend
    ax1.legend()
    plt.tight_layout()
    
    # Save the plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Speculation plot saved to {output_file}")
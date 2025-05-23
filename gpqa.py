"""
Constructs data/benchmarks/gpqa.toml from the GPQA Hugging Face dataset.

Calculates average expert validation times ('lengths') from the dataset.
Outputs dataset metadata (n_questions, chance_accuracy, lengths) to dataset.toml.

The 'lengths' can then be used by other scripts (e.g., calculate_horizons.py).
Scores are handled externally (e.g., manually in data/gpqa/scores.toml).
"""

import toml
import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Optional

def convert_to_float(value) -> Optional[float]:
    """Convert a value to float, return None if conversion fails."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def calculate_average_times(dataset, column1: str, column2: str) -> List[Optional[float]]:
    """Calculate the average of two time columns from the dataset."""
    times1 = dataset["train"][column1]
    times2 = dataset["train"][column2]
    
    result = []
    for t1, t2 in zip(times1, times2):
        t1_float = convert_to_float(t1)
        t2_float = convert_to_float(t2)
        
        valid_times = [t for t in [t1_float, t2_float] if t is not None]
        if valid_times:
            # Convert numpy float to Python float
            result.append(float(np.mean(valid_times)))
        else:
            result.append(None)
    
    return result

def write_dataset_toml(output_path: Path, n_questions: int, 
                       lengths: List[Optional[float]], chance_accuracy: float) -> None:
    """Write dataset metadata to a TOML file."""
    output_data = {
        "n_questions": n_questions,
        "chance_accuracy": chance_accuracy,
        "lengths": lengths,  # TOML library handles None as null
        "length_type": "baseline",
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            toml.dump(output_data, f)
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")

def main():
    # --- Configuration ---
    DATA_DIR = Path("data/benchmarks")
    DATASET_OUTPUT_FILE = DATA_DIR / "gpqa.toml"
    CHANCE_ACCURACY = 0.25
    # ---------------------

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the GPQA dataset
    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
    print("Dataset loaded.")
    
    # Get the number of questions
    n_questions = len(dataset["train"])
    
    # Calculate average times
    print("Calculating average times...")
    lengths = calculate_average_times(
        dataset, 
        "Self-reported time (minutes)_EV_1", 
        "Self-reported time (minutes)_EV_2"
    )
    print("Average times calculated.")

    # --- Statistical Analysis (optional, for info during run) ---
    s_lengths = pd.Series(lengths).dropna()
    print("\nQuestion length stats (based on calculated lengths):")
    print(s_lengths.describe())
    # Print 10th smallest/largest horizons
    if len(s_lengths) >= 10:
        print(f"10th smallest length: {s_lengths.nsmallest(10).iloc[-1]:.1f} minutes")
        print(f"10th largest length: {s_lengths.nlargest(10).iloc[-1]:.1f} minutes")
    # ----------------------------------------------------------

    # Write dataset metadata output file
    print(f"\nWriting dataset metadata to {DATASET_OUTPUT_FILE}...")
    write_dataset_toml(DATASET_OUTPUT_FILE, n_questions, lengths, CHANCE_ACCURACY)
    
    print(f"\nSuccessfully created {DATASET_OUTPUT_FILE}")

if __name__ == "__main__":
    main()

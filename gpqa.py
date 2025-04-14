"""
Constructs gpqa.toml from the GPQA dataset and the scores in data/gpqa_scores.toml

Average columns "Self-reported time (minutes)_EV_1" and "Self-reported time (minutes)_EV_2"
to get the time per question and put this in a list `lengths`

Scores are in data/gpqa_scores.toml and should go under heading `[scores]`
"""

import toml
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

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

def read_scores(scores_path: Path) -> Dict[str, str]:
    """Read model scores from a TOML file."""
    try:
        with open(scores_path, "r") as f:
            scores_content = f.read()
        
        scores = {}
        for line in scores_content.strip().split("\n"):
            if line and "=" in line:
                model, score = line.split("=", 1)
                scores[model.strip()] = score.strip().strip('"\'')
        return scores
    except Exception as e:
        print(f"Error loading scores file: {e}")
        return {}

def write_toml_file(output_path: Path, n_questions: int, lengths: List[Optional[float]], 
                   scores: Dict[str, str], chance_accuracy: float) -> None:
    """Write the data to a TOML file."""
    with open(output_path, "w") as f:
        f.write(f"n_questions = {n_questions}\n")
        f.write(f"chance_accuracy = {chance_accuracy}\n\n")
        # Write the lengths array
        f.write("lengths = [\n")
        for i, length in enumerate(lengths):
            value = f"  {length}" if length is not None else "  null"
            f.write(value)
            
            if i < len(lengths) - 1:
                f.write(",")
            f.write("\n")
        f.write("]\n\n")
        
        # Write the scores section
        f.write("[scores]\n")
        for model, score in scores.items():
            f.write(f'{model} = "{score}"\n')

def main():
    # Load the GPQA dataset
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
    
    # Get the number of questions
    n_questions = len(dataset["train"])
    
    # Calculate average times
    lengths = calculate_average_times(
        dataset, 
        "Self-reported time (minutes)_EV_1", 
        "Self-reported time (minutes)_EV_2"
    )

    CHANCE_ACCURACY = 0.25

    import pandas as pd

    s_lengths = pd.Series(lengths)
    s_lengths = s_lengths.dropna()

    print(f"Question length stats:")
    print(s_lengths.describe())

    # Print 10th smallest/largest horizons
    print(f"\n10th smallest horizon: {s_lengths.nsmallest(10).iloc[-1]:.1f} minutes")
    print(f"10th largest horizon: {s_lengths.nlargest(10).iloc[-1]:.1f} minutes")

    
    # Read scores
    scores_path = Path("data/gpqa_scores.toml")
    scores = read_scores(scores_path)
    
    # Write output file
    output_path = Path("data/gpqa.toml")
    write_toml_file(output_path, n_questions, lengths, scores, CHANCE_ACCURACY)
    
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    main()

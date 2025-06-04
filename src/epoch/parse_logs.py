# %%

import inspect_ai.log as log
import pandas as pd
import os
from pathlib import Path

rows = []
data_dir = Path("src/epoch/data")

OUTPUT_PARSED_CSV = "src/epoch/data/parsed.csv"

# Walk through all benchmark directories
for benchmark_dir in data_dir.iterdir():
    if not benchmark_dir.is_dir():
        continue
        
    # Find all .eval files in this benchmark directory
    for eval_file in benchmark_dir.glob("*.eval"):
        try:
            eval_log = log.read_eval_log(str(eval_file), header_only=True)
            eval = eval_log.eval

            allowed_reducers = ['mean', None]
            reductions = [r for r in eval_log.reductions if r.reducer in allowed_reducers]
            
            if len(reductions) != 1:
                for reduction in eval_log.reductions:
                    print(reduction.reducer)
                    print('\n\n')
                raise Exception(f"Expected 1 reduction in {eval_file}, got {len(reductions)}")
            reduction = reductions[0]
            
            for sample_score in reduction.samples:
                rows.append({
                    "model": eval.model,
                    "task": eval.task,
                    "sample_id": sample_score.sample_id,
                    "score": sample_score.value,
                    "benchmark": benchmark_dir.name,
                    "run_id": eval_file.stem
                })
        except Exception as e:
            print(f"Error processing {eval_file}: {e}")
            raise e

df = pd.DataFrame(rows)

# %%

with open(OUTPUT_PARSED_CSV, 'w') as f:
    df.to_csv(f, index=False)

# %%
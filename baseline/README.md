# AI Benchmark Baselining Website

A Flask-based web application for running AI benchmarks with human participants to establish baseline performance.

## Features

- Select from available benchmarks (currently supports Hendrycks MATH)
- Displays math problems with rendered LaTeX
- Timer tracks time spent on each question
- Review answers against correct solutions
- Mark answers as correct/incorrect
- Saves results to CSV files in `data/raw/baselines/`

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run the application:
```bash
cd baseline
python run.py
```

3. Open your browser to: http://localhost:5000

## Usage

1. Enter your name and select a benchmark
2. For each question:
   - Read the problem (LaTeX is rendered automatically)
   - Enter your solution in the text area
   - Submit your answer
   - Review your answer against the correct solution
   - Mark whether your answer was correct or incorrect
3. Results are automatically saved to CSV files

## Data Output

Results are saved to `data/raw/baselines/{benchmark_name}.csv` with columns:
- `unique_id`: Unique identifier for each response
- `problem`: The original problem text
- `level`: Difficulty level (if available)
- `subject`: Subject area (if available)
- `answer`: Correct solution from dataset
- `name`: Participant name
- `time_taken`: Time spent on question (seconds)
- `user_answer`: Participant's answer
- `correct`: Whether marked as correct (True/False)

## Adding New Benchmarks

To add a new benchmark, update the `BENCHMARKS` dictionary in `app.py`:

```python
BENCHMARKS = {
    'your-benchmark': {
        'name': 'Your Benchmark Name',
        'dataset': 'huggingface/dataset-name',
        'split': 'test'
    }
}
```

The dataset should have `problem` and `solution` fields. Optional fields include `level` and `type`. 
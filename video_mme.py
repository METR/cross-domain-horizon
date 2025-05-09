MODELS_TO_INCLUDE = {
    "Gemini 1.5 Pro": "google_gemini_1_5_pro_002",
    "Gemini 1.5 Flash": "google_gemini_1_5_flash_002",
    "LLaVA-Video": "llava_v1_14b",
    "GPT-4o": "openai_gpt_4o",
    "Qwen2-VL": "qwen_2_5_vl_72b",
    "GPT-4V": "openai_gpt_4_vision"
}

MODELS_TO_ADD = {
    # https://deepmind.google/technologies/gemini/pro/
    "google_gemini_2_5_pro_exp": 84.8,
}

CHANCE_ACCURACY = 0.25
import os
import csv
import numpy as np

def extract_video_mme_scores():
    scores = {}
    model_scores = {}
    
    with open('data/raw/video_mme.txt', 'r') as f:
        lines = f.readlines()
    
    # Skip header (first 3 lines)
    data_lines = lines[3:]
    
    # Process the data
    i = 0
    while i < len(data_lines):
        # Look for lines that start with a number followed by a tab
        line = data_lines[i].strip()
        
        parts = line.split('\t')

        if len(parts) >= 2:
            model_name = parts[1].strip()
            
        score_line = data_lines[i + 3].strip()
        score_line = score_line.split('\t')
        score = float(score_line[-7]) # 7th from last
        model_scores[model_name] = score
        i += 4
    print(model_scores)
    for model_alias, model_id in MODELS_TO_INCLUDE.items():
        if model_alias in model_scores:
            scores[model_id] = model_scores[model_alias]
        else:
            raise ValueError(f"Could not find a match for {model_alias}")
    
    return scores

def generate_toml_file(scores):
    os.makedirs('data/scores', exist_ok=True)
    
    with open('data/scores/video_mme.toml', 'w') as f:
        f.write("# VideoMME benchmark scores (%)\n")
        f.write("# Note: Scores represent overall performance with subtitles\n")
        f.write("source = \"https://video-mme.github.io/\"\n\n")
        
        f.write("[scores]\n")
        for model_id, score in scores.items():
            f.write(f"{model_id} = \"{score}%\"\n")


def generate_benchmark_file(output_filename="data/benchmarks/video_mme.toml", input_csv_path="data/raw/video_mme_duration.csv"):
    """
    Generates a TOML benchmark definition file for VideoMME.

    Reads duration intervals and question counts from a CSV file,
    calculates the total number of questions, and generates a list of
    video lengths assuming questions are evenly distributed within each interval.
    The output is a TOML file similar to gpqa.toml.
    """
    all_lengths = []
    total_questions = 0

    with open(input_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            min_duration_s = int(row['min_duration_s'])
            max_duration_s = int(row['max_duration_s'])
            count = int(row['counts'])

            total_questions += count

            if count == 0:
                continue
            if count == 1:
                all_lengths.append((min_duration_s + max_duration_s) / 2.0)
            else:
                # Distribute 'count' points evenly within the interval (min_duration_s, max_duration_s]
                # We use linspace from min_duration_s + step/2 to max_duration_s - step/2
                # where step = (max_duration_s - min_duration_s) / count
                # This ensures points are centered within their sub-intervals.
                step = (max_duration_s - min_duration_s) / count
                interval_lengths = np.linspace(min_duration_s + step / 2.0, max_duration_s - step / 2.0, count)
                all_lengths.extend(interval_lengths.tolist())
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w') as f:
        f.write(f"n_questions = {total_questions}\n")
        f.write(f"chance_accuracy = {CHANCE_ACCURACY}\n")
        # Format lengths similar to the gpqa.toml example
        lengths_str = ", ".join(f"{length:.1f}" for length in all_lengths)
        f.write(f"lengths = [ {lengths_str}, ]\n")
    
    print(f"Generated benchmark file: {output_filename} with {total_questions} questions.")

if __name__ == "__main__":
    scores = extract_video_mme_scores()
    scores.update(MODELS_TO_ADD)
    generate_toml_file(scores)
    print(f"\nGenerated TOML file with scores for {len(scores)} models")
    generate_benchmark_file()


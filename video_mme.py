import os
import csv
import numpy as np
import toml

MODELS_TO_INCLUDE = {
    "Gemini 1.5 Pro": "google_gemini_1_5_pro_002",
    "Gemini 1.5 Flash": "google_gemini_1_5_flash_002",
    "LLaVA-Video": "bytedance_llava_video_72b_qwen2",
    "GPT-4o": "openai_gpt_4o",
    "Qwen2-VL": "qwen_2_5_vl_72b",
    "GPT-4V": "openai_gpt_4_vision"
}

MODELS_TO_ADD = {
    # https://deepmind.google/technologies/gemini/pro/
    "google_gemini_2_5_pro_exp": 84.8,
}

SPLIT_NAMES = ["short", "med", "long"]

CHANCE_ACCURACY = 0.25

def extract_video_mme_scores():
    scores = {split_name: {} for split_name in SPLIT_NAMES + ["all"]}
    
    with open('data/raw/video_mme.txt', 'r') as f:
        lines = f.readlines()

    reader = csv.reader(lines, delimiter='\t')
    
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
        score = map(float, (score_line[-7], score_line[-5], score_line[-3], score_line[-1]))
        for split_name, score in zip(["all"] + SPLIT_NAMES, score):
            scores[split_name][MODELS_TO_INCLUDE.get(model_name, model_name)] = score
        i += 4

    return scores

def generate_toml_file(scores):
    os.makedirs('data/scores', exist_ok=True)

    extracted_scores = extract_video_mme_scores()

    data = {
        "source": "https://video-mme.github.io/",
        "splits": extracted_scores,
    }
    
    with open('data/scores/video_mme.toml', 'w') as f:
        toml.dump(data, f)
    print(f"\nGenerated TOML file with scores for {len(scores['short'])} models")


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

    def split(duration_mins):
        if duration_mins < 2:
            return "short"
        elif duration_mins < 15:
            return "med"
        else:
            return "long"

    with open(input_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert seconds to minutes
            min_duration_mins = int(row['min_duration_s']) / 60
            max_duration_mins = int(row['max_duration_s']) / 60
            count = int(row['counts'])

            total_questions += count

            if count == 0:
                continue
            if count == 1:
                all_lengths.append((min_duration_mins + max_duration_mins) / 2.0)
            else:
                # Distribute 'count' points evenly within the interval (min_duration_s, max_duration_s]
                # We use linspace from min_duration_s + step/2 to max_duration_s - step/2
                # where step = (max_duration_s - min_duration_s) / count
                # This ensures points are centered within their sub-intervals.
                step = (max_duration_mins - min_duration_mins) / count
                interval_lengths = np.linspace(min_duration_mins + step / 2.0, max_duration_mins - step / 2.0, count)
                all_lengths.extend(interval_lengths.tolist())
    
    splits = {split_name: dict(lengths=[]) for split_name in SPLIT_NAMES}
    for length in all_lengths:
        splits[split(length)]["lengths"].append(round(length, 3))

    toml_data = {
        "n_questions": total_questions,
        "chance_accuracy": CHANCE_ACCURACY,
        "length_type": "estimate",
        "splits": splits,
    }
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    toml.dump(toml_data, open(output_filename, 'w'))
    
    print(f"Generated benchmark file: {output_filename} with {total_questions} questions.")

if __name__ == "__main__":
    scores = extract_video_mme_scores()
    scores.update(MODELS_TO_ADD)
    generate_toml_file(scores)
    generate_benchmark_file()


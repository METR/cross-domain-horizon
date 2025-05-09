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
    "google_gemini_2_5_pro_exp": 0.848,
}
import os

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
        if line and line[0].isdigit() and '\t' in line:
            parts = line.split('\t')
            
            if len(parts) >= 2:
                # Extract full model name from the line with the rank number
                rank = parts[0]
                model_name_part = parts[1].strip()
                
            score_line = data_lines[i + 3].strip()
            score_line = score_line.split('\t')
            score = float(score_line[-7]) # 7th from last
            model_scores[model_name_part] = score
            i += 4
    
    # Match our target models to the extracted model names
    for target_model, model_id in MODELS_TO_INCLUDE.items():
        found = False
        
        # Try to find an exact match first
        for model_name, model_score in model_scores.items():
            if target_model in model_name:
                scores[model_id] = f"{model_score}%"
                print(f"Matched {target_model} to {model_name} with score {model_score}")
                found = True
                break
                
        if not found:
            raise ValueError(f"Could not find a match for {target_model}")
    
    return scores

def generate_toml_file(scores):
    os.makedirs('data/scores', exist_ok=True)
    
    with open('data/scores/video_mme.toml', 'w') as f:
        f.write("# VideoMME benchmark scores (%)\n")
        f.write("# Note: Scores represent overall performance with subtitles\n")
        f.write("source = \"https://video-mme.github.io/\"\n\n")
        
        f.write("[scores]\n")
        for model_id, score in scores.items():
            f.write(f"{model_id} = \"{score}\"\n")

if __name__ == "__main__":
    scores = extract_video_mme_scores()
    scores.update(MODELS_TO_ADD)
    generate_toml_file(scores)
    print(f"\nGenerated TOML file with scores for {len(scores)} models")


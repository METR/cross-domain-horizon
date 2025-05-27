from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datasets import load_dataset
import pandas as pd
import os
import time
import uuid
import random
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Ensure data directories exist - path relative to project root
data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'baselines'
data_dir.mkdir(parents=True, exist_ok=True)

# Available benchmarks
BENCHMARKS = {
    'hendrycks-math': {
        'name': 'Hendrycks MATH',
        'dataset': 'nlile/hendrycks-MATH-benchmark',
        'split': 'test'
    }
}

@app.route('/')
def index():
    return render_template('index.html', benchmarks=BENCHMARKS)

@app.route('/start', methods=['POST'])
def start_benchmark():
    print("=== START BENCHMARK ROUTE CALLED ===")
    benchmark = request.form['benchmark']
    name = request.form['name']
    level = request.form.get('level', '')  # Get level, default to empty string
    print(f"Received benchmark: {benchmark}, name: {name}, level: {level}")
    
    if not benchmark or not name:
        print("Missing benchmark or name, redirecting to index")
        return redirect(url_for('index'))
    
    try:
        print(f"Loading dataset: {BENCHMARKS[benchmark]['dataset']}")
        # Load dataset to verify it works and filter by level if specified
        ds = load_dataset(BENCHMARKS[benchmark]['dataset'])
        split = BENCHMARKS[benchmark]['split']
        
        # Filter by level if specified
        if level:
            level_int = int(level)
            filtered_indices = [i for i, item in enumerate(ds[split]) if item.get('level') == level_int]
            print(f"Filtered to level {level}: {len(filtered_indices)} questions out of {len(ds[split])}")
        else:
            filtered_indices = list(range(len(ds[split])))
            print(f"Using all levels: {len(filtered_indices)} questions")
        
        if not filtered_indices:
            return f"No questions found for level {level}", 400
        
        dataset_length = len(filtered_indices)
        print(f"Dataset loaded successfully, using split: {split}, length: {dataset_length}")
        
        # Store only essential info in session (not the entire dataset)
        session['benchmark'] = benchmark
        session['name'] = name
        session['level'] = level
        session['dataset_length'] = dataset_length
        session['current_question'] = 0
        session['start_time'] = None
        
        # Create a shuffled list of question indices for random selection
        question_indices = filtered_indices.copy()
        random.shuffle(question_indices)
        session['question_indices'] = question_indices
        
        print(f"Session data stored, dataset has {dataset_length} questions")
        print("Redirecting to question route")
        return redirect(url_for('question'))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return f"Error loading dataset: {e}", 500

@app.route('/question')
def question():
    print("=== QUESTION ROUTE CALLED ===")
    if 'benchmark' not in session or 'dataset_length' not in session or 'question_indices' not in session:
        print("No benchmark info in session, redirecting to index")
        return redirect(url_for('index'))
    
    current_idx = session['current_question']
    dataset_length = session['dataset_length']
    question_indices = session['question_indices']
    benchmark = session['benchmark']
    print(f"Current question index: {current_idx}, dataset length: {dataset_length}")
    
    if current_idx >= dataset_length:
        print("All questions completed, showing complete page")
        return render_template('complete.html')
    
    # Get the random question index
    random_question_idx = question_indices[current_idx]
    print(f"Using random question index: {random_question_idx}")
    
    # Load dataset and get current question
    try:
        ds = load_dataset(BENCHMARKS[benchmark]['dataset'])
        split = BENCHMARKS[benchmark]['split']
        question_data = ds[split][random_question_idx]
        session['start_time'] = time.time()
        print(f"Displaying question {current_idx + 1}, question keys: {list(question_data.keys())}")
        
        return render_template('question.html', 
                             question=question_data,
                             question_num=current_idx + 1,
                             total_questions=dataset_length)
    except Exception as e:
        print(f"Error loading question: {e}")
        return f"Error loading question: {e}", 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'benchmark' not in session:
        return redirect(url_for('index'))
    
    user_answer = request.form['answer']
    time_taken = time.time() - session['start_time']
    
    current_idx = session['current_question']
    question_indices = session['question_indices']
    random_question_idx = question_indices[current_idx]
    benchmark = session['benchmark']
    
    # Load current question data
    ds = load_dataset(BENCHMARKS[benchmark]['dataset'])
    split = BENCHMARKS[benchmark]['split']
    question_data = ds[split][random_question_idx]
    
    # Store the answer and time for the next step
    session['user_answer'] = user_answer
    session['time_taken'] = time_taken
    session['question_data'] = question_data
    
    return render_template('review.html',
                         question=question_data,
                         user_answer=user_answer,
                         time_taken=round(time_taken, 2))

@app.route('/mark_result', methods=['POST'])
def mark_result():
    if 'question_data' not in session:
        return redirect(url_for('index'))
    
    result = request.form['result']
    
    # Only save to CSV if not skipping
    if result != 'skip':
        is_correct = result == 'correct'
        
        # Save to CSV
        save_result(
            session['question_data'],
            session['name'],
            session['time_taken'],
            session['user_answer'],
            is_correct,
            session['benchmark']
        )
    
    # Move to next question
    session['current_question'] += 1
    
    # Clean up temporary session data
    session.pop('user_answer', None)
    session.pop('time_taken', None)
    session.pop('question_data', None)
    
    return redirect(url_for('question'))

def save_result(question_data, name, time_taken, user_answer, is_correct, benchmark):
    # Create unique ID
    unique_id = str(uuid.uuid4())
    
    # Prepare data row
    row_data = {
        'unique_id': unique_id,
        'problem': question_data['problem'],
        'level': question_data.get('level', ''),
        'subject': question_data.get('subject', ''),
        'answer': question_data['answer'],
        'name': name,
        'time_taken_seconds': time_taken,
        'user_answer': user_answer,
        'correct': is_correct
    }
    
    # Save to CSV - use absolute path relative to project root
    csv_path = Path(__file__).parent.parent / 'data' / 'raw' / 'baselines' / f'{benchmark}.csv'
    
    # Create DataFrame
    df = pd.DataFrame([row_data])
    
    # Append to existing file or create new one
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    app.run(debug=True) 
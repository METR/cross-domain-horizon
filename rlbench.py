import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import time
import json

def extract_json_from_script_tag(soup, script_id):
    """
    Extract JSON data from script tag by ID
    """
    script_tag = soup.find('script', id=script_id)
    if script_tag:
        try:
            json_data = json.loads(script_tag.string)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {script_id}: {e}")
    return None

def extract_paper_date(paper_url, headers):
    """
    Extract publication date from individual paper page
    """
    if paper_url == 'N/A' or not paper_url:
        return 'Unknown'
    
    try:
        print(f"Fetching date from: {paper_url}")
        response = requests.get(paper_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for date in author-span elements
        author_spans = soup.find_all('span', class_='author-span')
        
        for span in author_spans:
            text = span.get_text().strip()
            # Look for date pattern like "27 May 2025"
            date_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', text)
            if date_match:
                try:
                    date_str = date_match.group(1)
                    print("Found date:", date_str)
                    parsed_date = datetime.strptime(date_str, '%d %b %Y')
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return 'Unknown'
        
    except Exception as e:
        print(f"Error extracting date from {paper_url}: {e}")
        return 'Unknown'

def scrape_rlbench_leaderboard():
    """
    Scrape RLBench leaderboard from Papers with Code website
    """
    print("Fetching RLBench leaderboard data from Papers with Code...")
    
    url = "https://paperswithcode.com/sota/robot-manipulation-on-rlbench"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data from JSON script tags
        table_data = extract_json_from_script_tag(soup, 'evaluation-table-data')
        
        if not table_data:
            raise ValueError("No leaderboard data found in JSON")
        
        print(f"Found {len(table_data)} entries in JSON data")
        
        results = []
        for i, entry in enumerate(table_data):
            print(f"Processing entry {i+1}/{len(table_data)}")
            
            method = entry.get('method', 'Unknown')
            
            # Extract the main success rate metric
            metrics = entry.get('metrics', {})
            score = metrics.get('Succ. Rate (18 tasks, 100 demo/task)', 'N/A')
            
            # Extract paper information
            paper_info = entry.get('paper', {})
            paper_title = paper_info.get('title', 'Unknown')
            paper_url = paper_info.get('url', '')
            
            # Construct full paper link
            if paper_url and not paper_url.startswith('http'):
                paper_link = f"https://paperswithcode.com{paper_url}"
            else:
                paper_link = paper_url if paper_url else 'N/A'
            
            # Extract publication date from the paper page
            date = extract_paper_date(paper_link, headers)
            
            # Add a small delay to be respectful to the server
            time.sleep(0.2)
            
            results.append({
                'score': score,
                'date': date,
                'model': method,
                'paper_link': paper_link
            })
        
        df = pd.DataFrame(results)
        return df
            
    except Exception as e:
        print(f"Failed to fetch RLBench leaderboard data: {e}")
        return None

def main():
    """
    Main function to fetch RLBench leaderboard data
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nofetch', action='store_true', help='Skip fetching new data')
    args = parser.parse_args()

    output_path = 'data/external/rlbench_leaderboard.csv'
    
    if not args.nofetch:
        print("Fetching RLBench leaderboard data...")
        df = scrape_rlbench_leaderboard()
        
        if df is not None:
            # Clean up None values and ensure proper formatting
            df['score'] = df['score'].fillna('N/A')
            df = df[df['score'] != 'None']  # Remove rows where score is 'None' string
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} entries to {output_path}")
        else:
            print("Failed to fetch RLBench leaderboard data")
            return
    else:
        print("Skipping fetch, loading existing data...")
        try:
            df = pd.read_csv(output_path)
        except FileNotFoundError:
            print(f"Error: No existing data found at {output_path}")
            return

    print(f"Leaderboard preview:\n{df.head()}")

if __name__ == "__main__":
    main()

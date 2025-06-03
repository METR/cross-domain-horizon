import pandas as pd
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

benchmarks_df = pd.read_csv('src/epoch/benchmarks_tasks.csv')
print(benchmarks_df.head())

EPOCH_URL_TEMPLATE = "https://logs.epoch.ai/inspect_ai_logs/{}.eval"

print("Starting download of all eval files...")

# Create data directory if it doesn't exist
os.makedirs('src/epoch/data', exist_ok=True)

aws_waf_token = os.getenv('AWS_WAF_TOKEN')

# Set up browser-like headers with working authentication
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Cookie': f'aws-waf-token={aws_waf_token}',
    'Connection': 'keep-alive',
    'Sec-Ch-Ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1'
}

total_files = 0
downloaded = 0
failed = 0

benchmarks_to_skip = ['MATH level 5', 'FrontierMath-2025-02-28-Private']

for index, row in benchmarks_df.iterrows():
    benchmark_name = row['Name']
    
    # Skip MATH benchmark as it seems to have permission issues
    if benchmark_name in benchmarks_to_skip:
        print(f"⏭️  Skipping {benchmark_name} (known permission issues)")
        continue
        
    print(f"\n{'='*50}")
    print(f"Processing benchmark: {benchmark_name}")
    print(f"{'='*50}")
    
    runs = row["BenchmarkRuns"].split(",")
    total_files += len(runs)
    
    for run_id in runs:
        run_id = run_id.strip()
        url = EPOCH_URL_TEMPLATE.format(run_id)
        filepath = f'src/epoch/data/{benchmark_name}/{run_id}.eval'
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"⏭️  Skipping {run_id} (already exists)")
            downloaded += 1
            continue
            
        print(f"📥 Downloading {run_id}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=60)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size_mb = len(response.content) / (1024 * 1024)
                print(f"✅ Downloaded {run_id} ({file_size_mb:.1f} MB)")
                downloaded += 1
                
            else:
                print(f"❌ Failed {run_id}: Status {response.status_code}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error downloading {run_id}: {e}")
            failed += 1
            
        # Small delay to be respectful to the server
        time.sleep(0.2)

print(f"\n{'='*50}")
print(f"DOWNLOAD SUMMARY")
print(f"{'='*50}")
print(f"Total files: {total_files}")
print(f"Downloaded: {downloaded}")
print(f"Failed: {failed}")
print(f"Success rate: {downloaded/total_files*100:.1f}%")








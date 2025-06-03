import pandas as pd

benchmarks_df = pd.read_csv('src/epoch/benchmarks_tasks.csv')
print(benchmarks_df.head())

EPOCH_URL_TEMPLATE = "https://logs.epoch.ai/inspect_ai_logs/{}.eval"

for index, row in benchmarks_df.iterrows():
    for run in row["BenchmarkRuns"].split(","):
        url = EPOCH_URL_TEMPLATE.format(run)
        print(url)
        import requests
        import os

        # Create data directory if it doesn't exist
        os.makedirs('src/epoch/data', exist_ok=True)

        # Download the log file
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'src/epoch/data/{run}.json', 'w') as f:
                f.write(response.text)
            print(f"Downloaded {run}.json")
            break  # Exit after first successful download
        else:
            print(f"Failed to download {run}: {response.status_code}")








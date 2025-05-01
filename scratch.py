import yaml

with open('data/raw/release_dates.yaml', 'r') as file:
    data = yaml.safe_load(file)


out_dict = {}
for model, date in data['model_info'].items():
    out_dict[model] = {
        'release_date': date,
    }

with open('data/raw/model_info.yaml', 'w') as file:
    yaml.dump(out_dict, file)
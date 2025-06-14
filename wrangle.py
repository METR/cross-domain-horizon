import pandas as pd
import os
import glob
import re
from scipy.stats import gmean
import yaml
from pydantic import BaseModel, Field, validator
from typing import Union, Dict, List, Optional, Literal
import datetime

DATA_DIR = 'data'
HORIZONS_DIR = os.path.join(DATA_DIR, 'horizons') # New constant for horizons dir
RELEASE_DATES_FILE = os.path.join(DATA_DIR, 'raw', 'model_info.yaml') # Path to release dates
EPOCH_RELEASE_DATES_FILE = os.path.join('src', 'epoch', 'data','model_versions.csv') # Path to epoch release dates
MIN_HORIZON_THRESHOLD_SECONDS = 10 * 60  # 10 minutes


class ModelInfo(BaseModel):
    release_date: datetime.date | Literal['unknown']
    aliases: List[str] = Field(default_factory=list)

    @classmethod
    def from_yaml_data(cls, data: Union[str, Dict]) -> 'ModelInfo':
        """Create ModelInfo from YAML data that can be either a string or dict."""
        assert isinstance(data, dict)
        return cls(
            release_date=data.get('release_date', 'unknown'),
            aliases=data.get('aliases', [])
        )


class ModelInfoCollection(BaseModel):
    model_info: Dict[str, Dict]

    def get_validated_models(self) -> Dict[str, ModelInfo]:
        """Validate and return all model info as ModelInfo objects."""
        validated_models = {}
        for model_name, model_data in self.model_info.items():
            try:
                validated_models[model_name] = ModelInfo.from_yaml_data(model_data)
            except Exception as e:
                print(f"Warning: Failed to validate model {model_name}: {e}")
        return validated_models

def normalize_model_name(model_name):
    """Normalizes model names to lowercase and replaces non-alphanumeric characters with underscores."""

    prefixes_to_remove = [
        "together_",
        "fireworks_accounts_fireworks_models_",
        "sagemaker_",
        "hyperbolic_",
        "anthropic_",
        "openai_",
        "mistralai_",
        "deepseek_ai_",
        "google_",
        "microsoft_",
    ]

    # Remove part before first slash if there is a slash
    if '/' in model_name:
        model_name = model_name.split('/', 1)[1]

    # tesla fsd versions
    if re.match(r'\d+\.\d+\.(x|\d+)?', model_name):
        return model_name
    model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name.lower())

    for prefix in prefixes_to_remove:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]

    return model_name


def load_release_dates(release_dates_files: list[str]) -> pd.DataFrame:
    """Loads model release dates and handles aliases from multiple sources using Pydantic validation."""
    
    # If a single file is passed, convert to list for consistency
    if isinstance(release_dates_files, str):
        release_dates_files = [release_dates_files]
    
    combined_model_info = {}
    
    for file_path in release_dates_files:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml'):
                file_data_unnormalized = yaml.safe_load(f)
                assert 'model_info' in file_data_unnormalized
                file_data = {'model_info': {}}
                for model_name, model_data in file_data_unnormalized['model_info'].items():
                    normalized_name = normalize_model_name(model_name)
                    file_data['model_info'][normalized_name] = model_data
            else:
                assert file_path.endswith('.csv')
                file_data = {'model_info': {}}
                csv_df = pd.read_csv(file_path)
                # Filter out rows where 'id' is null/empty
                csv_df = csv_df.dropna(subset=['id'])
                csv_df = csv_df[csv_df['id'].str.strip() != '']  # Remove rows with empty strings
                
                for _, row in csv_df.iterrows():
                    try:
                        release_date = datetime.date.fromisoformat(row['Version release date'])
                    except (ValueError, TypeError):
                        release_date = 'unknown'

                    normalized_name = normalize_model_name(row['id'])
                    file_data['model_info'][normalized_name] = {
                        'release_date': release_date,
                        'aliases': []
                    }
            
        for k in file_data["model_info"].keys():
            assert ("anthropic" not in k), (file_path, k)
        # Merge model_info, with earlier files taking priority
        for model_name, model_data in file_data.get('model_info', {}).items():
            if model_name not in combined_model_info:
                combined_model_info[model_name] = model_data
    
    # Create the final release_data structure
    print(f"Combined model info for {len(combined_model_info)} models")
    release_data = {'model_info': combined_model_info}
    
    try:
        # Validate the combined data using Pydantic
        try:
            model_collection = ModelInfoCollection(**release_data)
        except Exception as e:
            error_msg = f"Error: Failed to validate combined YAML structure: {e}. Cannot proceed."
            print(error_msg)
            raise ValueError(error_msg)
            
        # Get validated model info
        validated_models = model_collection.get_validated_models()
        debug_file_path = os.path.join(DATA_DIR, 'interim', 'validated_model_info.yaml')
        os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
        with open(debug_file_path, 'w') as f:
            yaml.dump({'model_info': {name: info.dict() for name, info in validated_models.items()}}, f, default_flow_style=False, sort_keys=False)
        
        print(f"Validated model info written to {debug_file_path}")
        
        if not validated_models:
            error_msg = f"Error: No valid model info found after combining files. Cannot proceed."
            print(error_msg)
            raise ValueError(error_msg)

        all_releases = []
        for model_name, model_info in validated_models.items():
            if model_info.release_date:
                # Add the main model
                all_releases.append({'model': normalize_model_name(model_name), 'release_date': model_info.release_date})
                # Add aliases with the same release date
                for alias in model_info.aliases:
                    normalized_alias = normalize_model_name(alias)
                    if normalized_alias not in validated_models:
                        all_releases.append({'model': normalized_alias, 'release_date': model_info.release_date})
            else:
                print(f"Warning: Missing or unknown release_date for model: {model_name}")

        if not all_releases:
            raise ValueError(f"Warning: No release dates found after processing all files")

        release_df = pd.DataFrame(all_releases)
        # Attempt to convert 'release_date' to datetime, handling potential 'unknown' values
        release_df['release_date'] = pd.to_datetime(release_df['release_date'], errors='coerce')

        # Check for duplicates before processing
        duplicate_check = release_df.duplicated(subset=['model'], keep=False)
        if duplicate_check.any():
            duplicates = release_df[duplicate_check].sort_values(['model', 'release_date'])
            error_msg = f"Found duplicate models:\n{duplicates.to_string()}"
            print(error_msg)
            raise ValueError(error_msg)

        return release_df

    except Exception as e:
        error_msg = f"An unexpected error occurred while processing combined release dates: {e}. Cannot proceed."
        print(error_msg)
        raise e


def load_data(data_dir):
    """Loads horizon data and merges release dates."""
    # --- Load Horizon Data ---
    all_data = []
    horizons_dir = os.path.join(data_dir, 'horizons') # Use the specific horizons directory
    csv_files = glob.glob(os.path.join(horizons_dir, '*.csv')) # Find all CSVs in horizons dir

    if not csv_files:
        raise ValueError(f"No CSV files found in {horizons_dir}")

    for csv_path in csv_files:
        benchmark_name = os.path.basename(csv_path).replace('.csv', '') # Extract benchmark from filename
        print(f"Loading {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            columns_to_use = ['model', 'horizon', 'slope', 'slope_method', 'release_date', 'benchmark', "score"]

            for col in columns_to_use:
                if col not in df.columns:
                    df[col] = None

            df['horizon'] = df['horizon'].fillna(0)
            # Convert horizon from minutes to seconds
            df['horizon'] = df['horizon'] * 60
            df['benchmark'] = benchmark_name

            all_data.append(df[columns_to_use])

        except Exception as e:
            print(f"Warning: Could not read {csv_path}. Error: {e}")


    if not all_data:
        print("Warning: No data loaded after processing CSV files.")
        # Return empty DataFrame with all expected columns
        raise ValueError("No data loaded after processing CSV files.")

    horizon_df = pd.concat(all_data, ignore_index=True)
    horizon_df['model'] = horizon_df['model'].apply(normalize_model_name)

    # --- Load and Merge Release Dates ---
    release_df = load_release_dates([RELEASE_DATES_FILE, EPOCH_RELEASE_DATES_FILE])
    print(release_df.tail())

    # Merge with horizon data
    merged_df = pd.merge(horizon_df, release_df, on='model', how='left')
    merged_df['release_date'] = merged_df['release_date_x'].fillna(merged_df['release_date_y'])
    merged_df = merged_df.drop(columns=['release_date_x', 'release_date_y'])
    
    # Convert release dates to yyyy-mm-dd format (strip time component)
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date']).dt.date
    
    print(f"Number of models with release dates: {merged_df['release_date'].count()} / {len(merged_df)}")

    # Check for models without release dates after merge
    missing_dates = merged_df[merged_df['release_date'].isna()]['model'].unique()
    if len(missing_dates) > 0:
        
        # Filter out models that genuinely had 'unknown' or unparseable dates in the YAML
        # We only want to warn about models present in horizon data but *completely missing* from release info
        models_in_release_info = set(release_df['model'])
        truly_missing = [m for m in missing_dates if m not in models_in_release_info]
        if truly_missing:
            print(f"Warning: Missing release dates for models present in horizon data but not found in {RELEASE_DATES_FILE}: {list(truly_missing)}")

    return merged_df


 

if __name__ == "__main__":
    out_file = os.path.join(DATA_DIR, 'processed', 'all_data.csv')
    df = load_data(DATA_DIR)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Data written to {out_file}")

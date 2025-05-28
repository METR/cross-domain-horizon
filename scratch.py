# %%

import toml
import numpy as np
import pandas as pd

# Read the TOML file
with open("data/all_data.csv", "r") as f:
    df = pd.read_csv(f)

# %%

df.head()

# %%

aime_models = df[df["aime"].notna()]

hendrycks_models = df[df["hendrycks_math"].notna()]
# %%
aime_models.head()

# %%

hendrycks_models.head()

# %%
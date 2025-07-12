# cross-domain-horizon
Estimate the time horizon of AIs over time on various domains like knowledge and vision


### Methodology

- Only the frontier is used to fit the regression line
- Only models that get between 10% and 90% on some benchmark are counted
- The best agent is used on each model

### Usage

First run

```
pip install requirements.txt --no-deps
```

* Run benchmark-specific `.py` files to load scores for each dataset. Some also calculate horizons.

* Run `calculate_horizons.py` to estimate time horizons for each dataset

* Run `plots.py` to make all plots

To run the combined plot, you'll need cairosvg, see `https://stackoverflow.com/questions/73637315/oserror-no-library-called-cairo-2-was-found-from-custom-widgets-import-proje` if you get import errors
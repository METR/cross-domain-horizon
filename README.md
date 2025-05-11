# cross-domain-horizon
Estimate the time horizon of AIs over time on various domains like knowledge and vision


### Methodology

- Only the frontier is used to fit the regression line
- Only models that get between 10% and 90% on some benchmark are counted
- The best agent is used on each model


### Usage

* Run benchmark-specific .py files to load scores for each dataset. Some also calculate horizons.

* Run calculate_horizons.py to estimate time horizons for each dataset

* Run plots.py to make all plots

### TODO

P0-- blocking internal review
- [x] Change critical disengagement to disengagement
- [x] OSWorld updated data
- [ ] compare HCAST estimated horizon to actual horizon
- plot includes saturation points (90th %ile task)

P1-- blocking publication
- [ ] aggregating benchmarks into categories
- [ ] second visual understanding benchmark: UCF101
- [ ] easier math benchmark like amc8?

P2-- nice to have

P3-- low priority
- [ ] TheAgentCompany? but no human baselines
- [ ] upgrade to use quantiles rather than individual lengths

plan for Video-MME:
- 
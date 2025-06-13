from dataclasses import dataclass
from classes import BenchmarkScoresSpec, SplitScoresSpec




def make_groups(lengths: list[float], scores: BenchmarkScoresSpec | None, group_names: list[str] | None = None, group_cutoffs: list[float] | None = None, num_quantile_groups: int | None = None) -> dict[str, SplitScoresSpec]:
    """
    Separates lengths of tasks into groups based on either group cutoffs or percentiles

    Arguments:
    - lengths: task lengths in the benchmark, in minutes
    - make_scores: if True, will also make scores for each group
    - group_names: names for each group, must be same length as num_groups
    - group_cutoffs: cutoffs for each group. If using, creates n+1 groups
    - num_quantile_groups: number of quantile groups to make, must be same length as num_groups

    Returns:
    - dict of group names to SplitScoresSpec. If make_scores is False, there will be no scores.
    """
    
    assert group_cutoffs is not None or num_quantile_groups is not None
    assert group_cutoffs is None or num_quantile_groups is None

    method = "cutoffs" if group_cutoffs is not None else "percentiles"
    num_groups = len(group_cutoffs) + 1 if method == "cutoffs" else num_quantile_groups

    if group_names is None:
        group_names = [f"group{i}" for i in range(num_groups)]
    else:
        assert len(group_names) == num_groups


    # TODO assert all groups are nonempty

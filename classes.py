from dataclasses import dataclass

@dataclass
class SplitScoresSpec:
    lengths: list[int]
    scores: dict[str, float]

@dataclass
class BenchmarkScoresSpec:
    name: str
    chance_accuracy: float
    splits: dict[str, SplitScoresSpec]
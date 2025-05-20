
def make_toml(benchmark_name: str, n_questions: int | None, chance_accuracy: float | None, lengths: list[float]) -> str:
    """
    Lengths = task lengths in minutes
    """
    return f"""
n_questions = {n_questions}
chance_accuracy = {chance_accuracy}
lengths = {lengths}
    """

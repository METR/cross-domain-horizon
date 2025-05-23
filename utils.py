def make_toml(benchmark_name: str, **kwargs) -> str:
    """
    Lengths = task lengths in minutes
    """
    return f"""
{'\n'.join(f"{k} = {repr(v) if isinstance(v, str) else v}" for k, v in kwargs.items())}
    """

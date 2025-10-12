import os
from pathlib import Path
from typing import Optional

DEFAULT_ENV_VARS = {
    '101': 'NASBENCH101_PATH',
    '201': 'NASBENCH201_PATH',
    '301': 'NASBENCH301_PATH',
}

def resolve_path(benchmark: str, provided_path: Optional[str]) -> Path:
    """Resolve dataset path using provided path or environment variables.

    For 301, the env/path can point to either a pickle file or a directory-based pickle.
    """
    if provided_path:
        return Path(provided_path)
    env = DEFAULT_ENV_VARS.get(benchmark)
    if env and os.environ.get(env):
        return Path(os.environ[env])
    raise FileNotFoundError(
        f"Dataset path not provided and environment variable {env} is not set"
    )

def sizeof_fmt(num: float) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


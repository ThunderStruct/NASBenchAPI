"""Programmatic NAS benchmark dataset downloader."""

import os
from pathlib import Path
from typing import Optional

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


HUGGINGFACE_REPOSITORY = (
    'https://huggingface.co/datasets/ThunderStruct/NASBench/resolve/main'
)

DOWNLOAD_URLS = {
    '101': f'{HUGGINGFACE_REPOSITORY}/nasbench101_full.pkl',
    '201': f'{HUGGINGFACE_REPOSITORY}/nasbench201_v1_0-e61699.pkl',
    '301': f'{HUGGINGFACE_REPOSITORY}/nasbench301.pkl',
}

ENV_VARS = {
    '101': 'NASBENCH101_PATH',
    '201': 'NASBENCH201_PATH',
    '301': 'NASBENCH301_PATH',
}


def resolve_download_path(benchmark: str,
                          output: Optional[str] = None) -> Path:
    """Resolve the destination for a benchmark download."""
    benchmark = str(benchmark)
    if benchmark not in DOWNLOAD_URLS:
        raise ValueError(
            f'Invalid benchmark {benchmark!r}. Choose from 101, 201, 301.'
        )

    filename = f'nasbench{benchmark}.pkl'
    path = output or os.environ.get(ENV_VARS[benchmark])
    if path is None:
        return Path.cwd() / 'datasets' / filename

    resolved = Path(path)
    if resolved.is_dir() or not resolved.suffix:
        resolved = resolved / filename
    return resolved


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        if HAS_TQDM and total > 0:
            progress = tqdm(total=total,
                            unit='B',
                            unit_scale=True,
                            desc=f'Downloading {destination.name}')
        else:
            progress = None

        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))

        if progress is not None:
            progress.close()


def download_benchmark(benchmark: str,
                       output: Optional[str] = None,
                       force: bool = False) -> Path:
    """Download a NAS benchmark pickle and return its local path."""
    benchmark = str(benchmark)
    destination = resolve_download_path(benchmark, output)
    if destination.exists() and not force:
        return destination

    _download(DOWNLOAD_URLS[benchmark], destination)
    return destination

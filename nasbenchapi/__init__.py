"""
NASBenchAPI: Unified APIs for NASBench-101/201/301.
"""

from .base import DatasetInfo, get_dataset_info
from .download import download_benchmark
from .nb_api import NASBench101, NASBench201, NASBench301
from .nasbench101_api import Arch101


BENCHMARKS = {
    '101': NASBench101,
    '201': NASBench201,
    '301': NASBench301,
}


def load_benchmark(benchmark, data_path=None, download=False, verbose=True):
    """Load a supported benchmark, downloading its pickle when requested."""
    benchmark = str(benchmark)
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f'Invalid benchmark {benchmark!r}. Choose from 101, 201, 301.'
        )

    if download:
        data_path = download_benchmark(benchmark, output=data_path)

    return BENCHMARKS[benchmark](data_path=str(data_path)
                                 if data_path is not None else None,
                                 verbose=verbose)

__all__ = [
    "NASBench101",
    "NASBench201",
    "NASBench301",
    "Arch101",
    "DatasetInfo",
    "get_dataset_info",
    "download_benchmark",
    "load_benchmark",
]

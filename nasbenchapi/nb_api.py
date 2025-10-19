"""
Unified API wrappers for NASBench-101/201/301.

This module provides wrappers around the benchmark-specific implementations 
to expose a consistent interface across all benchmarks.
"""

import random
from typing import Any, Dict, Iterator, List, Optional

from .base import NASBenchBase
from .nasbench101_api import NASBench101 as _NB101, Arch101
from .nasbench201_api import NASBench201 as _NB201
from .nasbench301_api import NASBench301 as _NB301


class NASBench101(NASBenchBase):
    """Unified NB101 API wrapper.

    Delegates all operations to the underlying NASBench101 implementation.
    """

    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB101(data_path, verbose=verbose)

    def load(self, data_path: Optional[str] = None) -> "NASBench101":
        """Load NB101 from the provided path or environment."""
        return NASBench101(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb101'

    def datasets(self) -> List[str]:
        """Available datasets."""
        return ['cifar10']

    def splits(self, dataset: str) -> List[str]:
        """Supported splits."""
        return ['train', 'val', 'test']

    # Delegate to underlying API
    def op_set(self) -> List[str]:
        """Operations available in NB101 cell."""
        return self.api.op_set()

    def decode(self, encoding: Dict[str, str]) -> Arch101:
        """Decode architecture from native encoding."""
        return self.api.decode(encoding)

    def encode(self, arch: Arch101) -> Dict[str, str]:
        """Encode architecture to native format."""
        return self.api.encode(arch)

    def id(self, arch: Arch101) -> str:
        """Stable identifier for architecture."""
        return self.api.id(arch)

    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Arch101]:
        """Random sample from loaded NB101 architectures."""
        return self.api.random_sample(n, seed)

    def iter_all(self) -> Iterator[Arch101]:
        """Iterate all NB101 architectures."""
        return self.api.iter_all()

    def mutate(self, arch: Arch101, rng: random.Random, kind: Optional[str] = None) -> Arch101:
        """Mutate architecture."""
        return self.api.mutate(arch, rng, kind)

    def query(self, arch: Arch101, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for architecture."""
        return self.api.query(arch, dataset, split, seed, budget)

    def is_valid(self, arch: Arch101) -> bool:
        """Check if architecture is valid."""
        return self.api.is_valid(arch)

    def train_time(self, arch: Arch101, dataset: str = 'cifar10') -> Optional[float]:
        """Get training time for architecture."""
        return self.api.train_time(arch, dataset)


class NASBench201(NASBenchBase):
    """Unified NB201 API wrapper.

    Delegates all operations to the underlying NASBench201 implementation.
    """

    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB201(data_path, verbose=verbose)

    def load(self, data_path: Optional[str] = None) -> "NASBench201":
        """Load NB201 from the provided path or environment."""
        return NASBench201(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb201'

    def datasets(self) -> List[str]:
        """Available datasets."""
        return ['cifar10', 'cifar100', 'ImageNet16-120']

    def splits(self, dataset: str) -> List[str]:
        """Supported splits."""
        return ['train', 'val', 'test']

    # Delegate to underlying API
    def decode(self, encoding: Any) -> Any:
        """Decode architecture (pass-through for now)."""
        return encoding

    def encode(self, arch: Any) -> Any:
        """Encode architecture (pass-through for now)."""
        return arch

    def id(self, arch: Any) -> str:
        """Stable identifier for architecture."""
        import hashlib
        import json
        h = hashlib.sha256()
        h.update(json.dumps({'arch': arch}, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Random sample from NB201."""
        return self.api.random_sample(n, seed)

    def random_sample_str(self, n: int = 1, seed: Optional[int] = None) -> List[str]:
        """Random sample NB201 architectures as arch strings."""
        return self.api.random_sample_str(n, seed)

    def iter_all(self) -> Iterator[Any]:
        """Iterate all NB201 architectures."""
        return self.api.iter_all()

    def mutate(self, arch: Any, rng: random.Random, kind: Optional[str] = None) -> Any:
        """Mutate architecture (no-op for now)."""
        return arch

    def query(self, arch: Any, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for architecture."""
        return self.api.query(arch, dataset, split, seed, budget)

    # Conversion helpers (benchmark-specific)
    def index_to_arch_str(self, idx: int) -> str:
        """Convert NB201 index (0..15624) to arch string."""
        return self.api.index_to_arch_str(idx)

    def arch_str_to_index(self, arch_str: str) -> int:
        """Convert NB201 arch string to canonical index (0..15624)."""
        return self.api.arch_str_to_index(arch_str)


class NASBench301(NASBenchBase):
    """Unified NB301 API wrapper.

    Delegates all operations to the underlying NASBench301 implementation.
    """

    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB301(data_path, verbose=verbose)

    def load(self, data_path: Optional[str] = None) -> "NASBench301":
        """Load NB301 from the provided path or environment."""
        return NASBench301(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb301'

    def datasets(self) -> List[str]:
        """Available datasets."""
        return ['cifar10', 'cifar100']

    def splits(self, dataset: str) -> List[str]:
        """Supported splits."""
        return ['val', 'test']

    # Delegate to underlying API
    def decode(self, encoding: Any) -> Any:
        """Decode architecture (pass-through for now)."""
        return encoding

    def encode(self, arch: Any) -> Any:
        """Encode architecture (pass-through for now)."""
        return arch

    def id(self, arch: Any) -> str:
        """Stable identifier for architecture."""
        import hashlib
        import json
        h = hashlib.sha256()
        h.update(json.dumps({'arch': arch}, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Random sample from NB301."""
        return self.api.random_sample(n, seed)

    def iter_all(self) -> Iterator[Any]:
        """Iterate all NB301 architectures."""
        return self.api.iter_all()

    def mutate(self, arch: Any, rng: random.Random, kind: Optional[str] = None) -> Any:
        """Mutate architecture (no-op for now)."""
        return arch

    def query(self, arch: Any, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for architecture."""
        return self.api.query(arch, dataset, split, seed, budget)

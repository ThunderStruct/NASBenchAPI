import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from .base import NASBenchBase
from .nasbench101_api import NASBench101 as _NB101
from .nasbench201_api import NASBench201 as _NB201
from .nasbench301_api import NASBench301 as _NB301


def _hash_arch(payload: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@dataclass
class Arch101:
    adjacency: List[List[int]]
    operations: List[str]



class NASBench101(NASBenchBase):
    """Unified NB101 API.

    Loads NB101 from a lossless pickle and exposes a consistent interface.
    """
    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB101(data_path, verbose=verbose)
        self._entries = self.api.data.get('entries_by_arch', {})
        self._latest = self.api.data.get('latest_by_arch', {})

    def load(self, data_path: Optional[str] = None) -> "NASBench101":
        """Load NB101 from the provided path or environment.

        Args:
            data_path: Optional explicit path to the NB101 pickle.

        Returns:
            Loaded API instance.
        """
        return NASBench101(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb101'

    def op_set(self) -> List[str]:
        """Operations available in the NB101 cell.

        Returns:
            List of operation names.
        """
        return ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']

    # Architecture I/O (NB101-specific)
    def decode(self, encoding: Dict[str, str]) -> Arch101:
        """Decode adjacency/operations strings into an Arch101 object.

        Args:
            encoding: Dict with 'adjacency_str' and 'operations_str'.

        Returns:
            Arch101 instance.
        """
        adj_str = encoding['adjacency_str']
        ops_str = encoding['operations_str']
        mat = [[int(adj_str[r * 7 + c]) for c in range(7)] for r in range(7)]
        ops = ops_str.split(',')
        return Arch101(adjacency=mat, operations=ops)

    def encode(self, arch: Arch101) -> Dict[str, str]:
        """Encode Arch101 into native strings.

        Args:
            arch: Architecture instance.

        Returns:
            Dict with 'adjacency_str' and 'operations_str'.
        """
        flat = ''.join(str(v) for row in arch.adjacency for v in row)
        return {'adjacency_str': flat, 'operations_str': ','.join(arch.operations)}

    def id(self, arch: Arch101) -> str:
        """Stable identifier for NB101 architectures.

        Args:
            arch: Architecture instance.

        Returns:
            Stable identifier string.
        """
        payload = {"adjacency": arch.adjacency, "operations": arch.operations}
        return _hash_arch(payload)

    # Sampling / enumeration (NB101 supports full iteration)
    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Arch101]:
        """Random sample from the loaded NB101 latest entries.

        Args:
            n: Number of samples to return.
            seed: Optional random seed.

        Returns:
            List of sampled Arch101 architectures.
        """
        if seed is not None:
            random.seed(seed)
        keys = list(self._latest.keys())
        random.shuffle(keys)
        out: List[Arch101] = []
        for k in keys[:n]:
            enc = {k2: self._latest[k][k2] for k2 in ('adjacency_str', 'operations_str')}
            out.append(self.decode(enc))
        return out

    def iter_all(self) -> Iterator[Arch101]:
        """Iterate all NB101 architectures (latest entries).

        Returns:
            Iterator over Arch101 instances.
        """
        for k, last in self._latest.items():
            enc = {k2: last[k2] for k2 in ('adjacency_str', 'operations_str')}
            yield self.decode(enc)

    # Mutation (NB101-specific simple examples)
    def mutate(self, arch: Arch101, rng: random.Random, kind: Optional[str] = None) -> Arch101:
        """Apply a simple one-edit mutation (edge toggle or op swap).

        Args:
            arch: Architecture to mutate.
            rng: Random number generator to use.
            kind: Mutation kind ('edge_toggle' or 'op_swap').

        Returns:
            Mutated architecture.
        """
        kind = kind or 'edge_toggle'
        if kind == 'edge_toggle':
            r, c = rng.randrange(7), rng.randrange(7)
            if r == c:
                return arch
            new_adj = [row[:] for row in arch.adjacency]
            new_adj[r][c] = 1 - new_adj[r][c]
            return Arch101(new_adj, arch.operations[:])
        if kind == 'op_swap':
            i, j = rng.randrange(7), rng.randrange(7)
            new_ops = arch.operations[:]
            new_ops[i], new_ops[j] = new_ops[j], new_ops[i]
            return Arch101([row[:] for row in arch.adjacency], new_ops)
        return arch

    # Evaluation (NB101 uses derived validation accuracy if available)
    def evaluate(self, arch: Arch101, dataset: str = 'cifar10', split: str = 'val', seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Return validation metric and info from the lossless record if available.

        Args:
            arch: Architecture to evaluate.
            dataset: Dataset name.
            split: Split name ('val', 'test', etc.).
            seed: Optional seed for reproducibility.
            budget: Optional budget specification.

        Returns:
            Dictionary with keys: metric, metric_name, cost, std, info.
        """
        enc = self.encode(arch)
        # Find matching latest entry
        key = None
        # Use hash to speed, otherwise fallback linear scan
        target_hash = _hash_arch({"adjacency": arch.adjacency, "operations": arch.operations})
        # Fallback search by operations/adjacency_str
        for k, last in self._latest.items():
            if last['adjacency_str'] == enc['adjacency_str'] and last['operations_str'] == enc['operations_str']:
                key = k
                break
        info = {}
        metric_name = 'val_acc'
        metric = None
        if key is not None:
            d = last = self._latest[key]
            metric = d.get('derived', {}).get('validation_accuracy', None)
            info = d
        return {
            'metric': float(metric) if metric is not None else None,
            'metric_name': metric_name,
            'cost': info.get('derived', {}).get('training_time') if isinstance(info, dict) else None,
            'std': None,
            'info': info,
        }

    def is_valid(self, arch: Arch101) -> bool:
        return len(arch.adjacency) == 7 and all(len(r) == 7 for r in arch.adjacency) and len(arch.operations) == 7

    def train_time(self, arch: Arch101, dataset: str) -> Optional[float]:
        enc = self.encode(arch)
        for k, last in self._latest.items():
            if last['adjacency_str'] == enc['adjacency_str'] and last['operations_str'] == enc['operations_str']:
                return last.get('derived', {}).get('training_time')
        return None


class NASBench201(NASBenchBase):
    """Unified NB201 API (placeholders for encode/decode until defined)."""
    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB201(data_path, verbose=verbose)

    def load(self, data_path: Optional[str] = None) -> "NASBench201":
        """Load NB201 from the provided path or environment.

        Args:
            data_path: Optional explicit path to the NB201 pickle.

        Returns:
            Loaded API instance.
        """
        return NASBench201(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb201'

    def datasets(self) -> List[str]:
        """Supported datasets for NB201.

        Returns:
            List of dataset names.
        """
        return ['cifar10', 'cifar100', 'ImageNet16-120']

    def metric_convention(self) -> Dict[str, Any]:
        return {"val_acc": {"higher_is_better": True}}

    # Minimal pass-throughs until a canonical cell representation is chosen
    def decode(self, encoding: Any) -> Any:
        """Decode NB201 encoding (pass-through until defined).

        Args:
            encoding: Benchmark-native encoding.

        Returns:
            Decoded architecture (pass-through).
        """
        return encoding

    def encode(self, arch: Any) -> Any:
        """Encode NB201 architecture (pass-through until defined).

        Args:
            arch: Architecture object.

        Returns:
            Native encoding (pass-through).
        """
        return arch

    def id(self, arch: Any) -> str:
        return _hash_arch({'arch': arch})

    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Random sample for NB201 (empty until enumeration defined).

        Args:
            n: Number to sample.
            seed: Optional random seed.

        Returns:
            Empty list until enumeration is defined.
        """
        if seed is not None:
            random.seed(seed)
        # Without a canonical index set, return empty list
        return []

    def iter_all(self) -> Iterator[Any]:
        """Iterate all NB201 architectures (empty until defined).

        Returns:
            Empty iterator until enumeration is defined.
        """
        return iter(())

    def mutate(self, arch: Any, rng: random.Random, kind: Optional[str] = None) -> Any:
        """Mutate NB201 architecture (no-op until defined).

        Args:
            arch: Architecture to mutate.
            rng: Random generator.
            kind: Mutation kind.

        Returns:
            Unchanged architecture.
        """
        return arch

    def evaluate(self, arch: Any, dataset: str = 'cifar10', split: str = 'val', seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Evaluate NB201 architecture (placeholder).

        Args:
            arch: Architecture to evaluate.
            dataset: Dataset name.
            split: Split name.
            seed: Optional seed.
            budget: Optional budget.

        Returns:
            Placeholder metric dict.
        """
        return {'metric': None, 'metric_name': 'val_acc', 'cost': None, 'std': None, 'info': {}}


class NASBench301(NASBenchBase):
    """Unified NB301 API (placeholders until representation defined)."""
    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB301(data_path, verbose=verbose)

    def load(self, data_path: Optional[str] = None) -> "NASBench301":
        """Load NB301 from the provided path or environment.

        Args:
            data_path: Optional explicit path to the NB301 pickle.

        Returns:
            Loaded API instance.
        """
        return NASBench301(data_path)

    def bench_name(self) -> str:
        """Short benchmark name."""
        return 'nb301'

    def datasets(self) -> List[str]:
        """Supported datasets for NB301.

        Returns:
            List of dataset names.
        """
        return ['cifar10', 'cifar100']

    # Minimal pass-throughs until a canonical representation is chosen
    def decode(self, encoding: Any) -> Any:
        """Decode NB301 encoding (pass-through until defined).

        Args:
            encoding: Benchmark-native encoding.

        Returns:
            Decoded architecture (pass-through).
        """
        return encoding

    def encode(self, arch: Any) -> Any:
        """Encode NB301 architecture (pass-through until defined).

        Args:
            arch: Architecture object.

        Returns:
            Native encoding (pass-through).
        """
        return arch

    def id(self, arch: Any) -> str:
        return _hash_arch({'arch': arch})

    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Any]:
        """Random sample for NB301 (empty until enumeration defined).

        Args:
            n: Number to sample.
            seed: Optional random seed.

        Returns:
            Empty list until enumeration is defined.
        """
        if seed is not None:
            random.seed(seed)
        return []

    def iter_all(self) -> Iterator[Any]:
        """Iterate all NB301 architectures (empty until defined).

        Returns:
            Empty iterator until enumeration is defined.
        """
        return iter(())

    def mutate(self, arch: Any, rng: random.Random, kind: Optional[str] = None) -> Any:
        """Mutate NB301 architecture (no-op until defined).

        Args:
            arch: Architecture to mutate.
            rng: Random generator.
            kind: Mutation kind.

        Returns:
            Unchanged architecture.
        """
        return arch

    def evaluate(self, arch: Any, dataset: str = 'cifar10', split: str = 'val', seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Evaluate NB301 architecture (placeholder; surrogate not bundled).

        Args:
            arch: Architecture to evaluate.
            dataset: Dataset name.
            split: Split name.
            seed: Optional seed.
            budget: Optional budget.

        Returns:
            Placeholder metric dict.
        """
        # Surrogate model not bundled; return None metric by default
        return {'metric': None, 'metric_name': 'val_acc', 'cost': None, 'std': None, 'info': {}}



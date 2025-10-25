
import pickle
import random
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Iterator

from .utils import resolve_path, sizeof_fmt

try:
    # Optional import to minimize pip overhead
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



def _hash_arch(payload: Dict[str, Any]) -> str:
    """Hash an architecture payload for stable IDs."""
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@dataclass
class Arch101:
    """NASBench-101 architecture representation."""
    adjacency: List[List[int]]  # 7x7 adjacency matrix
    operations: List[str]  # 7 operations


class NASBench101:
    """NASBench-101 API.

    Expects a pickle with NB101 keys (entries_by_arch, 
    latest_by_arch, num_records).
    """

    def __init__(self, pickle_path: Optional[str] = None, verbose: bool = True):
        self.path = resolve_path('101', pickle_path)
        self.verbose = verbose
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        size = self.path.stat().st_size
        if self.verbose:
            print(f"Loading NB101 from {self.path} ({sizeof_fmt(size)})")
        with open(self.path, 'rb') as f:
            if HAS_TQDM and self.verbose and size > 0:
                bar = tqdm(total=size, unit='B', unit_scale=True, desc='Reading')
                raw = bytearray()
                chunk = f.read(1024 * 1024)
                while chunk:
                    raw.extend(chunk)
                    bar.update(len(chunk))
                    chunk = f.read(1024 * 1024)
                bar.close()
                # Unpickling stage
                if self.verbose:
                    print("Unpickling data...")
                self.data = pickle.loads(bytes(raw))
                if self.verbose:
                    print("Unpickling complete.")
            else:
                if self.verbose:
                    print("Unpickling data...")
                self.data = pickle.load(f)
                if self.verbose:
                    print("Unpickling complete.")
        if self.verbose:
            entries = self.data.get('entries_by_arch')
            if isinstance(entries, dict):
                arch_count = len(entries)
            else:
                arch_count = len(self.data.get('latest_by_arch', {}))
            records = self.data.get('num_records')
            extra = f" (records={records})" if records is not None else ""
            print(f"[NB101] Loaded {arch_count} architectures{extra}")

    def get_statistics(self) -> Dict[str, Any]:
        entries = self.data.get('entries_by_arch', {})
        return {
            'benchmark': 'nasbench101',
            'architectures': len(entries),
            'records': self.data.get('num_records', 0),
        }

    # Benchmark operations
    def op_set(self) -> List[str]:
        """Operations available in the NB101 cell."""
        return ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']

    # Architecture I/O
    def decode(self, encoding: Dict[str, str]) -> Arch101:
        """Decode adjacency/operations strings into an Arch101 object."""
        adj_str = encoding['adjacency_str']
        ops_str = encoding['operations_str']
        mat = [[int(adj_str[r * 7 + c]) for c in range(7)] for r in range(7)]
        ops = ops_str.split(',')
        return Arch101(adjacency=mat, operations=ops)

    def encode(self, arch: Arch101) -> Dict[str, str]:
        """Encode Arch101 into native strings."""
        flat = ''.join(str(v) for row in arch.adjacency for v in row)
        return {'adjacency_str': flat, 'operations_str': ','.join(arch.operations)}

    def id(self, arch: Arch101) -> str:
        """Stable identifier for NB101 architectures."""
        payload = {"adjacency": arch.adjacency, "operations": arch.operations}
        return _hash_arch(payload)

    def get_index(self, arch: Arch101) -> str:
        """Get the stable hash identifier for an architecture.

        Args:
            arch: Arch101 architecture object.

        Returns:
            String hash identifier for the architecture.

        Note:
            For NB101, architectures are identified by hash rather than numeric index.
            This method is equivalent to id() but provided for API consistency.
        """
        return self.id(arch)

    # Sampling / enumeration
    def random_sample(self, n: int = 1, seed: Optional[int] = None) -> List[Arch101]:
        """Random sample from the loaded NB101 latest entries."""
        if seed is not None:
            random.seed(seed)
        latest = self.data.get('latest_by_arch', {})
        keys = list(latest.keys())
        random.shuffle(keys)
        out: List[Arch101] = []
        for k in keys[:n]:
            enc = {k2: latest[k][k2] for k2 in ('adjacency_str', 'operations_str')}
            out.append(self.decode(enc))
        return out

    def iter_all(self) -> Iterator[Arch101]:
        """Iterate all NB101 architectures (latest entries)."""
        latest = self.data.get('latest_by_arch', {})
        for k, last in latest.items():
            enc = {k2: last[k2] for k2 in ('adjacency_str', 'operations_str')}
            yield self.decode(enc)

    # Mutation
    def mutate(self, arch: Arch101, rng: random.Random, kind: Optional[str] = None) -> Arch101:
        """Apply a simple one-edit mutation (edge toggle or op swap)."""
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

    # Query (renamed from evaluate)
    def query(self, arch: Arch101, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for an architecture from loaded data.

        Args:
            arch: Architecture to query.
            dataset: Dataset name ('cifar10').
            split: Split name ('val', 'test', 'train').
            seed: Optional seed for reproducibility.
            budget: Optional budget specification.

        Returns:
            Dictionary with keys: metric, metric_name, cost, std, info.
        """
        enc = self.encode(arch)
        latest = self.data.get('latest_by_arch', {})

        # Find matching latest entry
        key = None
        for k, last in latest.items():
            if last['adjacency_str'] == enc['adjacency_str'] and last['operations_str'] == enc['operations_str']:
                key = k
                break

        info = {}
        metric_name = f'{split}_acc'
        metric = None
        cost = None

        if key is not None:
            d = latest[key]
            derived = d.get('derived', {})

            # Get metrics based on split
            if split == 'val':
                metric = derived.get('validation_accuracy')
            elif split == 'test':
                metric = derived.get('test_accuracy')
            elif split == 'train':
                metric = derived.get('train_accuracy')

            cost = derived.get('training_time')
            info = d

        return {
            'metric': float(metric) if metric is not None else None,
            'metric_name': metric_name,
            'cost': float(cost) if cost is not None else None,
            'std': None,
            'info': info,
        }

    def is_valid(self, arch: Arch101) -> bool:
        """Check if architecture is valid."""
        return (len(arch.adjacency) == 7 and
                all(len(r) == 7 for r in arch.adjacency) and
                len(arch.operations) == 7)

    def train_time(self, arch: Arch101, dataset: str = 'cifar10') -> Optional[float]:
        """Get training time for an architecture."""
        enc = self.encode(arch)
        latest = self.data.get('latest_by_arch', {})
        for k, last in latest.items():
            if last['adjacency_str'] == enc['adjacency_str'] and last['operations_str'] == enc['operations_str']:
                time_val = last.get('derived', {}).get('training_time')
                return float(time_val) if time_val is not None else None
        return None

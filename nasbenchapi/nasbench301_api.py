
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


class NASBench301:
    """NAS-Bench-301 API.

    Supports directory-converted JSON payloads or reserialized pickle.
    """

    # NB301 uses DARTS-style search space
    OPS = [
        'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
        'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none'
    ]
    NUM_OPS = len(OPS)
    NUM_NODES = 4  # Intermediate nodes in the cell
    NUM_EDGES_PER_NODE = 2  # Each node has 2 input edges

    def __init__(self, pickle_path: Optional[str] = None, verbose: bool = True):
        self.path = resolve_path('301', pickle_path)
        self.verbose = verbose
        self.data: Any = None
        self._arch_keys = []
        self._load()

    def _load(self) -> None:
        size = self.path.stat().st_size
        if self.verbose:
            print(f"Loading NB301 from {self.path} ({sizeof_fmt(size)})")
        with open(self.path, 'rb') as f:
            if HAS_TQDM and size > 0:
                bar = tqdm(total=size, unit='B', unit_scale=True, desc='Reading')
                raw = bytearray()
                chunk = f.read(1024 * 1024)
                while chunk:
                    raw.extend(chunk)
                    bar.update(len(chunk))
                    chunk = f.read(1024 * 1024)
                bar.close()
                # Unpickling stage
                unp = tqdm(total=1, desc='Unpickling', unit='step')
                self.data = pickle.loads(bytes(raw))
                unp.update(1)
                unp.close()
            else:
                if HAS_TQDM:
                    unp = tqdm(total=1, desc='Unpickling', unit='step')
                    self.data = pickle.load(f)
                    unp.update(1)
                    unp.close()
                else:
                    self.data = pickle.load(f)
        if self.verbose:
            if isinstance(self.data, dict) and 'entries' in self.data:
                print(f"Loaded NB301 directory payload with {len(self.data['entries'])} JSON files")
            else:
                print("Loaded NB301 pickle")

        # Cache architecture keys if data is a dict
        if isinstance(self.data, dict):
            if 'entries' in self.data:
                self._arch_keys = list(range(len(self.data['entries'])))
            else:
                self._arch_keys = list(self.data.keys())

    def get_statistics(self) -> Dict[str, Any]:
        if isinstance(self.data, dict) and 'entries' in self.data:
            return {
                'benchmark': 'nasbench301',
                'files': self.data.get('num_files', len(self.data.get('entries', []))),
            }
        return {
            'benchmark': 'nasbench301',
            'files': None,
        }

    def random_sample(self, n: int = 1, seed: Optional[int] = None):
        """Sample random architectures from NB301 search space.

        NB301 uses DARTS-style search space which is continuous, but we can
        sample discrete architectures from it.

        Args:
            n: Number of samples to return.
            seed: Optional random seed.

        Returns:
            List of sampled architecture representations.
        """
        import random as rnd
        if seed is not None:
            rnd.seed(seed)

        if self._arch_keys:
            # Sample from loaded architectures
            return rnd.sample(self._arch_keys, min(n, len(self._arch_keys)))
        else:
            # Generate random DARTS-style architectures
            # Each cell has 4 nodes, each node selects 2 operations from previous nodes
            samples = []
            for _ in range(n):
                # Normal cell and reduction cell
                normal = []
                reduce = []
                for node_idx in range(self.NUM_NODES):
                    # For each node, select 2 edges (predecessors and operations)
                    for _ in range(self.NUM_EDGES_PER_NODE):
                        # Can connect to any previous node (0 to node_idx+1)
                        pred = rnd.randint(0, node_idx + 1)
                        op = rnd.choice(self.OPS)
                        normal.append((op, pred))
                        reduce.append((op, pred))
                samples.append({'normal': normal, 'reduce': reduce})
            return samples

    def iter_all(self):
        """Iterate over all architectures in the loaded data.

        Returns:
            Iterator over architecture keys/indices.
        """
        if self._arch_keys:
            return iter(self._arch_keys)
        return iter(())

    def query(self, arch: Any, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for an architecture from loaded data.

        Note: NB301 uses surrogate models for predictions. This method returns
        placeholder results unless surrogate models are loaded.

        Args:
            arch: Architecture representation (dict or index).
            dataset: Dataset name ('cifar10', 'cifar100').
            split: Split name ('val', 'test').
            seed: Optional seed for reproducibility (unused).
            budget: Optional budget specification (unused).

        Returns:
            Dictionary with keys: metric, metric_name, cost, std, info.
        """
        # NB301 typically requires surrogate models for predictions
        # Return placeholder until surrogate models are integrated
        return {
            'metric': None,
            'metric_name': f'{split}_acc',
            'cost': None,
            'std': None,
            'info': {'note': 'NB301 requires surrogate models for predictions'},
        }



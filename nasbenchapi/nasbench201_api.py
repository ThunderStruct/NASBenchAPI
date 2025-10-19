
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



class NASBench201:
    """NASBench-201 API"""

    # NB201 operations for the search space
    OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    NUM_OPS = len(OPS)
    NUM_EDGES = 6  # 6 edges in the cell

    def __init__(self, pickle_path: Optional[str] = None, verbose: bool = True):
        self.path = resolve_path('201', pickle_path)
        self.verbose = verbose
        self.data: Any = None
        self._arch_keys = []
        # Optional mappings if available from loaded data
        self._idx_to_str: Dict[int, str] = {}
        self._str_to_idx: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        size = self.path.stat().st_size
        if self.verbose:
            print(f"Loading NB201 from {self.path} ({sizeof_fmt(size)})")
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
                # Unpickling stage (no size info)
                if self.verbose:
                    print("Unpickling data...")
                self.data = pickle.load(f)
                if self.verbose:
                    print("Unpickling complete.")
        if self.verbose:
            size_info = 'dict' if isinstance(self.data, dict) else type(self.data).__name__
            print(f"Loaded NB201 data ({size_info})")

        # Cache architecture keys if data is a dict
        if isinstance(self.data, dict):
            # Check if this is the official NB201 format with nested structure
            # Official format has: meta_archs, arch2infos, evaluated_indexes, etc.
            if 'arch2infos' in self.data:
                # Use arch2infos for architecture data
                # Ensure indices are integers
                self._arch_keys = [int(k) for k in self.data['arch2infos'].keys()]
                if self.verbose:
                    print(f"Found {len(self._arch_keys)} architectures in NB201")
                # Build index<->arch_str mappings when available
                try:
                    for k in self._arch_keys:
                        entry = self.data['arch2infos'].get(k, {})
                        arch_str = None
                        if isinstance(entry, dict) and 'full' in entry and isinstance(entry['full'], dict):
                            arch_str = entry['full'].get('arch_str')
                        if isinstance(arch_str, str):
                            self._idx_to_str[int(k)] = arch_str
                            # If duplicates exist, keep the first seen mapping
                            self._str_to_idx.setdefault(arch_str, int(k))
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: failed to build NB201 arch_str mappings: {e}")
            else:
                # Fallback to top-level keys
                self._arch_keys = list(self.data.keys())
                if self.verbose:
                    print(f"Found {len(self._arch_keys)} entries in NB201")
                    if len(self._arch_keys) < 100:
                        print(f"  Top-level keys: {self._arch_keys}")
                        print(f"  Note: These may be metadata keys, not architectures")

    def get_statistics(self) -> Dict[str, Any]:
        if isinstance(self.data, dict) and 'arch2infos' in self.data:
            n = len(self.data['arch2infos'])
        else:
            n = len(self.data) if isinstance(self.data, dict) else None
        return {
            'benchmark': 'nasbench201',
            'entries': n,
        }

    def random_sample(self, n: int = 1, seed: Optional[int] = None):
        """Sample random architectures from NB201 search space.

        Args:
            n: Number of samples to return.
            seed: Optional random seed.

        Returns:
            List of sampled architecture strings in NB201 canonical format.
        """
        import random as rnd
        if seed is not None:
            rnd.seed(seed)

        # Sample indices first
        if self._arch_keys:
            # Sample from loaded architectures
            idxs = rnd.sample(self._arch_keys, min(n, len(self._arch_keys)))
        else:
            # Sample uniformly from the full index space (0..15624)
            max_idx = (self.NUM_OPS ** self.NUM_EDGES) - 1  # 15624
            idxs = [rnd.randint(0, max_idx) for _ in range(n)]

        # Convert to strings
        return [self._idx_to_str.get(i, self._index_to_arch_str(i)) for i in idxs]

    def iter_all(self):
        """Iterate over all architectures in the loaded data.

        Returns:
            Iterator over architecture strings.
        """
        if self._arch_keys:
            # Convert indices to strings
            return (self._idx_to_str.get(i, self._index_to_arch_str(i)) for i in self._arch_keys)
        return iter(())

    def get_index(self, arch: str) -> int:
        """Convert an architecture string to its canonical index.

        Args:
            arch: NB201 architecture string (e.g., '|none~0|+|skip_connect~0|nor_conv_1x1~1|+|...')

        Returns:
            Integer index (0..15624) corresponding to the architecture.

        Raises:
            ValueError: If the architecture string is invalid or cannot be parsed.
        """
        # Try mapping first (faster if loaded)
        arch_idx = self._str_to_idx.get(arch)
        if arch_idx is not None:
            return arch_idx

        # Otherwise decode from string
        return self._arch_str_to_index(arch)

    def query(self, arch: str, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for an architecture from loaded data.

        Args:
            arch: NB201 architecture string.
            dataset: Dataset name ('cifar10', 'cifar100', 'ImageNet16-120').
            split: Split name ('val', 'test', or 'train').
            seed: Optional seed (default: 777 for official NB201).
            budget: Optional epoch number (0-199, default: 199 for final epoch).

        Returns:
            Dictionary with keys: metric, metric_name, cost, std, info.
        """
        # Convert to index
        try:
            arch_idx = self.get_index(arch)
        except Exception:
            arch_idx = None

        if not isinstance(self.data, dict) or 'arch2infos' not in self.data:
            return {
                'metric': None,
                'metric_name': f'{split}_acc',
                'cost': None,
                'std': None,
                'info': {'note': 'NB201 data not loaded', 'arch': arch},
            }

        # Get architecture data
        arch_data = self.data['arch2infos'].get(arch_idx) if arch_idx is not None else None
        if arch_data is None:
            return {
                'metric': None,
                'metric_name': f'{split}_acc',
                'cost': None,
                'std': None,
                'info': {'error': 'architecture not found', 'arch': arch},
            }

        # Default values
        if seed is None:
            seed = 777  # Official NB201 seed
        if budget is None:
            budget = 199  # Final epoch (0-199)

        metric = None
        cost = None

        # Build clean info dict with only essential metadata
        info = {
            'arch_index': arch_idx,
            'dataset': dataset,
            'split': split,
            'seed': seed,
            'epoch': budget,
        }

        # Add architecture string if available
        if 'full' in arch_data and 'arch_str' in arch_data['full']:
            info['arch_str'] = arch_data['full']['arch_str']
        elif isinstance(arch, str):
            info['arch_str'] = arch

        # Navigate official NB201 structure
        # arch_data['full']['all_results'][(dataset, seed)]
        if 'full' in arch_data:
            full_data = arch_data['full']

            # Add basic arch info
            if 'arch_config' in full_data:
                info['params'] = full_data.get('params')
                info['flop'] = full_data.get('flop')

            if 'all_results' in full_data:
                all_results = full_data['all_results']

                # Map split names and construct result key
                result_key = None
                metric_key_prefix = None

                if split in ['val', 'valid']:
                    result_key = (f'{dataset}-valid', seed)
                    metric_key_prefix = 'x-valid@'
                elif split == 'test':
                    result_key = (dataset, seed)
                    metric_key_prefix = 'ori-test@'
                elif split == 'train':
                    result_key = (dataset, seed)
                    if result_key in all_results:
                        result = all_results[result_key]
                        if 'train_acc1es' in result and budget in result['train_acc1es']:
                            metric = result['train_acc1es'][budget]
                        if 'train_times' in result and result['train_times'] and budget in result['train_times']:
                            cost = result['train_times'][budget]
                    return {
                        'metric': float(metric) if metric is not None else None,
                        'metric_name': 'train_acc',
                        'cost': float(cost) if cost is not None else None,
                        'std': None,
                        'info': info,
                    }

                def _try_eval(result_key_tuple, key_prefix, budgets_to_try):
                    nonlocal metric, cost
                    if result_key_tuple not in all_results:
                        return False
                    result_local = all_results[result_key_tuple]
                    eval_acc = result_local.get('eval_acc1es', {})
                    eval_times = result_local.get('eval_times', {})
                    for b in budgets_to_try:
                        mkey = f"{key_prefix}{b}"
                        if mkey in eval_acc:
                            metric = eval_acc[mkey]
                            if mkey in eval_times:
                                cost = eval_times[mkey]
                            return True
                    return False

                # For val/test, search across typical budgets and fallbacks
                if result_key and metric_key_prefix:
                    budgets_try = []
                    # prioritize requested budget
                    if isinstance(budget, int):
                        budgets_try.append(budget)
                    # common final epochs in NB201
                    for b in (199, 200):
                        if b not in budgets_try:
                            budgets_try.append(b)

                    found = _try_eval(result_key, metric_key_prefix, budgets_try)

                    # Fallback: if requesting test and not found, try validation
                    if not found and split == 'test':
                        val_key = (f'{dataset}-valid', seed)
                        found = _try_eval(val_key, 'x-valid@', budgets_try)

        return {
            'metric': float(metric) if metric is not None else None,
            'metric_name': f'{split}_acc',
            'cost': float(cost) if cost is not None else None,
            'std': None,
            'info': info,
        }

    # --- Internal encoding helpers -------------------------------------------------
    def _index_to_arch_str(self, arch_idx: int) -> str:
        """Convert an architecture index (0..15624) to NB201 arch string.

        Uses a fixed edge order: (1<-0), (2<-0), (2<-1), (3<-0), (3<-1), (3<-2).
        """
        op_ids = self._index_to_ops(arch_idx)
        ops = [self.OPS[i] for i in op_ids]
        return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)

    def _arch_str_to_index(self, arch_str: str) -> int:
        """Convert a NB201 arch string to its canonical index (0..15624)."""
        op_ids = self._arch_str_to_ops(arch_str)
        return self._ops_to_index(op_ids)

    def _index_to_ops(self, idx: int) -> list:
        """Convert index to a list of 6 op IDs (0..4) in canonical order."""
        if idx < 0 or idx >= (self.NUM_OPS ** self.NUM_EDGES):
            raise ValueError(f"NB201 index out of range: {idx}")
        base = self.NUM_OPS
        out = []
        x = idx
        for _ in range(self.NUM_EDGES):
            out.append(x % base)
            x //= base
        # out[0] -> edge0, ... already in correct order
        return out

    def _ops_to_index(self, op_ids: list) -> int:
        """Convert a list of 6 op IDs (0..4) to canonical index."""
        if len(op_ids) != self.NUM_EDGES:
            raise ValueError("NB201 requires 6 operation IDs")
        base = self.NUM_OPS
        idx = 0
        mul = 1
        for i in range(self.NUM_EDGES):
            oid = int(op_ids[i])
            if oid < 0 or oid >= base:
                raise ValueError(f"Invalid op id {oid} for NB201")
            idx += oid * mul
            mul *= base
        return idx

    def _arch_str_to_ops(self, arch_str: str) -> list:
        """Parse a NB201 arch string into 6 operation IDs in canonical order."""
        # Extract tokens like 'op~0', 'op~1', ... in order
        # Replace separators and split
        s = arch_str.replace('|', ' ').replace('+', ' ').strip()
        tokens = [t for t in s.split() if '~' in t]
        if len(tokens) != self.NUM_EDGES:
            # Some formats may include trailing separators; try a more robust parse
            tokens = []
            buf = ''
            for ch in arch_str:
                if ch == '|':
                    if buf:
                        buf = buf.strip()
                        if buf:
                            tokens.extend([x for x in buf.split('|') if x])
                        buf = ''
                else:
                    buf += ch
            tokens = [t.strip() for t in tokens if '~' in t]
        if len(tokens) != self.NUM_EDGES:
            raise ValueError(f"Cannot parse NB201 arch_str: {arch_str}")
        ops_only = [t.split('~')[0] for t in tokens]
        op_ids = []
        for op in ops_only:
            if op not in self.OPS:
                raise ValueError(f"Unknown NB201 op '{op}' in arch_str")
            op_ids.append(self.OPS.index(op))
        return op_ids


import pickle
import random
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Iterator, Tuple

from .utils import resolve_path, sizeof_fmt

try:
    # Optional import to minimize pip overhead
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class NASBench301:
    """NASBench-301 API.

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
        self._path_to_index: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        size = self.path.stat().st_size
        if self.verbose:
            print(f"Loading NB301 from {self.path} ({sizeof_fmt(size)})")
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
        entry_count = None
        # Cache architecture keys if data is a dict
        if isinstance(self.data, dict):
            if 'entries' in self.data:
                entries = self.data.get('entries', [])
                self._arch_keys = list(range(len(entries)))
                entry_count = len(self._arch_keys)
                self._path_to_index = {}
                for idx, entry in enumerate(entries):
                    if isinstance(entry, dict):
                        path = entry.get('path')
                        if isinstance(path, str):
                            self._path_to_index[path] = idx
            else:
                self._arch_keys = list(self.data.keys())
                entry_count = len(self._arch_keys)
                self._path_to_index = {}
        else:
            self._arch_keys = []
            self._path_to_index = {}

        loaded_count = len(self._arch_keys)
        if entry_count is None:
            entry_count = loaded_count if loaded_count else None
        if self.verbose:
            if loaded_count:
                print(f"[NB301] Loaded {loaded_count} entries")
            else:
                print(f"[NB301] Loaded NB301 payload")

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

    def get_index(self, arch: Any) -> Optional[int]:
        """Get the index for an architecture in the loaded data.

        Args:
            arch: Architecture representation (dict, index, or path string).

        Returns:
            Integer index if architecture is found in loaded data, None otherwise.
        """
        return self._index_from_arch(arch)

    def query(self, arch: Any, dataset: str = 'cifar10', split: str = 'val',
              seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
        """Query performance metrics for an architecture from loaded data.

        Note: NB301 directory payloads include recorded learning curves and final metrics.
        Validation queries read from the stored curves; test queries report the declared
        final accuracy for each entry.

        Args:
            arch: Architecture representation (dict or index).
            dataset: Dataset name ('cifar10', 'cifar100').
            split: Split name ('val', 'test').
            seed: Optional seed for reproducibility (unused).
            budget: Optional budget specification (unused).

        Returns:
            Dictionary with keys: metric, metric_name, cost, std, info.
        """
        metric_name = f'{split}_acc'
        info: Dict[str, Any] = {
            'requested_dataset': dataset,
            'split': split,
        }

        entries = self.data.get('entries') if isinstance(self.data, dict) else None
        if not isinstance(entries, list):
            info['error'] = 'NB301 data not loaded'
            return {
                'metric': None,
                'metric_name': metric_name,
                'cost': None,
                'std': None,
                'info': info,
            }

        idx = self._index_from_arch(arch)
        if idx is None or idx < 0 or idx >= len(entries):
            info.update({'error': 'entry not found', 'arch': arch})
            return {
                'metric': None,
                'metric_name': metric_name,
                'cost': None,
                'std': None,
                'info': info,
            }

        entry = entries[idx]
        parsed = entry.get('parsed', {}) if isinstance(entry, dict) else {}
        dataset_actual = self._infer_dataset(parsed)
        if dataset_actual:
            info['dataset'] = dataset_actual
        path = entry.get('path') if isinstance(entry, dict) else None
        if isinstance(path, str):
            info['path'] = path
        optimizer = parsed.get('created_by_optimizer')
        if optimizer:
            info['optimizer'] = optimizer
        declared_budget = parsed.get('budget')
        if declared_budget is not None:
            info['declared_budget'] = declared_budget

        cost = parsed.get('runtime')
        try:
            cost_value = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost_value = None

        metric = None
        epoch_used: Optional[int] = None
        epochs_available: Optional[int] = None

        if split == 'val':
            curve = self._get_learning_curve(parsed, 'Train/val_accuracy')
            metric, epoch_used, epochs_available = self._curve_metric(curve, budget)
            if metric is None:
                metric = self._final_metric(parsed, 'val')
            if epochs_available is not None:
                info['epochs_available'] = epochs_available
            if epoch_used is not None:
                info['epoch_used'] = epoch_used
        elif split == 'test':
            metric = self._final_metric(parsed, 'test')
            if declared_budget is not None:
                info['epoch_used'] = declared_budget
        else:
            info['error'] = f"split '{split}' not supported for NB301"
            return {
                'metric': None,
                'metric_name': metric_name,
                'cost': cost_value,
                'std': None,
                'info': info,
            }

        if dataset_actual and dataset and dataset_actual != dataset:
            info['note'] = f"entry recorded on dataset '{dataset_actual}'"

        try:
            metric_value = float(metric) if metric is not None else None
        except (TypeError, ValueError):
            metric_value = None

        return {
            'metric': metric_value,
            'metric_name': metric_name,
            'cost': cost_value,
            'std': None,
            'info': info,
        }

    # --- Internal helpers -------------------------------------------------
    def _index_from_arch(self, arch: Any) -> Optional[int]:
        if not isinstance(self.data, dict) or 'entries' not in self.data:
            return None

        entries = self.data.get('entries', [])
        if not isinstance(entries, list):
            return None

        if isinstance(arch, int):
            return arch if 0 <= arch < len(entries) else None

        if isinstance(arch, str):
            return self._path_to_index.get(arch)

        if isinstance(arch, dict):
            idx = arch.get('index')
            if isinstance(idx, int) and 0 <= idx < len(entries):
                return idx
            path = arch.get('path')
            if isinstance(path, str) and path in self._path_to_index:
                return self._path_to_index[path]
            # Direct match
            if arch in entries:
                return entries.index(arch)
            parsed = arch.get('parsed')
            if parsed is not None:
                for cand_idx, candidate in enumerate(entries):
                    if isinstance(candidate, dict) and candidate.get('parsed') == parsed:
                        return cand_idx

        return None

    def _infer_dataset(self, parsed: Dict[str, Any]) -> Optional[str]:
        info_list = parsed.get('info')
        if isinstance(info_list, list) and info_list:
            first = info_list[0]
            if isinstance(first, dict):
                path = first.get('dataset_path', '')
                if isinstance(path, str):
                    lowered = path.lower()
                    if 'cifar100' in lowered:
                        return 'cifar100'
                    if 'cifar10' in lowered:
                        return 'cifar10'
                    if 'imagenet16-120' in lowered or 'imagenet16' in lowered:
                        return 'ImageNet16-120'
        return None

    def _get_learning_curve(self, parsed: Dict[str, Any], key: str) -> List[float]:
        curves = parsed.get('learning_curves')
        if isinstance(curves, dict):
            curve = curves.get(key)
            if isinstance(curve, list):
                numeric = []
                for v in curve:
                    try:
                        numeric.append(float(v))
                    except (TypeError, ValueError):
                        continue
                return numeric
        return []

    def _curve_metric(self, curve: List[float], budget: Optional[Any]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        if not curve:
            return None, None, None

        total_epochs = len(curve)
        epoch_used = total_epochs
        if budget is not None:
            try:
                epoch = int(budget)
                if epoch < 1:
                    return None, None, total_epochs
                if epoch > total_epochs:
                    epoch_used = total_epochs
                else:
                    epoch_used = epoch
            except (TypeError, ValueError):
                epoch_used = total_epochs

        value = curve[epoch_used - 1]
        try:
            metric = float(value)
        except (TypeError, ValueError):
            metric = None

        return metric, epoch_used, total_epochs

    def _final_metric(self, parsed: Dict[str, Any], split: str) -> Optional[float]:
        info_list = parsed.get('info')
        record = info_list[0] if isinstance(info_list, list) and info_list and isinstance(info_list[0], dict) else {}

        if split == 'val':
            for key in ('val_accuracy_final', 'val_accuracy'):
                value = record.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        elif split == 'test':
            value = parsed.get('test_accuracy')
            if value is None and isinstance(record, dict):
                value = record.get('test_accuracy')
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

        return None

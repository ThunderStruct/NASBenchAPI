"""
Unified API wrappers for NASBench-101/201/301.

This module provides wrappers around the benchmark-specific implementations 
to expose a consistent interface across all benchmarks.
"""

import random
from typing import Any, Dict, Iterator, List, Optional, Set

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

    def available_budgets(self, dataset: Optional[str] = None,
                          split: Optional[str] = None) -> Optional[List[Any]]:
        """NB101 does not track explicit training budgets."""
        return None

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

    def _supports_arch(self, arch: Arch101) -> bool:
        """Check if the architecture exists in the loaded dataset."""
        if not self.api.is_valid(arch):
            return False
        latest = self.api.data.get('latest_by_arch', {})
        arch_id = self.api.id(arch)
        if arch_id in latest:
            return True
        enc = self.api.encode(arch)
        for last in latest.values():
            if (last.get('adjacency_str') == enc.get('adjacency_str') and
                    last.get('operations_str') == enc.get('operations_str')):
                return True
        return False


class NASBench201(NASBenchBase):
    """Unified NB201 API wrapper.

    Delegates all operations to the underlying NASBench201 implementation.
    """

    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB201(data_path, verbose=verbose)
        self._budget_cache: Optional[Dict[str, Dict[str, List[int]]]] = None

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

    def available_budgets(self, dataset: Optional[str] = None,
                          split: Optional[str] = None) -> Optional[List[int]]:
        """Return available training budgets for NB201."""
        self._ensure_budget_cache()
        if not self._budget_cache:
            return None

        if dataset is None:
            combined: set = set()
            for ds_budgets in self._budget_cache.values():
                if split is None:
                    for values in ds_budgets.values():
                        combined.update(values)
                else:
                    combined.update(ds_budgets.get(split, []))
            return sorted(combined) if combined else None

        canonical = self._canonical_dataset_name(dataset)
        if canonical is None or canonical not in self._budget_cache:
            return None

        ds_budgets = self._budget_cache[canonical]
        if split is None:
            combined = set()
            for values in ds_budgets.values():
                combined.update(values)
            return sorted(combined) if combined else None

        values = ds_budgets.get(split)
        return list(values) if values else None
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

    def _supports_arch(self, arch: Any) -> bool:
        """Check whether an architecture exists in the loaded NB201 dataset."""
        if isinstance(arch, int):
            if self.api._arch_keys:
                return arch in self.api._arch_keys
            max_idx = (self.api.NUM_OPS ** self.api.NUM_EDGES) - 1
            return 0 <= arch <= max_idx

        if isinstance(arch, str):
            try:
                arch_idx = self.api.get_index(arch)
            except Exception:
                return False
            if self.api._arch_keys:
                return arch_idx in self.api._arch_keys
            return arch_idx is not None

        return False

    # --- Budget helpers -----------------------------------------------------
    def _ensure_budget_cache(self) -> None:
        if self._budget_cache is not None:
            return

        cache: Dict[str, Dict[str, Set[int]]] = {}
        data = getattr(self.api, 'data', {})
        if not isinstance(data, dict):
            self._budget_cache = {}
            return

        arch2infos = data.get('arch2infos', {})
        if not isinstance(arch2infos, dict) or not arch2infos:
            self._budget_cache = {}
            return

        for arch_info in arch2infos.values():
            if not isinstance(arch_info, dict):
                continue
            full = arch_info.get('full')
            if not isinstance(full, dict):
                continue
            all_results = full.get('all_results')
            if not isinstance(all_results, dict):
                continue

            for (dataset_key, _seed), result in all_results.items():
                if not isinstance(result, dict):
                    continue
                canonical = self._canonical_dataset_name(dataset_key)
                if canonical is None:
                    continue
                store = cache.setdefault(canonical, {})
                self._accumulate_budgets(store, result)

            # Budgets are consistent across architectures; first populated entry is enough.
            if cache:
                break

        # Convert sets to sorted lists for stability
        normalized_cache: Dict[str, Dict[str, List[int]]] = {}
        for ds, split_map in cache.items():
            normalized_cache[ds] = {}
            for split, values in split_map.items():
                if values:
                    normalized_cache[ds][split] = sorted(values)
        self._budget_cache = normalized_cache

    def _canonical_dataset_name(self, dataset: Optional[str]) -> Optional[str]:
        if dataset is None:
            return None
        key = str(dataset).lower()
        if key.startswith('cifar10'):
            return 'cifar10'
        if key.startswith('cifar100'):
            return 'cifar100'
        if key.startswith('imagenet16-120'):
            return 'ImageNet16-120'
        return dataset if isinstance(dataset, str) else None

    def _accumulate_budgets(self, store: Dict[str, Set[int]], result: Dict[str, Any]) -> None:
        eval_acc = result.get('eval_acc1es', {})
        if isinstance(eval_acc, dict):
            for metric_key in eval_acc.keys():
                if not isinstance(metric_key, str):
                    continue
                if metric_key.startswith('x-valid@'):
                    budget = self._parse_budget_suffix(metric_key, 'x-valid@')
                    if budget is not None:
                        store.setdefault('val', set()).add(budget)
                elif metric_key.startswith('ori-test@'):
                    budget = self._parse_budget_suffix(metric_key, 'ori-test@')
                    if budget is not None:
                        store.setdefault('test', set()).add(budget)

        train_acc = result.get('train_acc1es')
        if isinstance(train_acc, dict):
            budgets = []
            for key, value in train_acc.items():
                try:
                    budgets.append(int(key))
                except (TypeError, ValueError):
                    continue
            if budgets:
                store.setdefault('train', set()).update(budgets)
        elif isinstance(train_acc, (list, tuple)):
            store.setdefault('train', set()).update(range(len(train_acc)))

    @staticmethod
    def _parse_budget_suffix(metric_key: str, prefix: str) -> Optional[int]:
        try:
            return int(metric_key.replace(prefix, '', 1))
        except ValueError:
            return None


class NASBench301(NASBenchBase):
    """Unified NB301 API wrapper.

    Delegates all operations to the underlying NASBench301 implementation.
    """

    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        self.api = _NB301(data_path, verbose=verbose)
        self._budget_cache: Optional[Dict[str, Dict[str, List[int]]]] = None

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

    def available_budgets(self, dataset: Optional[str] = None,
                          split: Optional[str] = None) -> Optional[List[int]]:
        """Return available budgets derived from NB301 learning curves."""
        self._ensure_budget_cache()
        if not self._budget_cache:
            return None

        if dataset is None:
            combined: Set[int] = set()
            for ds_budgets in self._budget_cache.values():
                if split is None:
                    for values in ds_budgets.values():
                        combined.update(values)
                else:
                    combined.update(ds_budgets.get(split, []))
            return sorted(combined) if combined else None

        ds_budgets = self._budget_cache.get(dataset)
        if ds_budgets is None:
            return None

        if split is None:
            combined: Set[int] = set()
            for values in ds_budgets.values():
                combined.update(values)
            return sorted(combined) if combined else None

        values = ds_budgets.get(split)
        return list(values) if values else None

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

    def _supports_arch(self, arch: Any) -> bool:
        """Check if an architecture or index exists in the NB301 payload."""
        if isinstance(arch, int):
            if self.api._arch_keys:
                return 0 <= arch < len(self.api._arch_keys)
            return False

        if isinstance(arch, dict):
            idx = self.api.get_index(arch)
            if idx is not None:
                return True

        return False

    def _ensure_budget_cache(self) -> None:
        if self._budget_cache is not None:
            return

        cache: Dict[str, Dict[str, Set[int]]] = {}
        data = getattr(self.api, 'data', {})
        entries = data.get('entries') if isinstance(data, dict) else None
        if not isinstance(entries, list):
            self._budget_cache = {}
            return

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            parsed = entry.get('parsed', {})
            dataset_name = self.api._infer_dataset(parsed)
            if not dataset_name:
                continue
            store = cache.setdefault(dataset_name, {})
            val_curve = self.api._get_learning_curve(parsed, 'Train/val_accuracy')
            if val_curve:
                store.setdefault('val', set()).update(range(1, len(val_curve) + 1))
            budget = parsed.get('budget')
            try:
                if budget is not None:
                    store.setdefault('test', set()).add(int(budget))
            except (TypeError, ValueError):
                pass

        normalized: Dict[str, Dict[str, List[int]]] = {}
        for ds, split_map in cache.items():
            normalized[ds] = {}
            for split, values in split_map.items():
                if values:
                    normalized[ds][split] = sorted(values)
        self._budget_cache = normalized

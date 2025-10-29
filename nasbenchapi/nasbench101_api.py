
import base64
import pickle
import random
import hashlib
import json
import struct
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Iterator, Tuple, Union

from .utils import resolve_path, sizeof_fmt, display_path

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
        self._arch_lookup: Dict[Tuple[str, str], str] = {}
        self._load()

    def _load(self) -> None:
        start_time = time.perf_counter()
        size = self.path.stat().st_size
        if self.verbose:
            print(f"Loading NB101 from {display_path(self.path)} ({sizeof_fmt(size)})")
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
            elapsed = time.perf_counter() - start_time
            print(f"[NB101] Loaded {arch_count} architectures{extra} in {elapsed:.2f}s")
        latest = self.data.get('latest_by_arch', {})
        # Cache adjacency/operations lookups for O(1) queries
        self._arch_lookup = {
            (entry.get('adjacency_str'), entry.get('operations_str')): key
            for key, entry in latest.items()
            if isinstance(entry, dict)
        }

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
              seed: Optional[int] = None, budget: Optional[Any] = None,
              average: bool = False, summary: bool = False
              ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[int, Any]]]:
        """Query performance metrics for an architecture from loaded data.

        Args:
            arch: Architecture to query.
            dataset: Dataset name ('cifar10').
            split: Split name ('val', 'test', 'train').
            seed: Optional seed for reproducibility.
            budget: Optional budget specification.
            average: If True, average metrics across the three training runs.
            summary: If True, return the legacy summary dict (metric/metric_name/...).

        Returns:
            When summary=False (default): tuple (info_dict, metrics_by_budget)
            aligned with the original NASBench-101 API. When summary=True, the
            condensed dictionary from prior versions is returned for backwards
            compatibility.
        """
        if summary:
            return self._query_summary(arch, dataset, split, seed, budget)

        enc = self.encode(arch)
        key = self._arch_lookup.get((enc['adjacency_str'], enc['operations_str']))
        if key is None:
            # Fallback: linear scan (should rarely trigger, but keeps parity with previous behaviour)
            latest = self.data.get('latest_by_arch', {})
            for k, last in latest.items():
                if (last.get('adjacency_str') == enc['adjacency_str'] and
                        last.get('operations_str') == enc['operations_str']):
                    key = k
                    break
        if key is None:
            info = {
                'module_adjacency': arch.adjacency,
                'module_operations': arch.operations,
                'module_hash': None,
                'trainable_parameters': None,
                'training_time': None,
            }
            return info, {}

        entries = self.data.get('entries_by_arch', {}).get(key, [])
        if not entries:
            info = {
                'module_adjacency': arch.adjacency,
                'module_operations': arch.operations,
                'module_hash': key,
                'trainable_parameters': None,
                'training_time': None,
            }
            return info, {}

        metrics_by_budget: Dict[int, List[Dict[str, Optional[float]]]] = defaultdict(list)
        trainable_parameters: Optional[float] = None
        total_training_time: List[float] = []

        for record in entries:
            epoch = int(record.get('epoch', 0))
            decoded = self._decode_metrics(record)
            if not decoded:
                continue
            metrics_by_budget[epoch].append(decoded)
            final_time = decoded.get('final_training_time')
            if final_time is not None:
                total_training_time.append(final_time)
            if trainable_parameters is None:
                derived = record.get('derived', {})
                tp = derived.get('trainable_parameters')
                if tp is not None:
                    trainable_parameters = float(tp)

        info = {
            'module_adjacency': arch.adjacency,
            'module_operations': arch.operations,
            'module_hash': key,
            'trainable_parameters': trainable_parameters,
            'training_time': float(sum(total_training_time) / len(total_training_time))
            if total_training_time else None,
        }

        # Ensure deterministic ordering by sorting budget keys
        ordered_metrics: Dict[int, Any] = {}
        for epoch in sorted(metrics_by_budget.keys()):
            runs = metrics_by_budget[epoch]
            if average:
                ordered_metrics[epoch] = self._average_metrics(runs)
            else:
                ordered_metrics[epoch] = runs

        return info, ordered_metrics

    def _query_summary(self, arch: Arch101, dataset: str = 'cifar10', split: str = 'val',
                       seed: Optional[int] = None, budget: Optional[Any] = None) -> Dict[str, Any]:
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

    @staticmethod
    def _decode_metrics(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Decode the base64 metrics blob into halfway/final metrics."""
        metrics_b64 = record.get('metrics_b64')
        if not metrics_b64:
            return {}

        try:
            raw = base64.b64decode(metrics_b64)
        except Exception:
            return {}

        steps: Dict[int, Dict[int, float]] = {}
        idx = 0
        length = len(raw)
        while idx < length:
            tag = raw[idx]
            idx += 1
            if tag != 0x0A:  # Expecting length-delimited field
                continue
            chunk_len, shift = NASBench101._read_varint(raw, idx)
            idx += shift
            chunk = raw[idx:idx + chunk_len]
            idx += chunk_len
            if not chunk:
                continue

            cursor = 0
            fields: Dict[int, float] = {}
            while cursor < len(chunk):
                field_tag = chunk[cursor]
                cursor += 1
                field_number = field_tag >> 3
                wire_type = field_tag & 0x7
                if wire_type == 1:  # 64-bit
                    if cursor + 8 > len(chunk):
                        break
                    value = struct.unpack('<d', chunk[cursor:cursor + 8])[0]
                    cursor += 8
                elif wire_type == 5:  # 32-bit
                    if cursor + 4 > len(chunk):
                        break
                    value = float(struct.unpack('<f', chunk[cursor:cursor + 4])[0])
                    cursor += 4
                else:
                    # Unsupported wire type; stop parsing this chunk
                    break
                fields[field_number] = value
            if 1 in fields:
                step_key = int(round(fields[1]))
                steps[step_key] = fields

        epoch = int(record.get('epoch', 0))
        halfway = epoch // 2 if epoch else None

        def extract(step: Optional[Dict[int, float]]) -> Dict[str, Optional[float]]:
            if not step:
                return {
                    'training_time': None,
                    'train_accuracy': None,
                    'validation_accuracy': None,
                    'test_accuracy': None,
                }
            return {
                'training_time': float(step.get(2)) if step.get(2) is not None else None,
                'train_accuracy': float(step.get(3)) if step.get(3) is not None else None,
                'validation_accuracy': float(step.get(4)) if step.get(4) is not None else None,
                'test_accuracy': float(step.get(5)) if step.get(5) is not None else None,
            }

        halfway_metrics = extract(steps.get(halfway)) if halfway is not None else extract(None)
        final_metrics = extract(steps.get(epoch))

        return {
            'halfway_training_time': halfway_metrics['training_time'],
            'halfway_train_accuracy': halfway_metrics['train_accuracy'],
            'halfway_validation_accuracy': halfway_metrics['validation_accuracy'],
            'halfway_test_accuracy': halfway_metrics['test_accuracy'],
            'final_training_time': final_metrics['training_time'],
            'final_train_accuracy': final_metrics['train_accuracy'],
            'final_validation_accuracy': final_metrics['validation_accuracy'],
            'final_test_accuracy': final_metrics['test_accuracy'],
        }

    @staticmethod
    def _read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
        """Read a little-endian varint from buffer starting at offset.

        Returns:
            Tuple (value, bytes_consumed).
        """
        result = 0
        shift = 0
        idx = offset
        while idx < len(buffer):
            byte = buffer[idx]
            idx += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result, idx - offset
            shift += 7
        return result, idx - offset

    @staticmethod
    def _average_metrics(runs: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
        """Average metric dictionaries while ignoring missing values."""
        if not runs:
            return {}
        keys = runs[0].keys()
        averaged: Dict[str, Optional[float]] = {}
        for key in keys:
            values = [r[key] for r in runs if r.get(key) is not None]
            averaged[key] = float(sum(values) / len(values)) if values else None
        return averaged

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

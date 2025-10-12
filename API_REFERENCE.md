# NASBenchAPI Reference

This document summarizes the unified API surface for NAS-Bench-101/201/301, including method usage, arguments, return values, and benchmark metadata (datasets, epochs, metrics).

## Benchmarks Overview

| Benchmark       | Datasets                                 | Primary Metrics                     | Training Epochs           |
|-----------------|-------------------------------------------|-------------------------------------|---------------------------|
| NAS-Bench-101   | CIFAR-10                                  | train/val/test accuracy             | 4, 12, 36, 108            |
| NAS-Bench-201   | CIFAR-10, CIFAR-100, ImageNet16-120       | train/val/test accuracy, losses     | 12, 200                   |
| NAS-Bench-301   | CIFAR-10, CIFAR-100                       | surrogate val/test accuracy         | N/A (surrogate)           |

Notes:
- NAS-Bench-301 relies on surrogate models; epochs are not applicable.

## Common API Surface (all benchmarks)

All benchmarks expose the following methods via `nasbenchapi.nb_api` classes: `NASBench101`, `NASBench201`, and `NASBench301`.

### load
```python
api = NASBench101().load('/path/to/nb101.pkl')
```
Args:
- data_path: Optional[str] — explicit dataset path; if omitted, env var is used.

Returns: self instance

_Note: env vars are set as follows:_

```python
[
    'NASBENCH101_PATH',
    'NASBENCH201_PATH',
    'NASBENCH301_PATH'
]
```

### close
```python
api.close()
```
Args: none

Returns: None

### set_seed
```python
api.set_seed(42)
```
Args:
- seed: int — RNG seed

Returns: None

### bench_name
```python
name = api.bench_name()
```
Args: none

Returns: str — short name (e.g., 'nb101', 'nb201', 'nb301')

### datasets
```python
ds = api.datasets()
```
Args: none

Returns: list[str] — dataset names

### splits
```python
spl = api.splits('cifar10')
```
Args:
- dataset: str — dataset name

Returns: list[str] — supported splits (['train', 'val', 'test'])

### decode
```python
arch = api.decode(encoding)
```
Args:
- encoding: Any — benchmark-native representation

Returns: Any — decoded architecture object

### encode
```python
encoding = api.encode(arch)
```
Args:
- arch: Any — architecture object

Returns: Any — benchmark-native encoding

### id
```python
arch_id = api.id(arch)
```
Args:
- arch: Any — architecture object

Returns: str — stable identifier

### random_sample
```python
samples = api.random_sample(n=5, seed=123)
```
Args:
- n: int — number of samples (default 1)
- seed: Optional[int] — RNG seed

Returns: list[Any] — sampled architectures

### iter_all
```python
for arch in api.iter_all():
    ...
```
Args: none

Returns: Iterator[Any] — iterate all available architectures (if supported)

### mutate
```python
mut = api.mutate(arch, rng=random.Random(0), kind='edge_toggle')
```
Args:
- arch: Any — architecture to mutate
- rng: random.Random — RNG instance
- kind: Optional[str] — mutation kind (benchmark-defined)

Returns: Any — mutated architecture

## NAS-Bench-101 Specifics

Import:
```python
from nasbenchapi import NASBench101
api = NASBench101('/path/to/nb101.pkl')
```

Architecture type (Arch101):
- adjacency: list[list[int]] — 7x7 adjacency matrix
- operations: list[str] — 7 operations

NB101-specific methods:

### op_set
```python
ops = api.op_set()
```
Args: none

Returns: list[str] — available operations (e.g., input, conv3x3-bn-relu, ...)

### evaluate
```python
res = api.evaluate(arch, dataset='cifar10', split='val')
```
Args:
- arch: Arch101 — architecture to evaluate
- dataset: str — dataset name
- split: str — 'val' (or 'test' if available)
- seed: Optional[int] — reproducibility seed
- budget: Optional[Any] — optional budget (unused)

Returns: dict — keys: 
- metric: float | None — e.g., validation accuracy if available
- metric_name: str — e.g., 'val_acc'
- cost: float | None — e.g., training time if available
- std: float | None — predictive std (not used in NB101)
- info: dict — raw record info

### train_time
```python
t = api.train_time(arch, dataset='cifar10')
```
Args:
- arch: Arch101 — architecture
- dataset: str — dataset name

Returns: float | None — training time if available

Encoding/Decoding examples:
```python
# Encode
enc = api.encode(arch)
# Decode
arch2 = api.decode(enc)
# Stable ID
hid = api.id(arch)
```

## NAS-Bench-201 Specifics

Import:
```python
from nasbenchapi import NASBench201
api = NASBench201('/path/to/nb201.pkl')
```

Datasets: ['cifar10', 'cifar100', 'ImageNet16-120']

Current behavior:
- decode/encode/id: pass-through placeholders until canonical representation is defined
- random_sample/iter_all: empty until enumeration is defined
- mutate: no-op
- evaluate: placeholder dict with None metric

## NAS-Bench-301 Specifics

Import:
```python
from nasbenchapi import NASBench301
api = NASBench301('/path/to/nb301.pkl')
```

Datasets: ['cifar10', 'cifar100']

Current behavior:
- decode/encode/id: pass-through placeholders
- random_sample/iter_all: empty until enumeration is defined
- mutate: no-op
- evaluate: placeholder dict (surrogate model not bundled)



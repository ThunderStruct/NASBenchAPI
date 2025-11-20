
# NASBenchAPI

[![pypi](https://img.shields.io/badge/pypi%20package-1.0.3-lightgrey.svg)](https://pypi.org/project/nasbenchapi/) [![Platform](https://img.shields.io/badge/python-v3.8+-green)](https://github.com/ThunderStruct/NASBenchAPI) [![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/ThunderStruct/NASBenchAPI/blob/main/LICENSE) [![Read the Docs](https://readthedocs.org/projects/nasbenchapi/badge/?version=latest)](https://nasbenchapi.readthedocs.io/en/latest/)


A unified, lightweight interface for NASBench-101, 201, and 301 with optimized Pickle-based datasets.

 
------------------------

  

## Getting Started

  

**NASBenchAPI** is a lightweight, unified interface for Neural Architecture Search benchmarks (101, 201, and 301). All NASBench datasets (originally in `.tfrecord`, `.pth`, and `.json` formats) were extracted and saved as Pickle-based files for consistency.


### Related Works


This project is inspired by the holistic NAS Library, [NASLib](https://github.com/automl/NASLib), and the paper by [Mehta et al.](https://openreview.net/forum?id=0DLwqQLmqV).


The primary motivation for NASBenchAPI stems from the need to integrate NASBench datasets (101, 201, 301) into custom frameworks without the significant overhead and extraneous tools introduced by more comprehensive libraries. This API provides a focused, lightweight, and unified interface specifically for that purpose.


### Installation

  

#### PyPi (recommended)

  

The Python package is hosted on the [Python Package Index (PyPI)](https://pypi.org/project/nasbenchapi/).

  

The latest published version of NASBenchAPI can be installed using

  

```sh
pip install nasbenchapi
```

  

#### Manual Installation

Simply clone the entire repo and extract the files in the `nasbenchapi` folder, then import them into your project folder.

  

Or use one of the shorthand methods below

##### GIT

-  `cd` into your project directory

- Use `sparse-checkout` to pull the library files only into your project directory

```sh
git init nasbenchapi

cd nasbenchapi

git remote add -f origin https://github.com/ThunderStruct/NASBenchAPI.git
git config core.sparseCheckout true

echo "nasbenchapi/*"  >> .git/info/sparse-checkout

git pull --depth=1  origin  main
```

- Import the newly pulled files into your project folder

##### SVN

-  `cd` into your project directory

-  `checkout` the library files

```sh
svn checkout https://github.com/ThunderStruct/NASBenchAPI/trunk/nasbenchapi
```

- Import the newly checked out files into your project folder

  

### Quick Start

  

#### Basic Usage

  
#####  Loading and initializing a benchmark
```python
from nasbenchapi import NASBench101, NASBench201, NASBench301

# Initialize with explicit path

nb101 = NASBench101('/path/to/nb101.pkl')  # Same for 201, 301

# Or use environment variables
# export NASBENC2101_PATH=/path/to/nb201.pkl

nb201 =  NASBench201()

```

##### Sample random architectures

```python
archs = nb101.random_sample(n=5,  seed=42)    # randomly sample 5 architectures

print(f"Sampled {len(archs)} architectures")
```

##### Query performance of an architecture

```python
arch = archs[0]

# Tuple result: (info_dict, metrics_by_budget)
info, metrics = nb101.query(arch, dataset='cifar10', split='val')

# Accessing the final run at the 108-epoch budget
final_val = metrics[108][-1]['final_validation_accuracy']
print(f"Validation accuracy @108 epochs: {final_val}")

# Legacy condensed dict (metric / cost / info)
summary = nb101.query(arch, dataset='cifar10', split='val', summary=True)
print(f"Summary metric: {summary['metric']}")

```

##### Iterate over all architectures

```python
for i, arch in  enumerate(nb101.iter_all()):
    if i >=  10:
        break
    print(f"Architecture {i}: {nb101.id(arch)}")

```


## Benchmark Reference

### NASBench-101

- **Dataset format**: Converted from the original TensorFlow TFRecord into a Pickle for faster loading (up to 20x faster) and compatibility with modern libraries (does not depend on TF1.x).
- **Budgets**: Validation/test metrics are available at epochs 4, 12, 36, and 108.
- **Query return shape**:
  - Default: tuple ``(info_dict, metrics_by_budget)`` where each budget maps to a list of raw run dictionaries (`halfway_*`, `final_*` keys).
  - ``average=True`` collapses runs per budget; ``summary=True`` restores the legacy dict with ``metric``, ``metric_name``, ``cost``, ``std``, ``info``.

```python
from nasbenchapi import NASBench101, Arch101

nb101 = NASBench101('/path/to/nasbench101_full.pkl', verbose=False)
arch = nb101.random_sample(n=1, seed=0)[0]

info, metrics = nb101.query(arch, dataset='cifar10', split='val')
avg_metrics = nb101.query(arch, dataset='cifar10', split='val', average=True)[1]
summary = nb101.query(arch, dataset='cifar10', split='val', summary=True)

print(info['module_hash'])
print(metrics[108][-1]['final_test_accuracy'])
print(summary['metric'])
```

### NASBench-201

- **Dataset format**: Official PyTorch checkpoint (`NASBench-201-v1_1-096897.pth`) re-serialized to pickle with cached index ↔ string mappings.
- **Budgets**: Epochs 0–199 (commonly query 12 for early and 199 for final results) across CIFAR-10, CIFAR-100, and ImageNet16-120.
- **Query return shape**: dict with ``metric``, ``metric_name``, ``cost``, ``std``, and ``info`` (contains architecture index, arch string, dataset, split, seed, epoch, params, FLOPs).

```python
from nasbenchapi import NASBench201

nb201 = NASBench201('/path/to/nasbench201.pkl', verbose=False)
arch_str = nb201.random_sample(n=1, seed=7)[0]

result = nb201.query(arch_str, dataset='cifar10', split='val', budget=199)
print(result['metric'])
print(result['info']['arch_str'])
```

### NASBench-301

- **Dataset format**: The original directory of JSON surrogate models has been flattened into a single pickle for faster access; indices map directly to entries.
- **Budgets**: Validation budgets come from learning-curve lengths (typically 1–98 epochs for CIFAR-10/CIFAR-100); test metrics expose the declared training budget.
- **Query return shape**: dict with ``metric``, ``metric_name``, ``cost``, ``std``, and ``info`` (including entry index, dataset, optimizer tag, epochs available/used, JSON source path).

```python
from nasbenchapi import NASBench301

nb301 = NASBench301('/path/to/nasbench301.pkl', verbose=False)
idx = nb301.random_sample(n=1, seed=1)[0]

val_final = nb301.query(idx, dataset='cifar10', split='val')
val_epoch50 = nb301.query(idx, dataset='cifar10', split='val', budget=50)
test_final = nb301.query(idx, dataset='cifar10', split='test')

print(val_final['metric'], val_epoch50['metric'], test_final['metric'])
```

  

### Dataset Management

**Environment Variables (recommended)**

  

Set environment variables to avoid passing paths explicitly and work seamlessly across different projects:

```bash
export NASBENCH101_PATH=/path/to/nb101.pkl
export NASBENCH201_PATH=/path/to/nb201.pkl
export NASBENCH301_PATH=/path/to/nb301.pkl
```


**CLI Downloader (recommended)**

Download the Pickle-based benchmark datasets through the CLI:

```bash
nasbench-download
```

You may optionally set the `--benchmark={101|201|301}` argument. Otherwise, the tool will prompt for benchmark selection interactively.


**Manual Download**

Alternatively, manually download the Pickle-based benchmarks through the following links:

| Benchmark | Download Link |
|-----------|---------------|
| **NASBench-101** | [Figshare Link](https://figshare.com/ndownloader/files/59722685) |
| **NASBench-201** | [Figshare Link](https://figshare.com/ndownloader/files/58862743) |
| **NASBench-301** | [Figshare Link](https://figshare.com/ndownloader/files/58862737) |


### Documentation

Detailed examples and the full API docs are [hosted on Read the Docs](https://nasbenchapi.readthedocs.io/en/latest/).
  

## Benchmarks at a Glance

  

| Benchmark | Datasets | Metrics | Search Space Size |
|-----------|----------|---------|-------------------|
| **NASBench-101** | CIFAR-10 | train/val/test accuracy, training time | 423,624 |
| **NASBench-201** | CIFAR-10, CIFAR-100, ImageNet16-120 | train/val/test accuracy, losses | 15,625 |
| **NASBench-301** | CIFAR-10, CIFAR-100 | surrogate val/test accuracy | ~10^18 (surrogate) |
  


## Cite

If you use this library in your work, please use the following BibTeX entry:

```bibtex
@misc{nasbenchapi-2025, 
  title={NASBenchAPI: A unified interface for NASBench datasets}, 
  author={Shahawy, Mohamed}, 
  year={2025}, 
  publisher={GitHub}, 
  howpublished={\url{https://github.com/ThunderStruct/NASBenchAPI}} 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ThunderStruct/NASBenchAPI/blob/main/LICENSE) file for details

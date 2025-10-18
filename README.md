
# NASBenchAPI

[![pypi](https://img.shields.io/badge/pypi%20package-0.1.0-lightgrey.svg)](https://pypi.org/project/nasbenchapi/) [![Platform](https://img.shields.io/badge/python-v3.8+-green)](https://github.com/ThunderStruct/NASBenchAPI) [![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/ThunderStruct/NASBenchAPI/blob/main/LICENSE) [![Read the Docs](https://readthedocs.org/projects/nasbenchapi/badge/?version=latest)](https://nasbenchapi.readthedocs.io/en/latest/)


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

result = nb101.query(arch,  dataset='cifar10',  split='val')

print(f"Validation accuracy: {result['metric']}")
print(f"Training time: {result['cost']}")

```

##### Iterate over all architectures

```python
for i, arch in  enumerate(nb101.iter_all()):
    if i >=  10:
        break
    print(f"Architecture {i}: {nb101.id(arch)}")
    
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

### Documentation

Detailed examples and the full API docs are [hosted on Read the Docs](https://nasbenchapi.readthedocs.io/en/latest/).
  

## Benchmarks at a Glance

  

| Benchmark | Datasets | Metrics | Search Space Size |
|-----------|----------|---------|-------------------|
| **NAS-Bench-101** | CIFAR-10 | train/val/test accuracy, training time | 423,624 |
| **NAS-Bench-201** | CIFAR-10, CIFAR-100, ImageNet16-120 | train/val/test accuracy, losses | 15,625 |
| **NAS-Bench-301** | CIFAR-10, CIFAR-100 | surrogate val/test accuracy | ~10^18 (surrogate) |
  


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


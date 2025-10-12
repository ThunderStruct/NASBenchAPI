NASBenchAPI

Lightweight, unified pickle-based APIs for NAS-Bench-101, 201, and 301 with an optional downloader. This library focuses on fast loading of converted 1-to-1 pickle datasets and a consistent interface across benchmarks.

Features
- Unified APIs for NB101, NB201, NB301 (pickle-only)
- Environment variable path detection
- Optional nasbench-download CLI with y/n prompt
- tqdm progress bars for loading

Installation
```bash
pip install -e .
```

Environment Variables
- NASBENCH{1|2|3}01_PATH – path to the respective NASBench PKL file

Usage
```python
from nasbenchapi import NASBench101, NASBench201, NASBench301

nb101 = NASBench101('/path/to/nb101.pkl')
nb201 = NASBench201('/path/to/nb201.pkl')
nb301 = NASBench301('/path/to/nb301.pkl')

stats = nb101.get_statistics()
print(stats)
```

CLI
```bash
nasbench-download
```

Manual Download Locations
- NB101: https://figshare.com/placeholder/nb101.pkl
- NB201: https://figshare.com/placeholder/nb201.pkl
- NB301: https://figshare.com/placeholder/nb301.pkl

License
MIT


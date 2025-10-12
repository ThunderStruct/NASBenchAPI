
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

# Optional dependencies to avoid pip overhead
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .common import resolve_path, sizeof_fmt


class NASBench101:
    """NAS-Bench-101 API.

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
            print(f"Loaded {len(self.data.get('entries_by_arch', {}))} architectures")

    def get_statistics(self) -> Dict[str, Any]:
        entries = self.data.get('entries_by_arch', {})
        return {
            'benchmark': 'nasbench101',
            'architectures': len(entries),
            'records': self.data.get('num_records', 0),
        }


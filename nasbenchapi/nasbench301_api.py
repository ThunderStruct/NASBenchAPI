
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


class NASBench301:
    """NAS-Bench-301 API.

    Supports directory-converted JSON payloads or reserialized pickle.
    """

    def __init__(self, pickle_path: Optional[str] = None, verbose: bool = True):
        self.path = resolve_path('301', pickle_path)
        self.verbose = verbose
        self.data: Any = None
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



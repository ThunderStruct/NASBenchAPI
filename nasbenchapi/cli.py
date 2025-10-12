import os
import sys
import argparse
from pathlib import Path
import requests
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

FIGSHARE_URLS = {
    '101': 'https://figshare.com/placeholder/nb101.pkl',
    '201': 'https://figshare.com/placeholder/nb201.pkl',
    '301': 'https://figshare.com/placeholder/nb301.pkl',
}

ENV_VARS = {
    '101': 'NASBENCH101_PATH',
    '201': 'NASBENCH201_PATH',
    '301': 'NASBENCH301_PATH',
}


def _prompt_yes_no(msg: str) -> bool:
    resp = input(f"{msg} (y/n): ").strip().lower()
    return resp in {"y", "yes"}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        if HAS_TQDM and total > 0:
            bar = tqdm(total=total, unit='B', unit_scale=True, desc=f'Downloading {dst.name}')
        else:
            bar = None
        with open(dst, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        if bar is not None:
            bar.close()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="NASBenchAPI downloader")
    parser.add_argument('--benchmark', choices=['101', '201', '301'], required=False,
                        help='Benchmark to download (if omitted, prompts)')
    parser.add_argument('--output', type=str, required=False,
                        help='Output path for the pickle file (defaults to env var location or ./datasets)')
    args = parser.parse_args(argv)

    bench = args.benchmark
    if bench is None:
        bench = input("Which benchmark to download? (101/201/301): ").strip()
    if bench not in FIGSHARE_URLS:
        print("Invalid benchmark. Choose from 101, 201, 301.")
        sys.exit(1)

    env = ENV_VARS[bench]
    default_dir = Path.cwd() / 'datasets'
    if args.output:
        out_path = Path(args.output)
    else:
        # Prefer env var; if set to a directory, drop file into it; otherwise create default dir
        env_val = os.environ.get(env)
        if env_val:
            p = Path(env_val)
            if p.is_dir() or (bench == '301' and (not p.suffix or p.suffix.lower() != '.pkl')):
                out_path = p / f"nasbench{bench}.pkl"
            else:
                out_path = p
        else:
            out_path = default_dir / f"nasbench{bench}.pkl"

    print(f"Target file: {out_path}")
    if not _prompt_yes_no(f"Download NAS-Bench-{bench} from Figshare to this location?"):
        print("Aborted by user.")
        sys.exit(0)

    try:
        _download(FIGSHARE_URLS[bench], out_path)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print("Download completed.")


if __name__ == '__main__':
    main()



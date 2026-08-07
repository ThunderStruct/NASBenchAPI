import sys
import argparse
from typing import Optional

from .download import download_benchmark, resolve_download_path


def _prompt_yes_no(msg: str) -> bool:
    resp = input(f"{msg} (y/n): ").strip().lower()
    return resp in {"y", "yes"}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="NASBenchAPI downloader")
    parser.add_argument('--benchmark',
                        choices=['101', '201', '301'],
                        required=False,
                        help='Benchmark to download (if omitted, prompts)')
    parser.add_argument('--output', type=str, required=False,
                        help='Output path (file or directory). If directory, '
                             'filename is auto-appended. Defaults to env var '
                             'location or ./datasets')
    args = parser.parse_args(argv)

    bench = args.benchmark
    if bench is None:
        bench = input("Which benchmark to download? (101/201/301): ").strip()
    if bench not in ['101', '201', '301']:
        print("Invalid benchmark. Choose from 101, 201, 301.")
        sys.exit(1)

    out_path = resolve_download_path(bench, args.output)

    print(f"Target file: {out_path}")
    if not _prompt_yes_no(
            f"Download NASBench-{bench} from Hugging Face to this location?"):
        print("Aborted by user.")
        sys.exit(0)

    try:
        download_benchmark(bench, output=str(out_path), force=True)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print("Download completed.")


if __name__ == '__main__':
    main()

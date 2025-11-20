"""
Utility script to convert the original NAS-Bench-101 TFRecord
(`nasbench_full.tfrecord`) into the pickle format expected by
`nasbenchapi.NASBench101`.

This version keeps all architectures (423,624 for the full dataset),
not only the 7-node subset.

Example:

    python scripts/convert_nb101_tfrecord_to_pickle.py \\
        --tfrecord /path/to/nasbench_full.tfrecord \\
        --output   /path/to/nasbench101_full_all.pkl
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf  # type: ignore
from nasbench.lib import model_metrics_pb2  # type: ignore


def _build_record(epoch: int, raw_adj: str, raw_ops: str, raw_metrics: str) -> Dict[str, Any]:
    """Build a single record entry for entries_by_arch."""
    metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))
    final = metrics.evaluation_data[2]

    derived = {
        "trainable_parameters": int(metrics.trainable_parameters),
        "training_time": float(final.training_time),
        "train_accuracy": float(final.train_accuracy),
        "validation_accuracy": float(final.validation_accuracy),
        "test_accuracy": float(final.test_accuracy),
    }

    return {
        "epoch": int(epoch),
        "adjacency_str": raw_adj,
        "operations_str": raw_ops,
        "metrics_b64": raw_metrics,
        "derived": derived,
    }


def convert_nb101_tfrecord_to_pickle(
    tfrecord_path: Path,
    output_path: Path,
) -> None:
    """Convert NAS-Bench-101 TFRecord to NASBenchAPI pickle format."""
    start = time.time()
    tfrecord_path = tfrecord_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    entries_by_arch: Dict[str, Any] = defaultdict(list)

    num_rows = 0
    for serialized_row in tf.compat.v1.python_io.tf_record_iterator(str(tfrecord_path)):
        num_rows += 1
        module_hash, epochs, raw_adj, raw_ops, raw_metrics = json.loads(
            serialized_row.decode("utf-8")
        )
        record = _build_record(epochs, raw_adj, raw_ops, raw_metrics)
        entries_by_arch[module_hash].append(record)

    # Build latest_by_arch summary focusing on the longest training epoch (108)
    latest_by_arch: Dict[str, Any] = {}
    for module_hash, records in entries_by_arch.items():
        if not records:
            continue

        # All records for a given hash share adjacency/ops
        adj = records[0]["adjacency_str"]
        ops = records[0]["operations_str"]

        by_epoch: Dict[int, list] = defaultdict(list)
        for r in records:
            by_epoch[int(r["epoch"])] = by_epoch.get(int(r["epoch"]), [])
            by_epoch[int(r["epoch"])].append(r)

        # Prefer 108 epochs if available, otherwise the largest epoch key
        epoch_keys = sorted(by_epoch.keys())
        target_epoch = 108 if 108 in by_epoch else epoch_keys[-1]
        target_records = by_epoch[target_epoch]

        # Aggregate derived metrics across repeats for the target epoch
        n = len(target_records)
        # trainable_parameters is constant across repeats
        tp = float(target_records[0]["derived"]["trainable_parameters"])

        def _mean(field: str) -> float:
            return float(sum(r["derived"][field] for r in target_records) / n)

        derived = {
            "trainable_parameters": tp,
            "training_time": _mean("training_time"),
            "train_accuracy": _mean("train_accuracy"),
            "validation_accuracy": _mean("validation_accuracy"),
            "test_accuracy": _mean("test_accuracy"),
        }

        latest_by_arch[module_hash] = {
            "epoch": int(target_epoch),
            "adjacency_str": adj,
            "operations_str": ops,
            # Any of the target_records' metrics_b64 is fine here; API only
            # uses 'derived' for summary queries.
            "metrics_b64": target_records[0]["metrics_b64"],
            "derived": derived,
        }

    num_architectures = len(entries_by_arch)
    num_records = sum(len(v) for v in entries_by_arch.values())

    data = {
        "benchmark": "nasbench101",
        "format": "lossless",
        "entries_by_arch": dict(entries_by_arch),
        "latest_by_arch": latest_by_arch,
        "num_records": int(num_records),
        # Unused by the current API, kept for backwards compatibility.
        "fallback_records": [{} for _ in range(6)],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start
    print(
        f"Converted NAS-Bench-101 TFRecord -> pickle\n"
        f"  Architectures : {num_architectures}\n"
        f"  Records       : {num_records} (rows: {num_rows})\n"
        f"  Output        : {output_path}\n"
        f"  Time          : {elapsed:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert nasbench_full.tfrecord into a pickle usable by "
            "nasbenchapi.NASBench101 (including all architectures)."
        )
    )
    parser.add_argument(
        "--tfrecord",
        type=Path,
        required=True,
        help="Path to nasbench_full.tfrecord",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output pickle file",
    )

    args = parser.parse_args()
    convert_nb101_tfrecord_to_pickle(args.tfrecord, args.output)


if __name__ == "__main__":
    main()


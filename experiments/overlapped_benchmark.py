# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Runs a benchmark's detection and its reconstruction at the same time.

Detection saturates the GPU and reads the recordings; everything after it is
bundle adjustment, which saturates a core and reads only the cache. Run as one
pass they take turns leaving the other idle. This runs detection over the whole
dataset in one process and, alongside it, keeps handing the benchmark whichever
sequences are detected but not yet reconstructed.

Which sequences count as detected is `cache_status`'s question, not this one's.

Whether overlapping pays depends on the machine. Measured on one RTX 3090 with
the recordings on a spinning disk, it did not: reconstruction slowed from 7.6 to
10.6 minutes a sequence because bundle adjustment lost the cores detection was
decoding on, and the run came out slower than a single pass doing both. Measure
before assuming it helps.
"""

import argparse
import os
import subprocess
import sys
import time

from cache_status import (
    DEFAULT_KINDS,
    detected_sequences,
    resolved_config,
    sequence_view_counts,
)


def reconstructed_sequences(metrics_dir: str) -> set[str]:
    """Sequences the benchmark has already produced metrics for."""
    if not os.path.isdir(metrics_dir):
        return set()
    return {f[:-5] for f in os.listdir(metrics_dir) if f.endswith(".json")}


def _eval_command(
    eval_script: str, dataset_dir: str, config: str, sequences: list[str]
) -> list[str]:
    """The command running an evaluation script, over some sequences or all."""
    return [
        sys.executable,
        "-u",
        eval_script,
        dataset_dir,
        "--config-file",
        config,
        *(["--sequences-filter", *sequences] if sequences else []),
    ]


def main(
    dataset_dir: str,
    eval_script: str,
    sequences_file: str,
    cache_config: str,
    benchmark_config: str,
    kinds: tuple[str, ...],
    batch_size: int,
    settle_seconds: float,
    poll_seconds: float,
):
    cache_dir = resolved_config(cache_config).cache_root_dir
    metrics_dir = os.path.join(
        resolved_config(benchmark_config).output_root_dir, "metrics"
    )
    view_counts = sequence_view_counts(sequences_file)

    detector = subprocess.Popen(
        _eval_command(eval_script, dataset_dir, cache_config, [])
    )
    print(f"detecting {len(view_counts)} sequences as pid {detector.pid}", flush=True)

    try:
        while True:
            done = reconstructed_sequences(metrics_dir)
            ready = detected_sequences(cache_dir, view_counts, kinds, settle_seconds)
            pending = sorted(ready - done)

            if not pending:
                if detector.poll() is not None:
                    print(f"detection finished, {len(done)} reconstructed", flush=True)
                    return
                time.sleep(poll_seconds)
                continue

            batch = pending[:batch_size]
            print(
                f"reconstructing {len(batch)} of {len(pending)} pending, "
                f"{len(done)}/{len(view_counts)} done",
                flush=True,
            )
            subprocess.run(
                _eval_command(eval_script, dataset_dir, benchmark_config, batch),
                check=False,
            )
    finally:
        if detector.poll() is None:
            print("stopping detection", flush=True)
            detector.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--eval-script", type=str, required=True)
    parser.add_argument("--sequences-file", type=str, required=True)
    parser.add_argument("--cache-config", type=str, required=True)
    parser.add_argument("--benchmark-config", type=str, required=True)
    parser.add_argument("--expect-kinds", nargs="+", default=list(DEFAULT_KINDS))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sequences per reconstruction pass, before newly detected ones are picked up",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=90.0,
        help="How long a sequence's cache must be untouched before it is read",
    )
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    args = parser.parse_args()
    main(
        args.dataset_dir,
        args.eval_script,
        args.sequences_file,
        args.cache_config,
        args.benchmark_config,
        tuple(args.expect_kinds),
        args.batch_size,
        args.settle_seconds,
        args.poll_seconds,
    )

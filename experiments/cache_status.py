# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""What a detection cache holds, against what its dataset expects of it.

A sequence is detected once every cached kind holds one entry per view of that
sequence. Counting entries alone does not say it: intrinsics are written for
every view before detection starts on the first one, and a run interrupted part
way leaves a kind short without leaving it inconsistent. Only the dataset knows
how many views a sequence has, so it is read from its `sequences.json`.
"""

import argparse
import os
import time

import orjson
from omegaconf import DictConfig, OmegaConf

# The kinds a detection run caches: MoGe writes one, the keypoints detector the
# other. Configs that take either from the ground truth cache only the other.
DEFAULT_KINDS = ("cameras_intrinsics", "keypoints_2d")


def resolved_config(config_file: str) -> DictConfig:
    """Reads a pipeline config with its environment interpolations resolved."""
    cfg = OmegaConf.load(config_file)
    OmegaConf.resolve(cfg)
    return cfg


def sequence_view_counts(sequences_file: str) -> dict[str, int]:
    """How many views each sequence of a dataset has.

    Args:
        sequences_file: A dataset's `sequences.json`, as its preprocessing wrote
            it.

    Returns:
        Maps a sequence name to its number of views.
    """
    with open(sequences_file, "rb") as f:
        return {s["sequence_name"]: len(s["views"]) for s in orjson.loads(f.read())}


def cached_views(cache_dir: str, sequence_name: str) -> tuple[dict[str, int], float]:
    """Views cached per kind for one sequence, and when they were last written.

    Args:
        cache_dir: Directory the detection cache is written under.
        sequence_name: Sequence to look at.

    Returns:
        The number of cached views per kind, and the most recent modification
        time among them, zero when the sequence has nothing cached.
    """
    root = os.path.join(cache_dir, sequence_name)
    if not os.path.isdir(root):
        return {}, 0.0

    counts: dict[str, int] = {}
    newest = 0.0

    for kind in os.scandir(root):
        if not kind.is_dir():
            continue
        views = list(os.scandir(kind.path))
        counts[kind.name] = len(views)
        newest = max([newest] + [v.stat().st_mtime for v in views])

    return counts, newest


def detected_sequences(
    cache_dir: str,
    view_counts: dict[str, int],
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    settle_seconds: float = 0.0,
) -> set[str]:
    """Sequences detected in full and no longer being written to.

    Args:
        cache_dir: Directory the detection cache is written under.
        view_counts: Views each sequence has, as `sequence_view_counts` reads
            them.
        kinds: Kinds a detected sequence must hold one entry per view of.
        settle_seconds: How long a sequence's files must have been untouched,
            so that one still being written is not read.

    Returns:
        The names of the sequences that are ready to be used.
    """
    now = time.time()
    ready = set()

    for name, expected in view_counts.items():
        counts, newest = cached_views(cache_dir, name)
        if all(counts.get(k) == expected for k in kinds) and (
            now - newest >= settle_seconds
        ):
            ready.add(name)

    return ready


def main(
    sequences_file: str,
    cache_config: str,
    kinds: tuple[str, ...],
    settle_seconds: float,
):
    cache_dir = resolved_config(cache_config).cache_root_dir
    view_counts = sequence_view_counts(sequences_file)
    ready = detected_sequences(cache_dir, view_counts, kinds, settle_seconds)

    partial = []
    for name, expected in sorted(view_counts.items()):
        if name in ready:
            continue
        counts, _ = cached_views(cache_dir, name)
        if counts:
            partial.append((name, counts, expected))

    print(f"cache      {cache_dir}")
    print(f"kinds      {', '.join(kinds)}")
    print()
    print(f"sequences  {len(view_counts)}")
    print(f"  detected {len(ready)}")
    print(f"  partial  {len(partial)}")
    print(f"  absent   {len(view_counts) - len(ready) - len(partial)}")

    for name, counts, expected in partial:
        held = ", ".join(f"{k} {counts.get(k, 0)}/{expected}" for k in kinds)
        print(f"    {name:<36} {held}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences_file", type=str)
    parser.add_argument("--cache-config", type=str, required=True)
    parser.add_argument("--expect-kinds", nargs="+", default=list(DEFAULT_KINDS))
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="Ignore sequences written to more recently than this",
    )
    args = parser.parse_args()
    main(
        args.sequences_file,
        args.cache_config,
        tuple(args.expect_kinds),
        args.settle_seconds,
    )

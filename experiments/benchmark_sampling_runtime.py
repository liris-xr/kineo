# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Time the pair and bundle adjustment sampling stages of each strategy."""

import argparse
import os
import pickle
import statistics
import time

import orjson
import torch
from tqdm import tqdm

from kineo.annotations import (
    CameraExtrinsicsAnnotations,
    CameraIntrinsicsAnnotations,
    GlobalTimeReferenceAnnotations,
    Keypoints2DAnnotations,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.stages.bundle_adjustment_fps_sampling import (
    BundleAdjustmentFPSSamplingRuntimeConfig,
    BundleAdjustmentFPSSamplingStage,
)
from kineo.pipeline.stages.bundle_adjustment_sampling import (
    BundleAdjustmentSamplingRuntimeConfig,
    BundleAdjustmentSamplingStage,
)
from kineo.pipeline.stages.keypoints_pairs_fps_sampling import (
    KeypointsPairsFPSSamplingRuntimeConfig,
    KeypointsPairsFPSSamplingStage,
)
from kineo.pipeline.stages.keypoints_pairs_sampling import (
    KeypointsPairsSamplingRuntimeConfig,
    KeypointsPairsSamplingStage,
)

# EgoHumans benchmark budgets and threshold
N_PAIRS, N_KEYPOINTS, THRESHOLD = 3000, 500, 0.6
ANNOTATION_FILES = {
    "keypoints_2d": (Keypoints2DAnnotations, ("keypoints_2d",)),
    "global_time_reference": (GlobalTimeReferenceAnnotations, ("global_time_reference",)),
    "cameras_intrinsics": (CameraIntrinsicsAnnotations, ("cameras_intrinsics", "camera_intrinsics")),
    "cameras_extrinsics": (CameraExtrinsicsAnnotations, ("cameras_extrinsics", "camera_extrinsics")),
}


def build_strategies(n_pairs: int, n_keypoints: int, threshold: float) -> dict[str, tuple]:
    def pairs(strategy):
        return KeypointsPairsSamplingStage("Keypoints Pairs Sampling", 40, KeypointsPairsSamplingRuntimeConfig(
            max_points_pairs=n_pairs, pair_avg_conf_score_thr=threshold, strategy=strategy))

    def ba(strategy):
        return BundleAdjustmentSamplingStage("Bundle Adjustment Sampling", 55, BundleAdjustmentSamplingRuntimeConfig(
            n_kp_samples=n_keypoints, min_kp_score=threshold, strategy=strategy))

    return {
        "Random": (pairs("random"), ba("random")),
        "FPS": (
            KeypointsPairsFPSSamplingStage("Keypoints Pairs Sampling", 40, KeypointsPairsFPSSamplingRuntimeConfig(
                max_points_pairs=n_pairs, pair_avg_conf_score_thr=threshold)),
            BundleAdjustmentFPSSamplingStage("Bundle Adjustment Sampling", 55, BundleAdjustmentFPSSamplingRuntimeConfig(
                n_kp_samples=n_keypoints, min_kp_score=threshold)),
        ),
        "Top-P": (pairs("top_confidence"), ba("top_confidence")),
    }


def load_sequence(sequence_dir: str) -> dict:
    annotations = {}
    for key, (kind, names) in ANNOTATION_FILES.items():
        path = next(os.path.join(sequence_dir, f"{n}.pkl") for n in names
                    if os.path.exists(os.path.join(sequence_dir, f"{n}.pkl")))
        with open(path, "rb") as f:
            annotations[key] = kind.from_dict(pickle.load(f))
    return annotations


def time_stage(stage, pipeline: Pipeline, sequence: str, views: list, annotations: dict) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    stage.forward(sequence, pipeline, views, annotations, {}, stage.runtime_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main(run_dir: str, output_json: str, repeats: int, sequences: list[str] | None,
         n_pairs: int, n_keypoints: int, threshold: float) -> None:
    root = os.path.join(run_dir, "annotations")
    sequences = sorted(sequences or os.listdir(root))
    strategies = build_strategies(n_pairs, n_keypoints, threshold)
    pipeline = Pipeline(stages=[])
    names = list(strategies)
    results = {name: {} for name in names}

    for i, sequence in enumerate(tqdm(sequences, desc="Sequences")):
        base = load_sequence(os.path.join(root, sequence))
        views = [{"view_id": v} for v in base["cameras_intrinsics"].views_ids]
        # Extra warm-up round on the first sequence
        rounds = repeats + (1 if i == 0 else 0)
        times = {name: {"pairs": [], "ba": []} for name in names}
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for name in order:
                pairs_stage, ba_stage = strategies[name]
                annotations = dict(base)
                t_pairs = time_stage(pairs_stage, pipeline, sequence, views, annotations)
                t_ba = time_stage(ba_stage, pipeline, sequence, views, annotations)
                if i == 0 and r == 0:
                    continue
                times[name]["pairs"].append(t_pairs)
                times[name]["ba"].append(t_ba)
        for name in names:
            results[name][sequence] = {k: statistics.median(v) for k, v in times[name].items()}

    summary = {
        name: {
            "pairs_s": statistics.median(r["pairs"] for r in per_seq.values()),
            "ba_s": statistics.median(r["ba"] for r in per_seq.values()),
            "total_s": statistics.median(r["pairs"] + r["ba"] for r in per_seq.values()),
        }
        for name, per_seq in results.items()
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "wb") as f:
        f.write(orjson.dumps({"run_dir": run_dir, "repeats": repeats, "summary": summary,
                              "per_sequence": results}, option=orjson.OPT_INDENT_2))
    print(f"{len(sequences)} sequences, median over sequences of the median over {repeats} repeats:")
    for name, s in summary.items():
        print(f"  {name:7s} pairs {s['pairs_s']:.2f} s  BA {s['ba_s']:.2f} s  total {s['total_s']:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Benchmark run whose exported annotations are the inputs")
    parser.add_argument("output_json", help="Where to write the per-sequence and median timings")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sequences", nargs="+", help="Subset of sequences (default: all)")
    parser.add_argument("--pairs", type=int, default=N_PAIRS, help="Correspondence budget P")
    parser.add_argument("--keypoints", type=int, default=N_KEYPOINTS,
                        help="Bundle adjustment keypoint budget M")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Confidence thresholds tau = kappa")
    args = parser.parse_args()
    main(args.run_dir, args.output_json, args.repeats, args.sequences,
         args.pairs, args.keypoints, args.threshold)

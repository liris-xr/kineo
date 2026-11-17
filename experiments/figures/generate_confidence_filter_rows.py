# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the Human3.6M confidence filter table."""

import argparse
import json
import math
import multiprocessing
import os
import pickle

import cv2
import numpy as np
import torch
from kornia.geometry.epipolar import find_essential
from omegaconf import OmegaConf
from scipy.stats import binomtest

from experiments.figures.generate_online_rows import diverged_sequences, format_metric

_TEMPLATE = os.path.join(os.path.dirname(__file__), "confidence_filter_rows_template.tex")
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "experiments")

INPUTS = (("Original", ""), ("Occluded", "_occluded"))
# Confidence arm and its default run name
ARMS = (("off", "random_unfiltered"), ("on", "random"))
METRICS = (
    ("TE", "cam_stats"),
    ("s-TE", "cam_stats"),
    ("AE", "cam_stats"),
    ("vfov_error", "cam_stats"),
    ("w-mpjpe", "human_stats"),
    ("ga-mpjpe", "human_stats"),
    ("pa-mpjpe", "human_stats"),
)
MAX_ITERS = 1000
BOOTSTRAP_RESAMPLES = 10000


def run_config(run: str, input_suffix: str) -> str:
    # On the original images, random sampling is the benchmark
    if run == "random" and not input_suffix:
        return os.path.join(_CONFIGS_DIR, "benchmarks", "h36m_benchmark_rtmpose_estRt_estK_estD.yaml")
    return os.path.join(_CONFIGS_DIR, "ablation_study", "sampling",
                        f"h36m_sampling_rtmpose_{run}{input_suffix}.yaml")


def run_settings(config_path: str, cache_root: str | None = None) -> dict:
    if cache_root is not None:
        os.environ["CACHE_ROOT_DIR"] = cache_root
    cfg = OmegaConf.load(config_path)
    stages = cfg.pipeline.stages
    sfm = stages.sfm_camera_extrinsics_initialization.runtime_cfg
    return {
        "pair_threshold": float(stages.keypoints_pairs_sampling.runtime_cfg.pair_avg_conf_score_thr),
        "kp_threshold": float(stages.bundle_adjustment_sampling.runtime_cfg.min_kp_score),
        # Resolving the whole config would need PASCAL_VOC_DIR
        "cache_dir": str(cfg.cache_root_dir),
        "threshold_px": float(sfm.ransac_reproj_threshold),
        "confidence": float(sfm.ransac_confidence),
    }


def shared_sequences(run_dirs: list[str]) -> set[str]:
    scored = []
    for run in run_dirs:
        metrics_dir = os.path.join(run, "metrics")
        scored.append({f[:-5]: os.path.join(metrics_dir, f)
                       for f in os.listdir(metrics_dir) if f.endswith(".json")})
    common = set.intersection(*(set(files) for files in scored))
    for files in scored:
        common -= set(diverged_sequences({s: files[s] for s in common}))
    return common


def removed_fractions(scores: np.ndarray, pair_threshold: float,
                      kp_threshold: float) -> tuple[float, float]:
    n_views = scores.shape[0]
    kept_pairs = [np.sqrt(scores[i] * scores[j]) > pair_threshold
                  for i in range(n_views) for j in range(i + 1, n_views)]
    kept_keypoints = (scores > kp_threshold).sum(axis=0) >= 2
    return 1.0 - float(np.mean(kept_pairs)), 1.0 - float(np.mean(kept_keypoints))


def load_annotations(path: str) -> list:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["annotations"] if isinstance(obj, dict) else obj.annotations


def load_scores(annotations_path: str) -> np.ndarray:
    annotations = load_annotations(annotations_path)
    views = sorted({a["view_id"] for a in annotations})
    n_frames = max(a["frame_idx"] for a in annotations) + 1
    scores = np.zeros((len(views), n_frames, len(annotations[0]["scores"])))
    for a in annotations:
        scores[views.index(a["view_id"]), a["frame_idx"]] = a["scores"]
    return scores


def update_num_iters(confidence: float, outlier_ratio: float, n_model_points: int,
                     max_iters: int) -> int:
    # Port of OpenCV's RANSACUpdateNumIters
    num = math.log(max(1.0 - confidence, 1e-300))
    denom = 1.0 - (1.0 - outlier_ratio) ** n_model_points
    if denom < 1e-300:
        return 0
    denom = math.log(denom)
    if denom >= 0 or -num >= max_iters * (-denom):
        return max_iters
    return int(round(num / denom))


def squared_sampson(E: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    Ex1 = torch.einsum("mij,nj->mni", E, x1)
    Etx2 = torch.einsum("mji,nj->mni", E, x2)
    x2tEx1 = (x2.unsqueeze(0) * Ex1).sum(-1)
    return x2tEx1**2 / (Ex1[..., 0] ** 2 + Ex1[..., 1] ** 2 + Etx2[..., 0] ** 2 + Etx2[..., 1] ** 2)


def count_ransac_iterations(x1: torch.Tensor, x2: torch.Tensor, threshold: float,
                            confidence: float, rng: np.random.Generator,
                            max_iters: int = MAX_ITERS, batch: int = 256) -> int:
    # Reproduces OpenCV's RANSAC loop (calib3d ptsetreg.cpp)
    n = len(x1)
    h1 = torch.cat([x1, torch.ones(n, 1, dtype=x1.dtype)], 1)
    h2 = torch.cat([x2, torch.ones(n, 1, dtype=x2.dtype)], 1)
    n_iters, it, best = max_iters, 0, 0
    while it < n_iters:
        idx = torch.as_tensor(np.stack([rng.choice(n, 5, replace=False) for _ in range(batch)]))
        for E in find_essential(x1[idx], x2[idx]):
            if it >= n_iters:
                break
            it += 1
            E = E[torch.isfinite(E).all(dim=(1, 2)) & (E.abs().sum(dim=(1, 2)) > 0)]
            if len(E) == 0:
                continue
            good = int((squared_sampson(E, h1, h2) <= threshold**2).sum(-1).max())
            if good > max(best, 4):
                best = good
                n_iters = update_num_iters(confidence, (n - good) / n, 5, n_iters)
    return it


def essential_from_cameras(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray,
                           t2: np.ndarray) -> np.ndarray:
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R


def sequence_level_change(per_pair: list[np.ndarray], rng: np.random.Generator,
                          n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    pooled = np.concatenate(per_pair)
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(len(per_pair), size=len(per_pair))
        boot[b] = np.median(np.concatenate([per_pair[i] for i in pick]))
    seq_means = np.array([p.mean() for p in per_pair])
    n_improved = int((seq_means > 0).sum())
    n_changed = int((seq_means != 0).sum())
    return {
        "median": float(np.median(pooled)),
        "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "n_improved": n_improved,
        "n_sequences": len(per_pair),
        "sign_p": float(binomtest(n_improved, n_changed, 0.5).pvalue) if n_changed else 1.0,
    }


def pair_measures(points1: np.ndarray, points2: np.ndarray, K1_init: np.ndarray,
                  K2_init: np.ndarray, gt1: dict, gt2: dict, threshold_px: float,
                  confidence: float, n_seeds: int, rng: np.random.Generator) -> tuple:
    # Same normalization as OpenCV's two-camera findEssentialMat
    n1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), K1_init, None).reshape(-1, 2)
    n2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), K2_init, None).reshape(-1, 2)
    K_mean = (K1_init + K2_init) / 2
    threshold = threshold_px / ((K_mean[0, 0] + K_mean[1, 1]) / 2)
    _, mask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, confidence, threshold)
    ransac_ratio = float((mask.ravel() > 0).mean())

    t1, t2 = torch.as_tensor(n1), torch.as_tensor(n2)
    iterations = float(np.mean([
        count_ransac_iterations(t1, t2, threshold, confidence, rng) for _ in range(n_seeds)
    ]))

    g1, g2 = (undistort_normalized(p, gt) for p, gt in ((points1, gt1), (points2, gt2)))
    F = np.linalg.inv(gt2["K"]).T @ essential_from_cameras(
        gt1["R"], gt1["t"], gt2["R"], gt2["t"]) @ np.linalg.inv(gt1["K"])
    p1 = np.c_[g1, np.ones(len(g1))] @ gt1["K"].T
    p2 = np.c_[g2, np.ones(len(g2))] @ gt2["K"].T
    d2 = squared_sampson(torch.as_tensor(F)[None], torch.as_tensor(p1), torch.as_tensor(p2))[0]
    gt_share = float((d2 <= threshold_px**2).double().mean())
    return ransac_ratio, gt_share, iterations


def undistort_normalized(points: np.ndarray, camera: dict) -> np.ndarray:
    points = points.reshape(-1, 1, 2)
    if camera["model"] == "opencv_fisheye":
        return cv2.fisheye.undistortPoints(points, camera["K"], camera["D"]).reshape(-1, 2)
    return cv2.undistortPoints(points, camera["K"], camera["D"]).reshape(-1, 2)


def load_gt_cameras(dataset_dir: str, sequence: dict) -> dict:
    paths = sequence["annotations"]
    with open(os.path.join(dataset_dir, paths["cameras_intrinsics"])) as f:
        intrinsics = {a["view_id"]: a for a in json.load(f)["annotations"]}
    with open(os.path.join(dataset_dir, paths["cameras_extrinsics"])) as f:
        extrinsics = {a["view_id"]: a for a in json.load(f)["annotations"]}
    return {
        view: {
            "K": np.asarray(intrinsics[view]["K"], dtype=np.float64),
            "D": np.asarray(intrinsics[view]["distortion_coefficients"], dtype=np.float64),
            "model": intrinsics[view].get("distortion_model", "brown_conrady"),
            "R": np.asarray(extrinsics[view]["R"], dtype=np.float64),
            "t": np.asarray(extrinsics[view]["t"], dtype=np.float64),
        }
        for view in intrinsics
    }


def sequence_measures(task: tuple) -> dict:
    benchmark_dir, seq, gt, runs, n_seeds = task
    # Per-sequence seed keeps results independent of workers
    rng = np.random.default_rng(list(seq.encode("utf-8")))
    out = {}
    for arm, (name, s) in runs.items():
        calib = load_annotations(os.path.join(
            benchmark_dir, name, "annotations", seq, "calibration_points.pkl"))
        per_pair = {}
        for a in calib:
            vi, vj = a["view_i_id"], a["view_j_id"]
            if (vj, vi) in per_pair:
                continue
            K_init = {
                v: np.asarray(load_annotations(os.path.join(
                    s["cache_dir"], seq, "cameras_intrinsics", f"{v}.pkl"))[0]["K"],
                    dtype=np.float64)
                for v in (vi, vj)
            }
            per_pair[(vi, vj)] = pair_measures(
                np.asarray(a["points_i"], dtype=np.float64),
                np.asarray(a["points_j"], dtype=np.float64),
                K_init[vi], K_init[vj], gt[vi], gt[vj],
                s["threshold_px"], s["confidence"], n_seeds, rng)
        out[arm] = per_pair
    return out


def _init_worker():
    torch.set_num_threads(1)


def metric_cells(off: tuple[float, float], on: tuple[float, float],
                 name: str) -> tuple[list[str], list[str]]:
    parts = ([], [])
    for stat in range(2):
        printed = [format_metric(name, run[stat]) for run in (off, on)]
        values = [float(text) for text in printed]
        for run, text in enumerate(printed):
            bold = values[run] < values[1 - run]
            parts[run].append(rf"\textbf{{{text}}}" if bold else text)
    return parts


def ransac_cells(off: float, on: float, fmt: str,
                 higher_is_better: bool) -> tuple[str, str]:
    printed = [fmt.format(off), fmt.format(on)]
    values = [float(p) for p in printed]
    better = [values[0] > values[1], values[1] > values[0]]
    if not higher_is_better:
        better = [values[0] < values[1], values[1] < values[0]]
    return tuple(rf"\textbf{{{p}}}" if b else p for p, b in zip(printed, better))


def format_change(median: float, ci: tuple[float, float], fmt: str) -> str:
    return f"{fmt.format(median)} [{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def confidence_ablation_block(benchmark_dir: str, dataset_dir: str, sequences_file: str,
                              inputs: list[str], names: dict, settings: dict, seeds: int,
                              workers: int, boot_rng: np.random.Generator,
                              restrict_to: list[str] = ()) -> dict:
    if len({(s["threshold_px"], s["confidence"]) for s in settings.values()}) != 1:
        raise ValueError("The runs disagree on RANSAC settings")
    thresholds = {(s["pair_threshold"], s["kp_threshold"])
                  for (_, arm), s in settings.items() if arm == "on"}
    if len(thresholds) != 1:
        raise ValueError(f"The runs with confidence disagree on thresholds: {thresholds}")
    pair_threshold, kp_threshold = thresholds.pop()

    common = shared_sequences([os.path.join(benchmark_dir, name) for name in names.values()])
    if restrict_to:
        common &= shared_sequences(list(restrict_to))
    common = sorted(common)
    if not common:
        raise FileNotFoundError(f"No sequence was scored by all runs in {benchmark_dir}")

    with open(os.path.join(dataset_dir, sequences_file)) as f:
        sequences = {s["sequence_name"]: s for s in json.load(f)}
    tasks = [
        (benchmark_dir, seq, load_gt_cameras(dataset_dir, sequences[seq]),
         {arm: (names[(inp, arm)], settings[(inp, arm)]) for arm, _ in ARMS}, seeds)
        for inp in inputs for seq in common
    ]
    if workers > 1:
        with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(sequence_measures, tasks)
    else:
        _init_worker()
        results = [sequence_measures(t) for t in tasks]

    rows, comments = {}, []
    for i_input, inp in enumerate(inputs):
        ransac = results[i_input * len(common):(i_input + 1) * len(common)]
        on_name = names[(inp, "on")]
        removed = np.mean([
            removed_fractions(load_scores(os.path.join(
                benchmark_dir, on_name, "annotations", seq, "keypoints_2d.pkl")),
                pair_threshold, kp_threshold)
            for seq in common], axis=0)

        flat = {arm: np.array([m for r in ransac for m in r[arm].values()]) for arm, _ in ARMS}
        changes = {"ransac": [], "gt": [], "iterations": []}
        for r in ransac:
            pairs = sorted(set(r["off"]) & set(r["on"]))
            off = np.array([r["off"][p] for p in pairs])
            on = np.array([r["on"][p] for p in pairs])
            changes["ransac"].append(on[:, 0] - off[:, 0])
            changes["gt"].append(on[:, 1] - off[:, 1])
            changes["iterations"].append(np.log(off[:, 2] / on[:, 2]))
        stats = {k: sequence_level_change(v, boot_rng) for k, v in changes.items()}

        ransac_pairs = [
            ransac_cells(100 * np.median(flat["off"][:, 0]),
                         100 * np.median(flat["on"][:, 0]), "{:.1f}", True),
            ransac_cells(100 * np.median(flat["off"][:, 1]),
                         100 * np.median(flat["on"][:, 1]), "{:.1f}", True),
            ransac_cells(np.median(flat["off"][:, 2]),
                         np.median(flat["on"][:, 2]), "{:.0f}", False),
        ]
        metric_pairs = []
        for name, section in METRICS:
            stats_per_arm, values_per_arm = [], []
            for arm, _ in ARMS:
                metrics_dir = os.path.join(benchmark_dir, names[(inp, arm)], "metrics")
                values = []
                for seq in common:
                    with open(os.path.join(metrics_dir, f"{seq}.json")) as f:
                        values.append(json.load(f)[section][name]["mean"])
                values_per_arm.append(np.array(values))
                stats_per_arm.append((float(np.median(values)), float(np.percentile(values, 90))))
            metric_pairs.append(metric_cells(*stats_per_arm, name))
            change = values_per_arm[1] - values_per_arm[0]
            n_lower, n_changed = int((change < 0).sum()), int((change != 0).sum())
            sign_p = binomtest(n_lower, n_changed).pvalue if n_changed else 1.0
            comments.append(f"% {inp}, {name}: median per-sequence change {np.median(change):+.4f}, "
                            f"lower with confidence in {n_lower}/{len(change)} sequences "
                            f"(sign test p={sign_p:.2g})")

        rows[inp] = []
        for i_arm, (mark, removed_cells) in enumerate((
            ("--", ("--", "--")),
            (r"\checkmark", tuple(f"{100 * v:.1f}" for v in removed)),
        )):
            rows[inp].append([mark, *removed_cells] + [c[i_arm] for c in ransac_pairs] + [
                cell for c in metric_pairs for cell in c[i_arm]])

        iterations = stats["iterations"]
        lines = {
            "inliers (pts)": format_change(
                100 * stats["ransac"]["median"],
                tuple(100 * c for c in stats["ransac"]["ci"]), "{:+.1f}"),
            "GT-consistent (pts)": format_change(
                100 * stats["gt"]["median"], tuple(100 * c for c in stats["gt"]["ci"]), "{:+.1f}"),
            "iterations (off / on)": format_change(
                math.exp(iterations["median"]),
                tuple(math.exp(c) for c in iterations["ci"]), "{:.2f}"),
        }
        for (label, text), stat in zip(lines.items(), stats.values()):
            comments.append(f"% {inp}, {label}: {text}, improved in "
                            f"{stat['n_improved']}/{stat['n_sequences']} sequences "
                            f"(sign test p={stat['sign_p']:.1g})")

    return {"rows": rows, "comments": comments, "n_sequences": len(common),
            "threshold": pair_threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=str, help="Directory holding the four runs")
    parser.add_argument("dataset_dir", type=str, help="Human3.6M or EgoHumans dataset directory")
    parser.add_argument("--seeds", type=int, default=5,
                        help="RANSAC runs averaged per pair for the iteration count")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processes scoring sequences in parallel")
    parser.add_argument("--runs", nargs="+", metavar="RUN_DIR",
                        help="Original off, original on[, occluded off, occluded on] "
                             "(default: the RTMPose sampling runs)")
    parser.add_argument("--configs", nargs="+", metavar="CONFIG",
                        help="Config file of each run given with --runs")
    parser.add_argument("--cache-roots", nargs="+", metavar="CACHE_ROOT_DIR",
                        help="CACHE_ROOT_DIR of each run given with --runs")
    parser.add_argument("--restrict-to", nargs="+", default=[], metavar="RUN_DIR")
    parser.add_argument("--sequences-file", default="h36m_protocol1_sequences.json",
                        help="Sequence list in the dataset directory, e.g. "
                             "egohumans_sequences.json")
    args = parser.parse_args()

    if args.runs:
        if len(args.runs) not in (2, 4):
            parser.error("--runs takes 2 (original only) or 4 run directories")
        if not (args.configs and args.cache_roots) or not (
                len(args.runs) == len(args.configs) == len(args.cache_roots)):
            parser.error("--runs needs as many --configs and --cache-roots")
        INPUTS = INPUTS[:len(args.runs) // 2]
    keys = [(inp, arm) for inp, _ in INPUTS for arm, _ in ARMS]
    if args.runs:
        names = dict(zip(keys, args.runs))
        settings = {key: run_settings(config, cache_root) for key, config, cache_root
                    in zip(keys, args.configs, args.cache_roots)}
    else:
        configs = {(inp, arm): run_config(run, sfx) for inp, sfx in INPUTS for arm, run in ARMS}
        names = {key: os.path.splitext(os.path.basename(path))[0] for key, path in configs.items()}
        settings = {key: run_settings(path) for key, path in configs.items()}
    block = confidence_ablation_block(
        args.benchmark_dir, args.dataset_dir, args.sequences_file, [inp for inp, _ in INPUTS],
        names, settings, args.seeds, args.workers, np.random.default_rng(0), args.restrict_to)
    print("\n".join(block["comments"]))
    rows = []
    for inp in block["rows"]:
        if rows:
            rows.append(r"        \midrule")
        for i_arm, cells in enumerate(block["rows"][inp]):
            lead = rf"\multirow{{2}}{{*}}{{{inp}}}" if i_arm == 0 else ""
            rows.append("        " + " & ".join([lead, *cells]) + r" \\")

    with open(_TEMPLATE) as f:
        table = f.read().replace("{rows}", "\n".join(rows))
    table = table.replace("{n_sequences}", str(block["n_sequences"]))
    table = table.replace("{threshold}", f"{block['threshold']:g}")
    print(table)

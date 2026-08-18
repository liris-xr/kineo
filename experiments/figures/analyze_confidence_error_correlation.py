"""Within-joint Spearman correlation between 3D confidence score and per-joint
reconstruction error (W-MPJPE / PA-MPJPE).

Quick check: does the pipeline's per-joint confidence track per-joint error?
Expect NEGATIVE rho (high confidence -> low error).

Method:
  - Per joint, pool paired (confidence, error) samples over all frames and all
    sequences, compute Spearman rho.
  - Report distribution across joints (median + IQR). Avoids Simpson-paradox
    from pooling joints with different confidence/error scales.
  - Also reports a naive pooled rho over all (joint, frame) pairs for reference.

Usage:
  python -m experiments.figures.analyze_confidence_error_correlation \
      <dataset_dir> <predictions_dir> [--metric w-mpjpe|pa-mpjpe] \
      [--sequences-file NAME.json] [--drop-zero-conf] [--min-samples N]
"""

import argparse
import os

import numpy as np
import orjson
from scipy.stats import spearmanr

from kineo.eval.human_metrics import compute_human_metrics
from experiments.figures.egohumans_stats_utils import (
    load_gt_annotations,
    load_predicted_annotations,
)


def build_conf_lookup(pred_kp, gt_kp):
    """(frame_idx, subject_id, joint_name) -> confidence score, in GT joint order."""
    gt_format = gt_kp.metadata.formats[0]
    pred_format = pred_kp.metadata.formats[0]
    # Mirror compute_human_metrics: convert pred to GT format so joint indices align.
    if gt_format.name != pred_format.name:
        pred_kp = pred_kp.convert_to_format(gt_format)
    joint_names = gt_format.keypoints_names

    # Single-subject id mismatch remap (mirrors compute_human_metrics).
    gt_subjects = set(gt_kp.subjects_ids)
    pred_subjects = set(pred_kp.subjects_ids)
    remap = {}
    if gt_subjects != pred_subjects and len(gt_subjects) == len(pred_subjects) == 1:
        remap = {next(iter(pred_subjects)): next(iter(gt_subjects))}

    conf = {}
    for ann in pred_kp.annotations:
        sid = remap.get(ann.subject_id, ann.subject_id)
        scores = ann.scores.cpu().numpy()
        for kp_idx, jname in enumerate(joint_names):
            conf[(ann.frame_idx, sid, jname)] = float(scores[kp_idx])
    return conf


def collect_pairs(dataset_dir, predictions_dir, metric, sequences_file, drop_zero_conf):
    """joint_name -> (conf_array, err_array) pooled over all frames + sequences."""
    with open(os.path.join(dataset_dir, sequences_file), "rb") as f:
        sequences = orjson.loads(f.read())

    gt_kp_seq, _, gt_ext_seq = load_gt_annotations(dataset_dir, sequences)
    pred_kp_seq, _, pred_ext_seq, _ = load_predicted_annotations(
        predictions_dir, sequences
    )

    per_joint = {}  # joint_name -> [conf...], [err...]
    common = [s for s in gt_kp_seq if s in pred_kp_seq]
    print(f"Sequences with predictions: {len(common)}/{len(gt_kp_seq)}")

    for seq in common:
        gt_kp = gt_kp_seq[seq]
        pred_kp = pred_kp_seq[seq]
        err = compute_human_metrics(
            gt_kp, gt_ext_seq[seq], pred_kp, pred_ext_seq[seq]
        )
        conf = build_conf_lookup(pred_kp, gt_kp)

        for frame_idx, subjects in err.items():
            for subj in subjects:
                sid = subj["subject_id"]
                for j in subj["joints"]:
                    jname = j["joint_name"]
                    e = j[metric]
                    c = conf.get((frame_idx, sid, jname))
                    if c is None:
                        continue
                    if not np.isfinite(e) or not np.isfinite(c):
                        continue
                    if drop_zero_conf and c == 0.0:
                        continue
                    cl, el = per_joint.setdefault(jname, ([], []))
                    cl.append(c)
                    el.append(e)
    return per_joint


def compute_ause(conf, err, n_steps=100):
    """AUSE: does confidence rank errors like the oracle?

    Sparsification: repeatedly drop the lowest-confidence samples, track mean
    error of the remainder. Oracle: drop the highest-error samples. Both curves
    normalized by full-set mean error (start at 1.0). AUSE = area between them
    over fraction-removed in [0,1]. Lower = better. AURG = area between the
    random baseline (flat 1.0) and the sparsification curve; higher = better.
    Returns (ause, aurg).
    """
    conf = np.asarray(conf, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    n = len(err)
    base = err.mean()
    if n < 2 or base == 0:
        return np.nan, np.nan

    # Sparsification: keep samples with highest confidence -> drop lowest first.
    err_by_conf_desc = err[np.argsort(-conf)]  # index 0 = highest conf (kept longest)
    # Oracle: drop highest error first -> keep lowest errors.
    err_asc = np.sort(err)

    # Suffix means of remaining set as we remove k = t*n samples from the "drop" end.
    # spars: remaining = err_by_conf_desc[:n-k]; oracle: remaining = err_asc[:n-k].
    csum_conf = np.cumsum(err_by_conf_desc)
    csum_orac = np.cumsum(err_asc)

    ts = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    ks = np.floor(ts * n).astype(int)
    keep = n - ks  # >= 1
    spars = (csum_conf[keep - 1] / keep) / base
    orac = (csum_orac[keep - 1] / keep) / base

    ause = np.trapz(spars - orac, ts) if hasattr(np, "trapz") else np.trapezoid(spars - orac, ts)
    aurg = np.trapz(1.0 - spars, ts) if hasattr(np, "trapz") else np.trapezoid(1.0 - spars, ts)
    return float(ause), float(aurg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir")
    ap.add_argument("predictions_dir")
    ap.add_argument("--metric", choices=["w-mpjpe", "pa-mpjpe"], default="w-mpjpe")
    ap.add_argument("--sequences-file", default="egohumans_sequences.json")
    ap.add_argument(
        "--drop-zero-conf",
        action="store_true",
        help="Drop samples with confidence==0 (invalid triangulation).",
    )
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args()

    per_joint = collect_pairs(
        args.dataset_dir,
        args.predictions_dir,
        args.metric,
        args.sequences_file,
        args.drop_zero_conf,
    )

    rows = []
    all_conf, all_err = [], []
    for jname, (cl, el) in sorted(per_joint.items()):
        c = np.asarray(cl)
        e = np.asarray(el)
        all_conf.append(c)
        all_err.append(e)
        if len(c) < args.min_samples or np.std(c) == 0 or np.std(e) == 0:
            rows.append((jname, len(c), np.nan, np.nan, np.nan, np.nan))
            continue
        rho, p = spearmanr(c, e)
        ause, aurg = compute_ause(c, e)
        rows.append((jname, len(c), rho, p, ause, aurg))

    print(f"\n=== Within-joint: confidence vs {args.metric} ===")
    print("(rho NEGATIVE = high conf -> low error; AUSE low good; AURG high good)\n")
    print(f"{'joint':<24}{'n':>8}{'rho':>10}{'p':>12}{'AUSE':>10}{'AURG':>10}")
    for jname, n, rho, p, ause, aurg in rows:
        rho_s = f"{rho:>10.3f}" if np.isfinite(rho) else f"{'--':>10}"
        p_s = f"{p:>12.2e}" if np.isfinite(p) else f"{'--':>12}"
        ause_s = f"{ause:>10.3f}" if np.isfinite(ause) else f"{'--':>10}"
        aurg_s = f"{aurg:>10.3f}" if np.isfinite(aurg) else f"{'--':>10}"
        print(f"{jname:<24}{n:>8}{rho_s}{p_s}{ause_s}{aurg_s}")

    auses = np.array([a for *_, a, _ in rows if np.isfinite(a)])
    aurgs = np.array([g for *_, g in rows if np.isfinite(g)])
    rhos = np.array([r for _, _, r, _, _, _ in rows if np.isfinite(r)])
    print("\n--- Distribution across joints ---")
    if len(rhos):
        q25, q50, q75 = np.percentile(rhos, [25, 50, 75])
        print(f"joints used     : {len(rhos)}")
        print(f"median rho      : {q50:.3f}")
        print(f"IQR [q25, q75]  : [{q25:.3f}, {q75:.3f}]")
        print(f"min / max rho   : {rhos.min():.3f} / {rhos.max():.3f}")
        print(f"frac negative   : {np.mean(rhos < 0):.2f}")
        print(f"median AUSE      : {np.median(auses):.3f}   (0=perfect ranking)")
        print(f"median AURG      : {np.median(aurgs):.3f}   (>0=beats random)")
    else:
        print("no joints met min-samples / variance threshold")

    # Pooled AUSE over all samples (single ranking across joints).
    pc_all = np.concatenate(all_conf)
    pe_all = np.concatenate(all_err)
    pause, paurg = compute_ause(pc_all, pe_all)
    print(f"\n--- Pooled AUSE (all joints, single ranking) ---")
    print(f"AUSE = {pause:.3f}   AURG = {paurg:.3f}")

    # Naive pooled reference (subject to Simpson paradox).
    pc = np.concatenate(all_conf)
    pe = np.concatenate(all_err)
    if len(pc) > args.min_samples and np.std(pc) > 0 and np.std(pe) > 0:
        prho, pp = spearmanr(pc, pe)
        print(f"\n--- Naive pooled (all joints, reference only) ---")
        print(f"n = {len(pc)}   rho = {prho:.3f}   p = {pp:.2e}")


if __name__ == "__main__":
    main()

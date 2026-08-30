# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Names a config's per-view cache directory after what it actually caches.

The per-view cache is shared between configs, so two configs may point at the
same directory only when every caching stage in them produces the same thing
for a given view. This derives the directory name from exactly those inputs: a
readable stack token, then a hash over the model paths and the runtime knobs
that change what a stage outputs. Change any of them and the hash moves, which
is what keeps a stale cache from being read as a fresh one.

Run it over the configs to see the grouping:

    python scripts/cache_dir_name.py configs/experiments/benchmarks/*.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json

from omegaconf import DictConfig, OmegaConf

# Per caching stage: the constructor keys naming its weights, the runtime keys
# that change its output, and the token it contributes to the directory name.
CACHING_STAGES = {
    "moge_intrinsics_estimation.MoGeIntrinsicsEstimationStage": {
        "model_keys": ["model_name_or_path"],
        "runtime_keys": ["use_half_precision"],
        "token": "moge",
    },
    "nlf.skeleton_keypoints_detection.NLFSkeletonKeypointsDetectionStage": {
        "model_keys": ["torchscript_model_path"],
        "runtime_keys": ["skeleton_name", "use_half_precision", "frame_step"],
        "token": "nlf",
    },
    "nlf.smpl_keypoints_detection.NLFSMPLKeypointsDetectionStage": {
        "model_keys": ["torchscript_model_path"],
        "runtime_keys": ["model_name", "use_half_precision", "frame_step"],
        "token": "nlf",
    },
    "mmdet_bbox_detection.MMDetBboxDetectionStage": {
        "model_keys": ["det_model", "det_model_weights", "det_model_scope"],
        "runtime_keys": [
            "bbox_thr",
            "nms_iou_thr",
            "nms_pre_top_k",
            "best_bbox_only",
            "det_category_id",
            "use_half_precision",
            "frame_step",
            "default_subject_id",
        ],
        "token": "mmdet",
    },
    "mmpose_keypoints_detection.MMPoseKeypointsDetectionStage": {
        "model_keys": [
            "keypoints_model",
            "keypoints_model_weights",
            "keypoints_model_scope",
        ],
        "runtime_keys": [
            "use_half_precision",
            "force_zero_scores_outside_bbox",
            "frame_step",
            "disable_confidence",
            "use_flip_test",
        ],
        "token": "mmpose",
    },
    "mmlab_bbox_keypoints_detection.MMLabBboxKeypointsDetectionStage": {
        "model_keys": [
            "det_model",
            "det_model_weights",
            "det_model_scope",
            "keypoints_model",
            "keypoints_model_weights",
            "keypoints_model_scope",
        ],
        "runtime_keys": [
            "bbox_thr",
            "nms_iou_thr",
            "nms_pre_top_k",
            "best_bbox_only",
            "det_category_id",
            "use_half_precision",
            "frame_step",
            "default_subject_id",
            "force_zero_scores_outside_bbox",
            "use_flip_test",
        ],
        "token": "mmlab",
    },
    "rtmlib.rtmlib_bbox_detection.RtmlibBboxDetectionStage": {
        "model_keys": ["bbox_model", "bbox_model_input_shape_hw"],
        "runtime_keys": [
            "bbox_thr",
            "nms_iou_thr",
            "best_bbox_only",
            "det_category_id",
            "frame_step",
            "default_subject_id",
        ],
        "token": "rtmdet",
    },
    "rtmlib.rtmlib_keypoints_detection.RtmlibKeypointsDetectionStage": {
        "model_keys": ["keypoints_model", "keypoints_model_input_shape_hw"],
        "runtime_keys": ["frame_step"],
        "token": "rtmpose",
    },
    "rtmlib.rtmlib_bbox_keypoints_detection.RtmlibBboxKeypointsDetectionStage": {
        "model_keys": [
            "bbox_model",
            "bbox_model_input_shape_hw",
            "keypoints_model",
            "keypoints_model_input_shape_hw",
        ],
        "runtime_keys": [
            "bbox_thr",
            "nms_iou_thr",
            "best_bbox_only",
            "det_category_id",
            "frame_step",
            "default_subject_id",
        ],
        "token": "rtmlib",
    },
    "sam2_semiauto_bbox_detection.SAM2SemiAutoBboxDetectionStage": {
        "model_keys": [
            "sam2_model_cfg",
            "sam2_model_weights",
            "det_model",
            "det_model_weights",
            "det_model_scope",
        ],
        "runtime_keys": ["frame_step", "nms_pre_top_k"],
        "token": "sam2",
    },
    "sam2_semiauto_bbox_detection_rtmlib.SAM2SemiAutoBboxDetectionRtmlibStage": {
        "model_keys": [
            "sam2_model_cfg",
            "sam2_model_weights",
            "det_model",
            "det_model_input_shape_hw",
        ],
        "runtime_keys": ["frame_step", "bbox_thr", "nms_iou_thr"],
        "token": "sam2rtm",
    },
}


def cache_signature(cfg: DictConfig) -> list[dict]:
    """Collects what every caching stage of a config feeds into its output.

    Args:
        cfg: A loaded pipeline config.

    Returns:
        One entry per caching stage, sorted by stage token so the result does
        not depend on the order the stages happen to appear in the config.
    """
    stages = OmegaConf.select(cfg, "pipeline.stages") or {}
    signature = []

    for stage_name, stage in stages.items():
        target = str(stage.get("_target_", ""))
        spec = next(
            (spec for key, spec in CACHING_STAGES.items() if target.endswith(key)),
            None,
        )
        if spec is None:
            continue

        entry = {"stage": spec["token"]}
        for key in spec["model_keys"]:
            if key in stage:
                entry[key] = str(stage[key])
        for key in spec["runtime_keys"]:
            value = OmegaConf.select(
                cfg, f"pipeline.stages.{stage_name}.runtime_cfg.{key}"
            )
            if value is not None:
                entry[key] = str(value)
        signature.append(entry)

    return sorted(signature, key=lambda entry: entry["stage"])


def cache_dir_name(cfg: DictConfig) -> str | None:
    """Builds the cache directory name a config should use.

    Args:
        cfg: A loaded pipeline config.

    Returns:
        A name of the form `<stack tokens>-<precision>_<hash>`, or None when the
        config has no caching stage and so needs no cache directory.
    """
    signature = cache_signature(cfg)

    if not signature:
        return None

    tokens = []
    for entry in signature:
        token = entry["stage"]
        for key in ("skeleton_name", "model_name"):
            if key in entry:
                token += entry[key].replace("_", "")
        tokens.append(token)

    precisions = {
        entry["use_half_precision"]
        for entry in signature
        if "use_half_precision" in entry
    }
    if precisions == {"True"}:
        tokens.append("fp16")
    elif precisions == {"False"}:
        tokens.append("fp32")

    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    return "-".join(tokens) + "_" + digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    for path in args.configs:
        name = cache_dir_name(OmegaConf.load(path))
        print(f"{name or '<no cache>'}\t{path}")


if __name__ == "__main__":
    main()

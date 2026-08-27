# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Preprocessing of AIST++ on top of the raw AIST Dance Database videos.

AIST++ annotates the *refined* videos, which AIST produced by trimming the
pre-roll and post-roll away from each *raw* recording. Each camera was started
independently, so the same annotation lands at a different frame in each raw
video. Refined videos are a pure frame-range cut of the raw ones -- same
60000/1001 constant frame rate, no resampling -- so that per-camera shift is a
single integer, recovered here by matching refined frames back into the raw
video.

That shift is the dataset's reason to exist for temporal calibration: it is a
ground-truth inter-camera time offset, published as `cameras_temporal`
annotations and, unchanged, as a standalone offsets file that an experiment
estimating offsets can score itself against.
"""

import json
import os
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import orjson
import torch
from tqdm import tqdm

from kineo.annotations import KeypointsFormat
from kineo.annotations.bboxes_utils import generate_bboxes2d_from_kps2d
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import (
    CameraDistortionModel,
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_temporal import (
    CameraTemporalAnnotation,
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
)
from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotation,
    Keypoints2DAnnotations,
)
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.annotations.keypoints_utils import (
    generate_kps2d_from_kps3d_and_cameras,
)
from kineo.datasets import annotations_io
from kineo.datasets.aistpp.aistpp_download import AISTPP_SPLITS
from kineo.io.ffmpeg import decode_video_to_grayscale

# Every AIST video, raw and refined alike, is constant frame rate at 60000/1001.
VIDEO_FPS = 60000 / 1001

# AIST++ keypoints and camera translations are expressed in centimeters.
CENTIMETERS_TO_METERS = 0.01

KEYPOINTS_FORMAT = KeypointsFormat.from_mmpose_dataset("coco")

# `keypoints2d.zip` is deliberately absent: 2D keypoints are reprojected from
# the 3D ones so that they agree with the cameras, as the other datasets do.
ANNOTATION_ARCHIVES = ("cameras.zip", "keypoints3d.zip", "splits.zip")

# The offset is unambiguous long before the frames are legible.
MATCH_FRAME_WIDTH = 64
MATCH_FRAME_HEIGHT = 36


def _serializer_fallback(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, range):
        return {
            "type": "range",
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def estimate_frame_offset(
    raw_frames: torch.Tensor,
    refined_frames: torch.Tensor,
    n_probes: int = 6,
) -> tuple[int, float]:
    """Locates a refined video inside the raw recording it was cut from.

    Scores every candidate shift by the mean absolute difference between a few
    probe frames of the refined video and the raw frames they would land on,
    summed over probes. Probes are spread over the clip: a shift that is right
    for one of them but wrong for the others scores badly, which is what rules
    out a frame rate mismatch rather than assuming there is none.

    The first tenth of the clip is left out of the probes. Refined videos open
    on a freshly re-encoded keyframe whose quantization differs enough from the
    raw one to make it the least discriminative frame of the whole clip.

    Args:
        raw_frames: Grayscale raw frames, of shape (n_raw, n_pixels).
        refined_frames: Grayscale refined frames, of shape (n_refined,
            n_pixels), decoded at the same resolution as `raw_frames`.
        n_probes: Number of refined frames matched.

    Returns:
        The offset, in frames, such that `refined_frames[i]` is
        `raw_frames[offset + i]`, and the ratio between the best score outside
        the winner's neighbourhood and the winning score. That ratio is 1 when
        the match is indistinguishable from its runner-up and grows as the
        match gets more decisive.

    Raises:
        ValueError: If the refined video is longer than the raw one, or if the
            two were decoded at different resolutions.
    """
    if raw_frames.shape[1] != refined_frames.shape[1]:
        raise ValueError(
            f"Frames were decoded at different resolutions: "
            f"{raw_frames.shape[1]} vs {refined_frames.shape[1]} pixels."
        )

    n_raw, n_refined = raw_frames.shape[0], refined_frames.shape[0]

    if n_refined > n_raw:
        raise ValueError(
            f"Refined video is longer than the raw one: {n_refined} > {n_raw} frames."
        )

    max_offset = n_raw - n_refined

    probes = torch.linspace(
        n_refined // 10, n_refined - 1, n_probes, dtype=torch.long
    ).unique()

    raw = raw_frames.float()
    refined = refined_frames.float()

    scores = torch.zeros(max_offset + 1)
    for probe in probes:
        windows = raw[probe : probe + max_offset + 1]
        scores += (windows - refined[probe]).abs().mean(dim=1)

    offset = int(scores.argmin())

    # A near-miss shift shares most of its frames with the winner, so only
    # scores outside its neighbourhood say anything about ambiguity.
    neighbourhood = torch.arange(max_offset + 1)
    contenders = scores[(neighbourhood - offset).abs() > 2]
    margin = (
        float(contenders.min() / scores[offset])
        if contenders.numel() > 0
        else float("inf")
    )

    return offset, margin


def _extract_annotations(dataset_dir: str):
    """Unpacks the AIST++ annotation archives next to themselves."""
    annotations_dir = os.path.join(dataset_dir, "annotations")

    for archive_name in ANNOTATION_ARCHIVES:
        archive_path = os.path.join(annotations_dir, archive_name)

        if not os.path.exists(archive_path):
            raise FileNotFoundError(
                f"Missing annotation archive {archive_path}. Download the "
                "dataset first."
            )

        with zipfile.ZipFile(archive_path) as archive:
            members = [
                name for name in archive.namelist() if not name.startswith("__MACOSX/")
            ]
            archive.extractall(annotations_dir, members=members)


def _load_split_sequences(
    dataset_dir: str, split: str, drop_ignored: bool
) -> list[str]:
    annotations_dir = os.path.join(dataset_dir, "annotations")

    with open(os.path.join(annotations_dir, "splits", f"{split}.txt")) as f:
        sequences = f.read().split()

    if not drop_ignored:
        return sorted(sequences)

    with open(os.path.join(annotations_dir, "ignore_list.txt")) as f:
        ignored = set(f.read().split())

    return sorted(set(sequences) - ignored)


def _load_camera_settings(dataset_dir: str) -> tuple[dict, dict]:
    """Reads the sequence-to-setting mapping and every camera setting."""
    cameras_dir = os.path.join(dataset_dir, "annotations", "cameras")

    with open(os.path.join(cameras_dir, "mapping.txt")) as f:
        mapping = dict(line.split() for line in f.read().splitlines() if line)

    settings = {}
    for setting_name in set(mapping.values()):
        with open(os.path.join(cameras_dir, f"{setting_name}.json")) as f:
            settings[setting_name] = {camera["name"]: camera for camera in json.load(f)}

    return mapping, settings


def _find_sequence_videos(
    dataset_dir: str, sequence_name: str, camera_names: list[str]
) -> dict[str, str]:
    """Maps each camera with a downloaded raw video to that video's path.

    Args:
        dataset_dir: Absolute path to the dataset root.
        sequence_name: Sequence name carrying "cAll", as the annotations use.
        camera_names: Cameras the sequence's setting declares.

    Returns:
        Maps a camera name to its raw video path, relative to `dataset_dir`,
        for the cameras whose raw and refined videos are both on disk.
    """
    videos = {}

    for camera_name in camera_names:
        video_name = f"{sequence_name.replace('cAll', camera_name)}.mp4"
        raw_relpath = os.path.join("videos", "raw", video_name)
        refined_path = os.path.join(dataset_dir, "videos", "refined", video_name)

        if os.path.exists(os.path.join(dataset_dir, raw_relpath)) and os.path.exists(
            refined_path
        ):
            videos[camera_name] = Path(raw_relpath).as_posix()

    return videos


def _estimate_sequence_offsets(
    dataset_dir: str, videos: dict[str, str]
) -> dict[str, dict]:
    """Matches every camera's refined video back into its raw recording."""
    offsets = {}

    for camera_name, raw_relpath in videos.items():
        raw_frames = decode_video_to_grayscale(
            os.path.join(dataset_dir, raw_relpath),
            MATCH_FRAME_WIDTH,
            MATCH_FRAME_HEIGHT,
        )
        refined_frames = decode_video_to_grayscale(
            os.path.join(
                dataset_dir, raw_relpath.replace("videos/raw", "videos/refined")
            ),
            MATCH_FRAME_WIDTH,
            MATCH_FRAME_HEIGHT,
        )

        offset, margin = estimate_frame_offset(raw_frames, refined_frames)

        offsets[camera_name] = {
            "offset_frames": offset,
            "offset_seconds": offset / VIDEO_FPS,
            "match_margin": margin,
            "n_raw_frames": int(raw_frames.shape[0]),
            "n_refined_frames": int(refined_frames.shape[0]),
        }

    return offsets


def _build_camera_annotations(
    setting: dict, camera_names: list[str]
) -> tuple[CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations]:
    """Converts an AIST++ camera setting to intrinsics and extrinsics.

    AIST++ stores the pose as the OpenCV rotation vector and translation that
    take a world point to the camera frame, which is what the annotations want,
    so only the rotation vector is expanded and the translation rescaled.
    """
    intrinsics = []
    extrinsics = []

    for camera_name in camera_names:
        camera = setting[camera_name]
        width, height = camera["size"]

        # AIST++ stores OpenCV's (k1, k2, p1, p2, k3), Brown-Conrady's own
        # coefficients in the order the annotation expects.
        distortion_coefficients = torch.tensor(
            camera["distortions"], dtype=torch.float32
        )

        intrinsics.append(
            CameraIntrinsicsAnnotation(
                view_id=camera_name,
                frame_idx=0,
                K=torch.tensor(camera["matrix"], dtype=torch.float32),
                distortion_coefficients=distortion_coefficients,
                distortion_model=CameraDistortionModel.BROWN_CONRADY,
                resolution_hw=(height, width),
            )
        )

        R, _ = cv2.Rodrigues(np.asarray(camera["rotation"], dtype=np.float64))
        t = np.asarray(camera["translation"], dtype=np.float64)

        extrinsics.append(
            CameraExtrinsicsAnnotation(
                view_id=camera_name,
                frame_idx=0,
                R=torch.from_numpy(R).to(torch.float32),
                t=torch.from_numpy(t * CENTIMETERS_TO_METERS).to(torch.float32),
            )
        )

    return (
        CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=intrinsics,
        ),
        CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=extrinsics,
        ),
    )


def _shift_kps2d_to_local_frames(
    kps2d: Keypoints2DAnnotations, offsets: dict[str, dict]
) -> Keypoints2DAnnotations:
    """Re-indexes projected 2D keypoints onto each view's own raw frames.

    The projection carries the 3D annotation's frame index, which counts frames
    of the refined timeline shared by every view. Views are read from their raw
    videos, where that same instant sits `offset_frames` later.
    """
    return Keypoints2DAnnotations(
        metadata=kps2d.metadata,
        annotations=[
            Keypoints2DAnnotation(
                view_id=annotation.view_id,
                frame_idx=annotation.frame_idx
                + offsets[annotation.view_id]["offset_frames"],
                subject_id=annotation.subject_id,
                xy=annotation.xy,
                scores=annotation.scores,
                format=annotation.format,
            )
            for annotation in kps2d.annotations
        ],
    )


def _build_camera_temporal_annotations(
    offsets: dict[str, dict], reference_camera: str
) -> CameraTemporalAnnotations:
    """Turns per-camera cut points into inter-camera time offsets.

    A view's local timestamps plus its time offset must land on one shared
    timeline, so the offsets are the cut points negated and rebased on a
    reference view, which is left at zero.
    """
    reference_offset = offsets[reference_camera]["offset_frames"]

    return CameraTemporalAnnotations(
        metadata=CameraTemporalAnnotationsMetadata(),
        annotations=[
            CameraTemporalAnnotation(
                view_id=camera_name,
                frame_idx=0,
                time_offset=(reference_offset - offset["offset_frames"]) / VIDEO_FPS,
            )
            for camera_name, offset in offsets.items()
        ],
    )


def preprocess_aistpp(
    dataset_dir: str,
    split: str = "pose_test",
    drop_ignored: bool = True,
    min_match_margin: float = 1.5,
    skip_extract: bool = False,
):
    """Preprocesses AIST++ over the raw AIST videos for use in Kineo.

    Writes, per sequence, the annotation files `annotations_io` defines, plus
    `aistpp_<split>_sequences.json` and `aistpp_<split>_time_offsets.json` at
    the dataset root. Sequences whose videos were not downloaded are skipped,
    so preprocessing a subset needs no extra configuration.

    Args:
        dataset_dir: Dataset root, as passed to the download script.
        split: AIST++ split to preprocess, one of `AISTPP_SPLITS`.
        drop_ignored: Whether to drop the sequences AIST++ flags as poorly
            reconstructed in its `ignore_list.txt`.
        min_match_margin: Least decisive raw-to-refined frame match accepted,
            as returned by `estimate_frame_offset`. A sequence with a weaker
            match on any of its views is dropped rather than annotated against
            a frame offset that may be wrong.
        skip_extract: Whether to skip unpacking the annotation archives.

    Raises:
        ValueError: If an unknown split is requested.
        FileNotFoundError: If the annotation archives are missing.
    """
    if split not in AISTPP_SPLITS:
        raise ValueError(
            f"Unknown split '{split}', expected any of {list(AISTPP_SPLITS)}."
        )

    if not skip_extract:
        _extract_annotations(dataset_dir)

    sequence_names = _load_split_sequences(dataset_dir, split, drop_ignored)
    mapping, settings = _load_camera_settings(dataset_dir)

    sequences_infos = []
    time_offsets = {}
    n_missing_videos = 0
    n_weak_matches = 0

    pbar = tqdm(sequence_names, desc=f"Preprocessing AIST++ ({split})")

    for sequence_name in pbar:
        pbar.set_postfix_str(sequence_name)

        setting = settings[mapping[sequence_name]]
        camera_names = sorted(setting)

        videos = _find_sequence_videos(dataset_dir, sequence_name, camera_names)
        if not videos:
            n_missing_videos += 1
            continue

        offsets = _estimate_sequence_offsets(dataset_dir, videos)

        weak_views = [
            camera_name
            for camera_name, offset in offsets.items()
            if offset["match_margin"] < min_match_margin
        ]
        if weak_views:
            tqdm.write(
                f"Skipping {sequence_name}: raw-to-refined match too weak on "
                f"{weak_views}."
            )
            n_weak_matches += 1
            continue

        keypoints_3d = _load_keypoints_3d(dataset_dir, sequence_name)

        # AIST++ annotates a handful of frames past the end of the refined
        # videos, and a raw recording may stop before its own last annotated
        # frame, so the sequence is as long as its shortest evidence.
        n_frames = min(
            [keypoints_3d.shape[0]]
            + [offset["n_refined_frames"] for offset in offsets.values()]
            + [
                offset["n_raw_frames"] - offset["offset_frames"]
                for offset in offsets.values()
            ]
        )
        keypoints_3d = keypoints_3d[:n_frames]

        matched_cameras = sorted(videos)

        kps3d_annotations = Keypoints3DAnnotations(
            metadata=Keypoints3DAnnotationsMetadata(formats=[KEYPOINTS_FORMAT]),
            annotations=[
                Keypoints3DAnnotation(
                    frame_idx=frame_idx,
                    subject_id=sequence_name,
                    xyz=keypoints_3d[frame_idx],
                    scores=torch.ones_like(keypoints_3d[frame_idx][..., 0]),
                    format=KEYPOINTS_FORMAT.name,
                )
                for frame_idx in range(n_frames)
            ],
        )

        camera_intrinsics_annotations, camera_extrinsics_annotations = (
            _build_camera_annotations(setting, matched_cameras)
        )

        kps2d_annotations = _shift_kps2d_to_local_frames(
            generate_kps2d_from_kps3d_and_cameras(
                kps_annotations=kps3d_annotations,
                cam_intrinsics=camera_intrinsics_annotations,
                cam_extrinsics=camera_extrinsics_annotations,
            ),
            offsets,
        )

        bboxes2d_annotations = generate_bboxes2d_from_kps2d(
            kps2d_annotations=kps2d_annotations,
            views_resolution_hw={
                camera_name: tuple(setting[camera_name]["size"][::-1])
                for camera_name in matched_cameras
            },
            category_id=0,
        )

        camera_temporal_annotations = _build_camera_temporal_annotations(
            offsets, matched_cameras[0]
        )

        annotations_relpath = os.path.join("annotations", "kineo", sequence_name)
        annotations_relpaths = annotations_io.write_sequence_annotations(
            dataset_dir=dataset_dir,
            annotations_reldir=annotations_relpath,
            annotations={
                "keypoints_2d": kps2d_annotations,
                "keypoints_3d": kps3d_annotations,
                "bboxes_2d": bboxes2d_annotations,
                "cameras_temporal": camera_temporal_annotations,
                "cameras_intrinsics": camera_intrinsics_annotations,
                "cameras_extrinsics": camera_extrinsics_annotations,
            },
        )

        time_offsets[sequence_name] = offsets

        sequences_infos.append(
            {
                "sequence_name": sequence_name,
                "split": split,
                "n_frames": n_frames,
                "annotations": annotations_relpaths,
                "views": {
                    camera_name: {"video_path": videos[camera_name]}
                    for camera_name in matched_cameras
                },
            }
        )

    pbar.close()

    _write_json(
        os.path.join(dataset_dir, f"aistpp_{split}_sequences.json"),
        sequences_infos,
    )
    _write_json(
        os.path.join(dataset_dir, f"aistpp_{split}_time_offsets.json"),
        time_offsets,
    )

    print(
        f"Preprocessed {len(sequences_infos)} sequences, skipped "
        f"{n_missing_videos} with no downloaded video and {n_weak_matches} "
        "with an inconclusive raw-to-refined match."
    )


def _load_keypoints_3d(dataset_dir: str, sequence_name: str) -> torch.Tensor:
    """Reads a sequence's optimized 3D keypoints, in meters."""
    keypoints_path = os.path.join(
        dataset_dir, "annotations", "keypoints3d", f"{sequence_name}.pkl"
    )

    with open(keypoints_path, "rb") as f:
        keypoints_3d = pickle.load(f)["keypoints3d_optim"]

    return torch.from_numpy(
        np.ascontiguousarray(keypoints_3d) * CENTIMETERS_TO_METERS
    ).to(torch.float32)


def _write_json(filepath: str, content):
    with open(filepath, "wb") as f:
        f.write(
            orjson.dumps(
                content,
                default=_serializer_fallback,
                option=orjson.OPT_INDENT_2,
            )
        )
    print(f'Saved "{filepath}"')

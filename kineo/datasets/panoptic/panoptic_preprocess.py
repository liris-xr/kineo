# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Preprocessing of the CMU Panoptic Studio under the TEMPO protocol.

TEMPO (Choudhury et al., ICCV 2023) evaluates on five HD views of the dome,
cameras 3, 6, 12, 13 and 23, over whole sequences read at every third annotated
frame. Its configuration, `configs/panoptic/resnet_rnn_panoptic_cam5.py`, sets
`seq_frame_interval=3` on both the train and the test split and marks the test
split `subset='test'`, which is what keeps it off the `subset == 'validation'`
branch that truncates a sequence to its first 30%. The 29.97Hz HD stream read
every third frame is the 9.99Hz this module writes.

The dome is genlocked, so no `cameras_temporal` annotations are emitted and the
views read back as mutually synchronized. 2D keypoints are reprojected from the
3D ones rather than read from the dataset, so they agree with the cameras, as
the other Kineo datasets do.
"""

import glob
import os
import re
from pathlib import Path
from typing import Sequence

import cv2
import orjson
import torch
from tqdm import tqdm

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
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.annotations.keypoints_utils import (
    generate_kps2d_from_kps3d_and_cameras,
)
from kineo.datasets import annotations_io
from kineo.datasets.panoptic.panoptic_download import (
    HD_PANEL,
    TEMPO_HD_CAMERAS,
    TEMPO_TEST_SEQUENCES,
    split_sequences,
)
from kineo.datasets.preprocess_utils import standardize_keypoints
from kineo.geometry.conversions import (
    OPENCV_WORLD_BASIS,
    RH_NEG_Y_UP_BASIS,
    convert_Rt_basis,
)
from kineo.io.file import extract_tar_with_progress
from kineo.pipeline.stages.nlf.skeleton_keypoints_format import (
    COCO_19_KEYPOINTS_FORMAT,
)

# Panoptic keypoints and camera translations are expressed in centimeters.
CENTIMETERS_TO_METERS = 0.01

# Every HD stream of the dome, and every 'fpsType' the 3D body annotations
# carry, is 'hd_29_97'.
HD_FPS = 30000 / 1001

# TEMPO reads every third annotated frame, on both of its splits.
TEMPO_FRAME_INTERVAL = 3

# The dome's world axes are +X right, +Y down, +Z forward: right-handed with Y
# pointing down. VoxelPose and TEMPO spell the same change of basis as a
# hardcoded M = [[1, 0, 0], [0, 0, -1], [0, 1, 0]].
PANOPTIC_WORLD_BASIS = RH_NEG_Y_UP_BASIS

KEYPOINTS_FORMAT = COCO_19_KEYPOINTS_FORMAT

# Index of 'pelvis' in coco_19, the joint a body is accepted on. TEMPO applies
# the same threshold, but to the joint its COCO-17 reordering puts at index 11,
# which is the left hip.
ROOT_KEYPOINT_IDX = 2
MIN_ROOT_KEYPOINT_SCORE = 0.1

KEYPOINTS_3D_ARCHIVE = "hdPose3d_stage1_coco19.tar"
KEYPOINTS_3D_DIRNAME = "hdPose3d_stage1_coco19"

SEQUENCES_FILENAME = "panoptic_tempo_sequences.json"

_FRAME_NUMBER_RE = re.compile(r"body3DScene_(\d+)\.json$")


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


def _view_id(node: int) -> str:
    """Names an HD camera the way the dome's calibration does, e.g. '00_03'."""
    return f"{HD_PANEL:02d}_{node:02d}"


def _extract_keypoints_3d(sequence_dir: str):
    """Unpacks a sequence's 3D body keypoint archive next to itself.

    Raises:
        FileNotFoundError: If the archive is missing.
    """
    archive_path = os.path.join(sequence_dir, KEYPOINTS_3D_ARCHIVE)

    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"Missing keypoints archive {archive_path}. Download the sequence "
            "first."
        )

    extract_tar_with_progress(
        archive_path,
        sequence_dir,
        desc=f"Extracting {os.path.basename(sequence_dir)} keypoints",
        leave=False,
    )


def _find_sequence_videos(
    dataset_dir: str, sequence: str, cameras: Sequence[int]
) -> dict[str, str]:
    """Maps each requested camera with a downloaded video to that video.

    Args:
        dataset_dir: Absolute path to the dataset root.
        sequence: Sequence name, e.g. "160906_pizza1".
        cameras: HD camera nodes the protocol reads.

    Returns:
        Maps a view id to its video path relative to `dataset_dir`, for the
        cameras whose video is on disk.
    """
    videos = {}

    for node in cameras:
        relpath = os.path.join(
            sequence, "hdVideos", f"hd_{HD_PANEL:02d}_{node:02d}.mp4"
        )
        if os.path.exists(os.path.join(dataset_dir, relpath)):
            videos[_view_id(node)] = Path(relpath).as_posix()

    return videos


def _count_video_frames(video_path: str) -> int:
    """Reads a video's frame count without decoding it.

    Uses the same container metadata `VideoLoader` trusts, so a frame this
    accepts is a frame the loader will accept too.
    """
    reader = cv2.VideoCapture(video_path)
    try:
        return int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        reader.release()


def _annotated_frame_numbers(sequence_dir: str) -> list[int]:
    """Lists the HD frame numbers a sequence has 3D body annotations for.

    The annotation file names carry the index of the HD frame they describe,
    which is the index `panoptic-toolbox`'s own extraction gives that frame.
    """
    keypoints_dir = os.path.join(sequence_dir, KEYPOINTS_3D_DIRNAME)

    frame_numbers = []
    for path in glob.iglob(os.path.join(keypoints_dir, "body3DScene_*.json")):
        match = _FRAME_NUMBER_RE.search(os.path.basename(path))
        if match is not None:
            frame_numbers.append(int(match.group(1)))

    return sorted(frame_numbers)


def _load_frame_bodies(
    sequence_dir: str, frame_number: int
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Reads the bodies annotated on one HD frame.

    Args:
        sequence_dir: Absolute path to the sequence directory.
        frame_number: HD frame number, as the file name spells it.

    Returns:
        One (subject id, keypoints, scores) triple per accepted body, with
        keypoints of shape (19, 3) still in centimeters and in the dome's own
        basis, and scores of shape (19,) clamped to be non-negative. Bodies
        whose root joint scores at or below `MIN_ROOT_KEYPOINT_SCORE` are left
        out, as the protocol does. Empty for a frame the dome did not annotate:
        a run of annotations holds the odd hole, one frame wide.
    """
    path = os.path.join(
        sequence_dir, KEYPOINTS_3D_DIRNAME, f"body3DScene_{frame_number:08d}.json"
    )

    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        bodies = orjson.loads(f.read())["bodies"]

    accepted = []

    for body in bodies:
        joints = torch.tensor(body["joints19"], dtype=torch.float32).reshape(
            -1, 4
        )
        scores = joints[:, 3]

        if scores[ROOT_KEYPOINT_IDX] <= MIN_ROOT_KEYPOINT_SCORE:
            continue

        accepted.append(
            (str(body["id"]), joints[:, :3], scores.clamp(min=0.0))
        )

    return accepted


def _load_camera_annotations(
    sequence_dir: str, sequence: str, view_ids: list[str]
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    dict[str, tuple[int, int]],
]:
    """Reads a sequence's calibration for the views the protocol reads.

    The dome stores the rotation and translation that take a world point to the
    camera frame, which is what the annotations want, so only the units and the
    world basis change.

    Returns:
        The intrinsics, the extrinsics, and each view's (height, width).

    Raises:
        KeyError: If the calibration does not describe one of `view_ids`.
    """
    calibration_path = os.path.join(
        sequence_dir, f"calibration_{sequence}.json"
    )

    with open(calibration_path, "rb") as f:
        cameras = {
            camera["name"]: camera
            for camera in orjson.loads(f.read())["cameras"]
        }

    intrinsics = []
    extrinsics = []
    resolutions_hw = {}

    for view_id in view_ids:
        if view_id not in cameras:
            raise KeyError(
                f"Camera '{view_id}' is missing from {calibration_path}."
            )

        camera = cameras[view_id]
        width, height = camera["resolution"]
        resolutions_hw[view_id] = (height, width)

        intrinsics.append(
            CameraIntrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                K=torch.tensor(camera["K"], dtype=torch.float32),
                # The dome stores OpenCV's (k1, k2, p1, p2, k3), Brown-Conrady's
                # own coefficients in the order the annotation expects.
                distortion_coefficients=torch.tensor(
                    camera["distCoef"], dtype=torch.float32
                ),
                distortion_model=CameraDistortionModel.BROWN_CONRADY,
                resolution_hw=(height, width),
            )
        )

        R = torch.tensor(camera["R"], dtype=torch.float32)
        t = torch.tensor(camera["t"], dtype=torch.float32).reshape(3)
        Rt = torch.cat([R, (t * CENTIMETERS_TO_METERS).unsqueeze(1)], dim=1)
        Rt = convert_Rt_basis(Rt, PANOPTIC_WORLD_BASIS, OPENCV_WORLD_BASIS)

        extrinsics.append(
            CameraExtrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                R=Rt[:, :3].contiguous(),
                t=Rt[:, 3].contiguous(),
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
        resolutions_hw,
    )


def _build_keypoints_3d_annotations(
    sequence_dir: str, selected_frames: range
) -> Keypoints3DAnnotations:
    """Reads the selected frames' bodies into world-space 3D keypoints.

    Frames nobody was annotated on carry no annotation rather than an empty
    one, and a body that comes and goes is annotated on the frames it appears
    on, so the subject set varies over a sequence.

    Args:
        sequence_dir: Absolute path to the sequence directory.
        selected_frames: HD frame numbers read, in order. An annotation's
            `frame_idx` indexes into this range rather than into the video.

    Returns:
        The sequence's 3D keypoints, in meters and in `OPENCV_WORLD_BASIS`.
    """
    frame_idxs: list[int] = []
    subject_ids: list[str] = []
    keypoints: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []

    for frame_idx, frame_number in enumerate(selected_frames):
        for subject_id, body_keypoints, body_scores in _load_frame_bodies(
            sequence_dir, frame_number
        ):
            frame_idxs.append(frame_idx)
            subject_ids.append(subject_id)
            keypoints.append(body_keypoints)
            scores.append(body_scores)

    metadata = Keypoints3DAnnotationsMetadata(formats=[KEYPOINTS_FORMAT])

    if not keypoints:
        return Keypoints3DAnnotations(metadata=metadata, annotations=[])

    # Converted in one call rather than per body: the conversion moves to the
    # GPU when there is one, which is only worth paying for once.
    keypoints_world = standardize_keypoints(
        torch.stack(keypoints),
        src_world_basis=PANOPTIC_WORLD_BASIS,
        src_world_unit_in_meters=CENTIMETERS_TO_METERS,
    )

    return Keypoints3DAnnotations(
        metadata=metadata,
        annotations=[
            Keypoints3DAnnotation(
                frame_idx=frame_idxs[i],
                subject_id=subject_ids[i],
                xyz=keypoints_world[i],
                scores=scores[i],
                format=KEYPOINTS_FORMAT.name,
            )
            for i in range(len(keypoints))
        ],
    )


def _selected_frames(
    dataset_dir: str,
    sequence_dir: str,
    videos: dict[str, str],
    frame_interval: int,
) -> range | None:
    """Picks the HD frames of a sequence that are both annotated and recorded.

    Args:
        dataset_dir: Absolute path to the dataset root.
        sequence_dir: Absolute path to the sequence directory.
        videos: Maps a view id to its video path relative to `dataset_dir`.
        frame_interval: Read one frame out of this many annotated frames.

    Returns:
        The frame numbers to read, or `None` if the sequence has no annotated
        frame every one of its videos is long enough to cover. The range spans
        the annotated run rather than tracking it exactly: a frame the dome
        skipped is read and left without an annotation, which is what a frame
        nobody was detected on already looks like.
    """
    frame_numbers = _annotated_frame_numbers(sequence_dir)

    if not frame_numbers:
        return None

    first, last = frame_numbers[0], frame_numbers[-1]

    n_video_frames = min(
        _count_video_frames(os.path.join(dataset_dir, relpath))
        for relpath in videos.values()
    )

    # A sequence is as long as its shortest evidence: the dome sometimes
    # annotates past the end of the shortest recording.
    stop = min(last + 1, n_video_frames)

    if stop <= first:
        return None

    return range(first, stop, frame_interval)


def preprocess_panoptic(
    dataset_dir: str,
    split: str = "all",
    cameras: Sequence[int] = TEMPO_HD_CAMERAS,
    drop_ignored: bool = True,
    frame_interval: int = TEMPO_FRAME_INTERVAL,
    skip_extract: bool = False,
):
    """Preprocesses the CMU Panoptic Studio for use in Kineo.

    Writes `panoptic_tempo_sequences.json` at the dataset root, listing one
    entry per sequence with its own annotation files. Sequences whose videos
    were not downloaded are skipped, so preprocessing after downloading only
    the test split needs no extra configuration.

    Args:
        dataset_dir: Dataset root, as passed to the download script.
        split: TEMPO split to preprocess, one of `PANOPTIC_SPLITS`. The default
            covers both, and skips whatever is not on disk.
        cameras: HD camera nodes to read. The TEMPO protocol reads
            `TEMPO_HD_CAMERAS`.
        drop_ignored: Whether to drop the sequences of the repository's
            `panoptic_ignore_list.txt`.
        frame_interval: Read one annotated frame out of this many. TEMPO reads
            every third, which is 9.99Hz out of the dome's 29.97Hz.
        skip_extract: Whether to skip unpacking the keypoint archives.

    Raises:
        ValueError: If an unknown split or a non-positive frame interval is
            requested.
        FileNotFoundError: If a sequence's keypoint archive is missing.
    """
    if frame_interval < 1:
        raise ValueError(
            f"frame_interval must be at least 1, got {frame_interval}."
        )

    sequences = split_sequences(split, drop_ignored)
    test_sequences = set(TEMPO_TEST_SEQUENCES)

    sequences_infos = []
    n_missing_videos = 0
    n_missing_annotations = 0

    pbar = tqdm(sequences, desc=f"Preprocessing CMU Panoptic ({split})")

    for sequence in pbar:
        pbar.set_postfix_str(sequence)

        sequence_dir = os.path.join(dataset_dir, sequence)

        videos = _find_sequence_videos(dataset_dir, sequence, cameras)
        if not videos:
            n_missing_videos += 1
            continue

        if not skip_extract:
            _extract_keypoints_3d(sequence_dir)

        selected_frames = _selected_frames(
            dataset_dir, sequence_dir, videos, frame_interval
        )
        if selected_frames is None:
            tqdm.write(f"Skipping {sequence}: no annotated frame is recorded.")
            n_missing_annotations += 1
            continue

        view_ids = sorted(videos)

        (
            camera_intrinsics_annotations,
            camera_extrinsics_annotations,
            resolutions_hw,
        ) = _load_camera_annotations(sequence_dir, sequence, view_ids)

        kps3d_annotations = _build_keypoints_3d_annotations(
            sequence_dir, selected_frames
        )
        kps2d_annotations = generate_kps2d_from_kps3d_and_cameras(
            kps_annotations=kps3d_annotations,
            cam_intrinsics=camera_intrinsics_annotations,
            cam_extrinsics=camera_extrinsics_annotations,
        )
        bboxes2d_annotations = generate_bboxes2d_from_kps2d(
            kps2d_annotations=kps2d_annotations,
            views_resolution_hw=resolutions_hw,
            category_id=0,
        )

        annotations_reldir = os.path.join(sequence, "annotations", "kineo")
        annotations_relpaths = annotations_io.write_sequence_annotations(
            dataset_dir=dataset_dir,
            annotations_reldir=annotations_reldir,
            annotations={
                "keypoints_2d": kps2d_annotations,
                "keypoints_3d": kps3d_annotations,
                "bboxes_2d": bboxes2d_annotations,
                "cameras_intrinsics": camera_intrinsics_annotations,
                "cameras_extrinsics": camera_extrinsics_annotations,
            },
        )

        sequences_infos.append(
            {
                "sequence_name": sequence,
                "split": "test" if sequence in test_sequences else "train",
                "n_frames": len(selected_frames),
                "fps": HD_FPS / frame_interval,
                "annotations": annotations_relpaths,
                "views": {
                    view_id: {
                        "video_path": videos[view_id],
                        "selected_frames": selected_frames,
                    }
                    for view_id in view_ids
                },
            }
        )

    pbar.close()

    _write_json(
        os.path.join(dataset_dir, SEQUENCES_FILENAME), sequences_infos
    )

    print(
        f"Preprocessed {len(sequences_infos)} sequences, skipped "
        f"{n_missing_videos} with no downloaded video and "
        f"{n_missing_annotations} with no recorded annotated frame."
    )

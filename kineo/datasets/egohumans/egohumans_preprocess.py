# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.io.file import extract_tar_with_progress
from glob import glob
import os
from tqdm import tqdm
import numpy as np
from kineo.datasets.egohumans.configs import CONFIG_ROOT_DIR, load_config
from typing import Any
import torch
import roma
import pickle
from kineo.annotations.keypoints_format import KeypointsFormat

from kineo.io.frame_sequence_loader import ImagesLoader
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.visualization.viz_3d import show_keypoints_and_cameras
from kineo.visualization.viz_2d import show_bboxes_and_reproj_keypoints
from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotations,
    GlobalTimeReferenceAnnotationsMetadata,
    GlobalTimeReferenceAnnotation,
)
from kineo.geometry.camera import inverse_Rt
from kineo.annotations.bboxes_utils import generate_bboxes2d_from_kps3d_and_cameras
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_utils import generate_kps2d_from_kps3d_and_cameras
from kineo.annotations.bboxes_2d import (
    BBox2DAnnotation,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
)
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.annotations.camera_temporal import (
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
    CameraTemporalAnnotation,
)
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
    CameraDistortionModel,
)
import orjson
import warnings
from pathlib import Path

FPS = 20

SEQUENCES = [
    {
        "directory": "01_tagging",
        "name": "tagging",
        "n_subsequences": 14,
    },
    {
        "directory": "02_lego",
        "name": "legoassemble",
        "n_subsequences": 6,
    },
    {
        "directory": "03_fencing",
        "name": "fencing",
        "n_subsequences": 14,
    },
    {
        "directory": "04_basketball",
        "name": "basketball",
        "n_subsequences": 14,
        "excluded_subsequences": [9],  # 010_basketball is missing processed_data
    },
    {
        "directory": "05_volleyball",
        "name": "volleyball",
        "n_subsequences": 11,
    },
    {
        "directory": "06_badminton",
        "name": "badminton",
        "n_subsequences": 61,
        "excluded_subsequences": [50],  # 051_badminton is missing colmap data
    },
    {
        "directory": "07_tennis",
        "name": "tennis",
        "n_subsequences": 13,
    },
]


def preprocess_egohumans(
    dataset_dir: str,
    skip_extract: bool = False,
    skip_annotations_generation: bool = False,
    use_original_bboxes: bool = False,
    sequences: list[str] = None,
    show: bool = False,
    merge_translation_tolerance: float = 0.02,
    merge_rotation_tolerance: float = 1.0,
):
    if not skip_extract:
        _extract_data(dataset_dir, sequences)

    _generate_annotations(
        dataset_dir,
        skip_annotations_generation=skip_annotations_generation,
        use_original_bboxes=use_original_bboxes,
        included_subsequences=sequences,
        show=show,
        merge_translation_tolerance=merge_translation_tolerance,
        merge_rotation_tolerance=merge_rotation_tolerance,
    )


def _generate_annotations(
    dataset_dir: str,
    skip_annotations_generation: bool = False,
    use_original_bboxes: bool = False,
    included_subsequences: list[str] = None,
    show: bool = False,
    merge_translation_tolerance: float = 0.02,
    merge_rotation_tolerance: float = 1.0,
):
    total_n_subsequences = sum(sequence["n_subsequences"] for sequence in SEQUENCES)

    pbar = tqdm(
        total=total_n_subsequences,
        desc="Generating annotations",
        leave=True,
    )

    if included_subsequences is not None:
        included_subsequences = [s.split("_") for s in included_subsequences]
        try:
            included_subsequences = [
                (sequence[0], int(sequence[1])) for sequence in included_subsequences
            ]
        except ValueError:
            raise ValueError(
                "Invalid subsequence format, expected format: 'sequencename_subsequenceidx'"
            )
    included_subsequences: list[tuple[int, str]] = included_subsequences

    sequences_infos: list[dict[str, Any]] = []

    for sequence in SEQUENCES:
        sequence_dir = sequence["directory"]
        sequence_name = sequence["name"]
        n_subsequences = sequence["n_subsequences"]

        for subsequence_idx in range(n_subsequences):
            pbar.set_postfix(sequence=f"{subsequence_idx + 1:03d}_{sequence_name}")

            if subsequence_idx in sequence.get("excluded_subsequences", []):
                tqdm.write(
                    f"Skipping subsequence '{subsequence_idx + 1:03d}_{sequence_name}' because it is excluded"
                )
                pbar.update(1)
                continue

            sequence_dir = sequence["directory"]
            sequence_name = sequence["name"]
            subsequence_dir = os.path.join(
                sequence_dir, f"{subsequence_idx + 1:03d}_{sequence_name}"
            )
            subsequence_abs_dir = os.path.join(dataset_dir, subsequence_dir)
            subseq_annotations_dir = os.path.join(subsequence_dir, "annotations")

            if (
                included_subsequences is not None
                and (sequence_name, subsequence_idx + 1) not in included_subsequences
            ):
                pbar.update(1)
                tqdm.write(
                    f"Skipping subsequence '{subsequence_idx + 1:03d}_{sequence_name}' because it is not in the list of sequences to process"
                )
                continue

            os.makedirs(
                os.path.join(dataset_dir, subseq_annotations_dir), exist_ok=True
            )

            kps_2d_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "keypoints_2d.json",
            )
            kps_2d_annotations_abspath = os.path.join(
                dataset_dir, kps_2d_annotations_relpath
            )

            kps_3d_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "keypoints_3d.json",
            )
            kps_3d_annotations_abspath = os.path.join(
                dataset_dir, kps_3d_annotations_relpath
            )

            bboxes_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "bboxes_2d.json",
            )
            bboxes_annotations_abspath = os.path.join(
                dataset_dir, bboxes_annotations_relpath
            )

            cam_temporal_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "cameras_temporal.json",
            )
            cam_temporal_annotations_abspath = os.path.join(
                dataset_dir, cam_temporal_annotations_relpath
            )

            cam_intrinsics_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "cameras_intrinsics.json",
            )
            cam_intrinsics_annotations_abspath = os.path.join(
                dataset_dir, cam_intrinsics_annotations_relpath
            )

            cam_extrinsics_annotations_relpath = os.path.join(
                subseq_annotations_dir,
                "cameras_extrinsics.json",
            )
            cam_extrinsics_annotations_abspath = os.path.join(
                dataset_dir, cam_extrinsics_annotations_relpath
            )

            sequence_dir = sequence["directory"]
            sequence_name = sequence["name"]
            subsequence_dir = os.path.join(
                sequence_dir, f"{subsequence_idx + 1:03d}_{sequence_name}"
            )
            subsequence_abs_dir = os.path.join(dataset_dir, subsequence_dir)
            config_path = os.path.join(
                CONFIG_ROOT_DIR,
                sequence_name,
                f"{subsequence_idx + 1:03d}_{sequence_name}.yaml",
            )
            cfg = load_config(config_path)

            exo_dir = os.path.join(subsequence_abs_dir, "exo")
            included_cams = [
                exo_camera_name
                for exo_camera_name in sorted(os.listdir(exo_dir))
                if exo_camera_name not in cfg.INVALID_EXOS
                and exo_camera_name.startswith("cam")
            ]

            # We load the cameras annotations here because we need to know the views ids that are valid (i.e. static only)
            cam_intrinsics_annotations, cam_extrinsics_annotations = (
                _load_cameras_annotations(
                    included_cams=included_cams,
                    colmap_dir=os.path.join(subsequence_abs_dir, "colmap/workplace"),
                    cfg=cfg,
                    static_only=True,
                    merge_static_extrinsics=True,
                    merge_translation_tolerance=merge_translation_tolerance,
                    merge_rotation_tolerance=merge_rotation_tolerance,
                )
            )

            views_ids = cam_extrinsics_annotations.views_ids

            if not skip_annotations_generation:
                (
                    kps_2d_annotations,
                    kps_3d_annotations,
                    bboxes_annotations,
                    cam_temporal_annotations,
                ) = _generate_subsequence_annotations(
                    subsequence_abs_dir=subsequence_abs_dir,
                    cam_intrinsics_annotations=cam_intrinsics_annotations,
                    cam_extrinsics_annotations=cam_extrinsics_annotations,
                    cfg=cfg,
                    use_original_bboxes=use_original_bboxes,
                    show=show,
                )

                with open(kps_2d_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(kps_2d_annotations.to_dict()))
                tqdm.write(
                    f"Saved keypoints 2d annotations to {kps_2d_annotations_abspath}"
                )

                with open(kps_3d_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(kps_3d_annotations.to_dict()))
                tqdm.write(
                    f"Saved keypoints 3d annotations to {kps_3d_annotations_abspath}"
                )

                with open(bboxes_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(bboxes_annotations.to_dict()))
                tqdm.write(
                    f"Saved bboxes 2d annotations to {bboxes_annotations_abspath}"
                )

                with open(cam_temporal_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(cam_temporal_annotations.to_dict()))
                tqdm.write(
                    f"Saved cameras temporal annotations to {cam_temporal_annotations_abspath}"
                )

                with open(cam_intrinsics_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(cam_intrinsics_annotations.to_dict()))
                tqdm.write(
                    f"Saved cameras intrinsics to {cam_intrinsics_annotations_abspath}"
                )

                with open(cam_extrinsics_annotations_abspath, "wb") as f:
                    f.write(orjson.dumps(cam_extrinsics_annotations.to_dict()))
                tqdm.write(
                    f"Saved cameras extrinsics to {cam_extrinsics_annotations_abspath}"
                )

            sequences_infos.append(
                {
                    "sequence_name": f"{sequence_name}_{subsequence_idx + 1:03d}",
                    "annotations": {
                        "keypoints_2d": Path(kps_2d_annotations_relpath).as_posix(),
                        "keypoints_3d": Path(kps_3d_annotations_relpath).as_posix(),
                        "bboxes_2d": Path(bboxes_annotations_relpath).as_posix(),
                        "cameras_temporal": Path(
                            cam_temporal_annotations_relpath
                        ).as_posix(),
                        "cameras_intrinsics": Path(
                            cam_intrinsics_annotations_relpath
                        ).as_posix(),
                        "cameras_extrinsics": Path(
                            cam_extrinsics_annotations_relpath
                        ).as_posix(),
                    },
                    "views": {
                        view_id: {
                            "images_dir": (
                                Path(subsequence_dir) / "exo" / view_id / "images"
                            ).as_posix(),
                            "fps": FPS,
                        }
                        for view_id in views_ids
                    },
                }
            )

            pbar.update(1)

    sequences_annotations_filename = os.path.join(
        dataset_dir,
        "egohumans_sequences.json",
    )

    with open(sequences_annotations_filename, "wb") as f:
        f.write(orjson.dumps(sequences_infos, option=orjson.OPT_INDENT_2))
    tqdm.write(f"Saved sequences annotations to {sequences_annotations_filename}")


def _generate_subsequence_annotations(
    subsequence_abs_dir: str,
    cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    cfg: dict[str, Any],
    use_original_bboxes: bool = False,
    show: bool = False,
) -> tuple[
    Keypoints2DAnnotations,
    Keypoints3DAnnotations,
    BBox2DAnnotations,
    CameraTemporalAnnotations,
]:
    assert cam_intrinsics_annotations.views_ids == cam_extrinsics_annotations.views_ids

    cameras = cam_extrinsics_annotations.views_ids

    kps_3d_annotations = _load_keypoints_annotations(
        os.path.join(subsequence_abs_dir, "processed_data", "fit_poses3d"),
        cfg,
    )
    has_bboxes = os.path.exists(
        os.path.join(subsequence_abs_dir, "processed_data", "bboxes")
    )

    if has_bboxes and use_original_bboxes:
        bboxes_annotations = _load_bboxes_annotations(
            cameras,
            os.path.join(subsequence_abs_dir, "processed_data", "bboxes"),
            kps_annotations=kps_3d_annotations,
            cam_intrinsics=cam_intrinsics_annotations,
            cam_extrinsics=cam_extrinsics_annotations,
            cfg=cfg,
        )
    else:
        bboxes_annotations = generate_bboxes2d_from_kps3d_and_cameras(
            kps_annotations=kps_3d_annotations,
            cam_intrinsics=cam_intrinsics_annotations,
            cam_extrinsics=cam_extrinsics_annotations,
            category_id=0,
        )

    kps_2d_annotations = generate_kps2d_from_kps3d_and_cameras(
        kps_annotations=kps_3d_annotations,
        cam_intrinsics=cam_intrinsics_annotations,
        cam_extrinsics=cam_extrinsics_annotations,
    )

    if show:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        views = []
        for camera in cameras:
            images_dir = os.path.join(subsequence_abs_dir, "exo", camera, "images")
            imgs_paths = sorted(
                glob(
                    os.path.join(
                        images_dir,
                        "*.jpg",
                    )
                )
            )
            n_imgs = len(imgs_paths)

            frame_timestamps_local = (torch.arange(n_imgs) / FPS).tolist()

            views.append(
                ViewInput(
                    view_id=camera,
                    frame_loader=ImagesLoader(
                        img_paths=imgs_paths,
                        frame_timestamps_local=frame_timestamps_local,
                        device=device,
                    ),
                    audio_loader=None,
                )
            )

        global_time_reference = GlobalTimeReferenceAnnotations(
            metadata=GlobalTimeReferenceAnnotationsMetadata(),
            annotations=[
                GlobalTimeReferenceAnnotation(
                    timestamps=torch.arange(len(views[0]["frame_loader"])) / FPS,
                    closest_local_frame_idx={
                        view["view_id"]: torch.arange(len(view["frame_loader"]))
                        for view in views
                    },
                )
            ],
        )
        show_bboxes_and_reproj_keypoints(
            views=views,
            bboxes_2d=bboxes_annotations,
            keypoints_3d=kps_3d_annotations,
            camera_extrinsics=cam_extrinsics_annotations,
            camera_intrinsics=cam_intrinsics_annotations,
            global_time_reference=global_time_reference,
            fps=FPS,
        )
        show_keypoints_and_cameras(
            kps_3d_annotations,
            cam_extrinsics_annotations,
            cam_intrinsics_annotations,
            fps=FPS,
        )

    # All cameras are synchronized
    cam_temporal_annotations = CameraTemporalAnnotations(
        metadata=CameraTemporalAnnotationsMetadata(),
        annotations=[
            CameraTemporalAnnotation(view_id=view_id, frame_idx=0, time_offset=0.0)
            for view_id in cameras
        ],
    )

    return (
        kps_2d_annotations,
        kps_3d_annotations,
        bboxes_annotations,
        cam_temporal_annotations,
    )


def _load_cameras_annotations(
    included_cams: list[str],
    colmap_dir: str,
    cfg: dict[str, Any],
    static_only: bool = True,
    merge_static_extrinsics: bool = True,
    merge_translation_tolerance: float = 0.02,  # in meters
    merge_rotation_tolerance: float = 1.0,  # in degrees
) -> tuple[CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations]:
    cam_intrinsics_annotations = _load_cameras_intrinsics(
        included_cams, colmap_dir, cfg
    )
    cam_extrinsics_annotations = _load_cameras_extrinsics(
        included_cams, colmap_dir, cfg
    )

    if merge_static_extrinsics:
        cam_extrinsics_annotations = _merge_extrinsics_annotations(
            cam_extrinsics_annotations,
            translation_tolerance=merge_translation_tolerance,
            rotation_tolerance=merge_rotation_tolerance,
        )

    if static_only:
        new_annotations: list[CameraExtrinsicsAnnotation] = []
        views_ids = cam_extrinsics_annotations.views_ids
        for view_id in views_ids:
            view_annotations = cam_extrinsics_annotations.filter_by_view_id(view_id)
            if len(view_annotations) == 1:
                new_annotations.append(view_annotations.first_or_default())
            else:
                tqdm.write(
                    f"View {view_id} has multiple extrinsics annotations (not static), not including it."
                )
        cam_extrinsics_annotations = CameraExtrinsicsAnnotations(
            metadata=cam_extrinsics_annotations.metadata,
            annotations=new_annotations,
        )

    # Remove any camera in intrinsics annotations that is not both in the intrinsics and the extrinsics annotations
    extrinsics_views_ids = cam_extrinsics_annotations.views_ids
    intrinsics_views_ids = cam_intrinsics_annotations.views_ids
    intersection_views_ids = set(extrinsics_views_ids) & set(intrinsics_views_ids)
    cam_intrinsics_annotations = cam_intrinsics_annotations.filter_by_view_ids(
        intersection_views_ids
    )
    cam_extrinsics_annotations = cam_extrinsics_annotations.filter_by_view_ids(
        intersection_views_ids
    )
    return cam_intrinsics_annotations, cam_extrinsics_annotations


def _merge_extrinsics_annotations(
    cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    translation_tolerance: float = 0.01,  # in meters
    rotation_tolerance: float = 0.5,  # in degrees
) -> CameraExtrinsicsAnnotations:
    """
    If consecutive extrinsics annotations are close enough in translation and rotation, keep only one annotation.
    """
    new_extrinsics_annotations: list[CameraExtrinsicsAnnotation] = []

    views_ids = cam_extrinsics_annotations.views_ids

    for view_id in views_ids:
        view_annotations = cam_extrinsics_annotations.filter_by_view_id(view_id)
        view_annotations = sorted(
            view_annotations.annotations, key=lambda x: x.frame_idx
        )

        new_view_extrinsics_annotations: list[CameraExtrinsicsAnnotation] = []

        for ann in view_annotations:
            if len(new_view_extrinsics_annotations) == 0:
                new_view_extrinsics_annotations.append(ann)
                continue

            last_view_ann = new_view_extrinsics_annotations[-1]

            last_view_Rt_inv = inverse_Rt(last_view_ann.Rt)
            last_view_cam_pose = last_view_Rt_inv[:3, 3]
            last_view_cam_rot = last_view_Rt_inv[:3, :3]

            Rt_inv = inverse_Rt(ann.Rt)
            cam_pose = Rt_inv[:3, 3]
            cam_rot = Rt_inv[:3, :3]

            translation_diff = torch.linalg.norm(cam_pose - last_view_cam_pose)
            rotation_diff = (
                roma.rotmat_geodesic_distance(cam_rot, last_view_cam_rot)
                .rad2deg()
                .squeeze()
            )
            if (
                translation_diff < translation_tolerance
                and rotation_diff < rotation_tolerance
            ):
                continue

            new_view_extrinsics_annotations.append(ann)

        # If the view has only one annotation, set the frame_idx to 0 for simplicity
        if len(new_view_extrinsics_annotations) == 1:
            new_view_extrinsics_annotations[0] = CameraExtrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                R=new_view_extrinsics_annotations[0].R,
                t=new_view_extrinsics_annotations[0].t,
            )
        new_extrinsics_annotations.extend(new_view_extrinsics_annotations)

    new_extrinsics_annotations = CameraExtrinsicsAnnotations(
        metadata=cam_extrinsics_annotations.metadata,
        annotations=new_extrinsics_annotations,
    )

    return new_extrinsics_annotations


def _load_bboxes_annotations(
    cameras: list[str],
    bboxes_dir: str,
    kps_annotations: Keypoints3DAnnotations,
    cam_intrinsics: CameraIntrinsicsAnnotations,
    cam_extrinsics: CameraExtrinsicsAnnotations,
    cfg: dict[str, Any],
) -> BBox2DAnnotations:
    bboxes_annotations: list[BBox2DAnnotation] = []
    total_files = 0

    cameras_bboxes_files = {
        camera_name: sorted(glob(os.path.join(bboxes_dir, camera_name, "rgb", "*.npy")))
        for camera_name in cameras
    }

    total_files += sum(len(files) for files in cameras_bboxes_files.values())
    pbar = tqdm(total=total_files, desc="Loading bounding boxes", leave=False)

    if total_files == 0:
        raise FileNotFoundError(f"No bboxes files found in {bboxes_dir}")

    for camera_name, camera_bboxes_files in cameras_bboxes_files.items():
        if len(camera_bboxes_files) == 0:
            warnings.warn(
                f"No bboxes files found for camera '{camera_name}' in {os.path.join(bboxes_dir, camera_name)}. Generating bboxes from keypoints 3d."
            )
            generated_bboxes_annotations = generate_bboxes2d_from_kps3d_and_cameras(
                kps_annotations=kps_annotations,
                cam_intrinsics=cam_intrinsics.filter_by_view_id(camera_name),
                cam_extrinsics=cam_extrinsics.filter_by_view_id(camera_name),
                category_id=0,
            )

            bboxes_annotations.extend(generated_bboxes_annotations.annotations)
            continue

        for bbox_file in camera_bboxes_files:
            frame_idx = int(os.path.basename(bbox_file).split(".")[0])
            bboxes_data = np.load(bbox_file, allow_pickle=True)

            for bbox_data in bboxes_data:
                bbox_annotation = BBox2DAnnotation(
                    view_id=camera_name,
                    frame_idx=frame_idx,
                    subject_id=str(bbox_data["human_name"]),
                    category_id=0,
                    xyxy=torch.from_numpy(bbox_data["bbox"][:4]).float(),
                    score=float(bbox_data["bbox"][4]),
                )
                bboxes_annotations.append(bbox_annotation)

            pbar.update(1)

    pbar.close()

    bboxes_annotations: BBox2DAnnotations = BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(), annotations=bboxes_annotations
    )
    subject_ids = bboxes_annotations.subjects_ids
    subject_ids = [
        subject_id for subject_id in subject_ids if subject_id not in cfg.INVALID_ARIAS
    ]
    bboxes_annotations = bboxes_annotations.filter_by_subject_ids(subject_ids)

    return bboxes_annotations


def _load_keypoints_annotations(
    keypoints_dir: str, cfg: dict[str, Any]
) -> Keypoints3DAnnotations:
    kps_format = KeypointsFormat.from_mmpose_dataset("coco")

    kps_files = sorted(glob(os.path.join(keypoints_dir, "*.npy")))

    if len(kps_files) == 0:
        raise FileNotFoundError(f"No keypoints files found in {keypoints_dir}")

    pbar = tqdm(total=len(kps_files), desc="Loading keypoints", leave=False)
    kps_annotations: list[Keypoints3DAnnotation] = []

    for kps_file in kps_files:
        frame_idx = int(os.path.basename(kps_file).split(".")[0]) - 1
        data = np.load(kps_file, allow_pickle=True).item()

        for subject_id, subject_data in data.items():
            kps = torch.from_numpy(subject_data[:, :3]).float()

            if subject_id in cfg.INVALID_ARIAS:
                continue

            kps_annot = Keypoints3DAnnotation(
                frame_idx=frame_idx,
                subject_id=subject_id,
                xyz=kps,
                annotated=torch.ones_like(kps[:, 0], dtype=torch.bool),
                scores=torch.ones_like(kps[:, 0], dtype=torch.float32),
                format=kps_format.name,
            )
            kps_annotations.append(kps_annot)

        pbar.update(1)

    kps_annotations = Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(
            formats=[kps_format],
        ),
        annotations=kps_annotations,
    )

    return kps_annotations


def _get_camera_name_from_colmap_camera_id(colmap_dir: str, colmap_camera_id: int):
    """
    Get the camera name from the Colmap camera id.
    """
    extrinsics_calibration_file = os.path.join(colmap_dir, "images.txt")

    with open(extrinsics_calibration_file) as f:
        extrinsics = f.readlines()
        extrinsics = extrinsics[4:]  # drop the first 4 lines (commented lines)
        extrinsics = extrinsics[::2]  # only alternate lines

    for line in extrinsics:
        line = line.strip().split()
        camera_id = int(line[-2])
        image_path = line[-1]
        camera_name = image_path.split("/")[0]

        if camera_id == colmap_camera_id:
            return camera_name

    return None


def _load_coordinate_transform(colmap_dir: str, cfg: dict[str, Any]) -> np.ndarray:
    colmap_transforms_file = os.path.join(colmap_dir, "colmap_from_aria_transforms.pkl")

    with open(colmap_transforms_file, "rb") as handle:
        colmap_from_aria_transforms = pickle.load(handle)

    anchor_ego_camera = cfg.CALIBRATION.ANCHOR_EGO_CAMERA
    return colmap_from_aria_transforms[anchor_ego_camera]


def _load_cameras_intrinsics(
    included_cams: list[str], colmap_dir: str, cfg: dict[str, Any]
) -> CameraIntrinsicsAnnotations:
    intrinsics_calibration_file = os.path.join(colmap_dir, "cameras.txt")

    with open(intrinsics_calibration_file) as f:
        intrinsics = f.readlines()
        intrinsics = intrinsics[3:]  # drop the first 3 lines (commented lines)

    cam_intrinsics_annotations: list[CameraIntrinsicsAnnotation] = []

    for line in intrinsics:
        line = line.split()
        cam_id = int(line[0])
        cam_name = _get_camera_name_from_colmap_camera_id(
            colmap_dir=colmap_dir, colmap_camera_id=cam_id
        )

        if cam_name not in included_cams:
            continue

        if cam_name in cfg.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS:
            continue

        cam_model = line[1]
        assert cam_model == "OPENCV_FISHEYE"
        image_width = int(line[2])
        image_height = int(line[3])
        fx = float(line[4])
        fy = float(line[5])
        cx = float(line[6])
        cy = float(line[7])
        k1 = float(line[8])
        k2 = float(line[9])
        k3 = float(line[10])
        k4 = float(line[11])

        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = torch.tensor([k1, k2, k3, k4])

        cam_intrinsics_annot = CameraIntrinsicsAnnotation(
            view_id=cam_name,
            frame_idx=0,
            K=K,
            distortion_coefficients=D,
            distortion_model=CameraDistortionModel.OPENCV_FISHEYE,
            resolution_hw=(image_height, image_width),
        )
        cam_intrinsics_annotations.append(cam_intrinsics_annot)

    for cam_name in cfg.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS:
        if cam_name not in included_cams:
            continue

        idx = cfg.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS.index(cam_name)
        other_cam_name = cfg.CALIBRATION.MANUAL_INTRINSICS_FROM_EXO_CAMERAS[idx]
        other_cam_intrinsics = next(
            a for a in cam_intrinsics_annotations if a.view_id == other_cam_name
        )
        cam_intrinsics_annot = CameraIntrinsicsAnnotation(
            view_id=cam_name,
            frame_idx=0,
            K=other_cam_intrinsics.K.clone(),
            distortion_coefficients=other_cam_intrinsics.distortion_coefficients.clone(),
            distortion_model=other_cam_intrinsics.distortion_model,
            resolution_hw=other_cam_intrinsics.resolution_hw,
        )
        cam_intrinsics_annotations.append(cam_intrinsics_annot)

    cam_intrinsics_annotations = CameraIntrinsicsAnnotations(
        metadata=CameraIntrinsicsAnnotationsMetadata(),
        annotations=cam_intrinsics_annotations,
    )

    return cam_intrinsics_annotations


def _apply_coordinate_transform(
    extrinsics: np.ndarray, coordinate_transform: np.ndarray
) -> np.ndarray:
    # Fix incorrect rotation matrices as per https://github.com/rawalkhirodkar/egohumans/issues/12#issue-2630185222
    raw_R = extrinsics[:3, :3]
    raw_t = extrinsics[:3, 3]
    R = coordinate_transform[:3, :3]
    t = coordinate_transform[:3, 3]
    c = np.sqrt((R @ R.T)[0, 0])
    R = R / c
    # Apply the coordinate transform to go from the anchor (aria camera) to world frame
    new_R = raw_R @ R
    new_t = (raw_R @ t + raw_t) / c
    Rt = np.concatenate([new_R.reshape(3, 3), new_t.reshape(3, 1)], axis=1)
    return Rt


def _load_cameras_extrinsics(
    included_cams: list[str],
    colmap_dir: str,
    cfg: dict[str, Any],
) -> CameraExtrinsicsAnnotations:
    coordinate_transform = _load_coordinate_transform(colmap_dir, cfg)

    cam_extrinsics_annotations: list[CameraExtrinsicsAnnotation] = []

    for cam_name in cfg.CALIBRATION.MANUAL_EXO_CAMERAS:
        if cam_name not in included_cams:
            continue

        if cam_name in cfg.INVALID_EXOS:
            continue

        Rt = np.load(os.path.join(colmap_dir, "{}.npy".format(cam_name)))[:3, :]
        Rt = _apply_coordinate_transform(Rt, coordinate_transform)
        Rt = torch.from_numpy(Rt).float()
        cam_extrinsics_annotations.append(
            CameraExtrinsicsAnnotation(
                view_id=cam_name,
                frame_idx=0,
                R=Rt[:3, :3],
                t=Rt[:3, 3],
            )
        )

    extrinsics_calibration_file = os.path.join(colmap_dir, "images.txt")

    with open(extrinsics_calibration_file) as f:
        extrinsics = f.readlines()
        extrinsics = extrinsics[4:]  # drop the first 4 lines (commented lines)
        extrinsics = extrinsics[::2]  # only alternate lines (skip points2d)

    for line in extrinsics:
        line = line.strip().split()

        name = line[9]
        name = str(Path(name).with_suffix(""))
        # Split name by /, first element is the subject name, second is the frame_idx
        cam_name, frame_id = name.split("/")
        frame_id = int(frame_id)

        if (
            cam_name in cfg.CALIBRATION.MANUAL_EXO_CAMERAS
            or cam_name in cfg.INVALID_EXOS
            or cam_name not in included_cams
        ):
            continue

        qw, qx, qy, qz = (
            float(line[1]),
            float(line[2]),
            float(line[3]),
            float(line[4]),
        )
        tx, ty, tz = float(line[5]), float(line[6]), float(line[7])

        R = roma.unitquat_to_rotmat(torch.tensor([qx, qy, qz, qw])).numpy()
        t = np.array([tx, ty, tz]).reshape(3, 1)
        Rt = np.concatenate([R, t], axis=1)
        Rt = _apply_coordinate_transform(Rt, coordinate_transform)
        Rt = torch.from_numpy(Rt).float()

        cam_extrinsics_annotations.append(
            CameraExtrinsicsAnnotation(
                view_id=cam_name,
                frame_idx=frame_id,
                R=Rt[:3, :3],
                t=Rt[:3, 3],
            )
        )

    cam_extrinsics_annotations = CameraExtrinsicsAnnotations(
        metadata=CameraExtrinsicsAnnotationsMetadata(),
        annotations=cam_extrinsics_annotations,
    )

    return cam_extrinsics_annotations


def _extract_data(dataset_dir: str, sequences: list[str] = None):
    for sequence in SEQUENCES:
        sequence_dir = sequence["directory"]
        sequence_name = sequence["name"]
        sequence_dir = os.path.join(dataset_dir, sequence_dir)

        pbar = tqdm(
            glob(os.path.join(sequence_dir, "*.tar.gz")),
            desc=f"Extracting sequence '{sequence_name}'",
            leave=False,
            disable=True,
        )

        for subsequence_tarfile in pbar:
            tarfile_name = os.path.basename(subsequence_tarfile).split(".")[0]
            subsequence_idx = int(tarfile_name.split("_")[0])
            subsequence_name = f"{subsequence_idx:03d}_{sequence_name}"
            subsequence_dir = os.path.join(sequence_dir, subsequence_name)

            if sequences is not None and subsequence_name not in sequences:
                continue

            pbar.disable = False

            if os.path.exists(subsequence_dir):
                tqdm.write(
                    f"Skipping subsequence '{tarfile_name}' because folder already exists"
                )
                continue

            os.makedirs(subsequence_dir, exist_ok=True)

            try:
                extract_tar_with_progress(
                    tar=subsequence_tarfile,
                    output_dir=sequence_dir,
                    strip_components=8,
                )
            except Exception as e:
                warnings.warn(f"Error extracting subsequence '{tarfile_name}': {e}")
                os.remove(subsequence_tarfile)
                continue

import torch
import orjson
import os
import cv2
import glob

from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import ImagesLoader
from kineo.visualization.viz_2d import (
    draw_bboxes,
    draw_keypoints,
    get_subject_color_rgba,
)


def load_gt_keypoints_and_bboxes_2d(
    dataset_dir: str,
    sequence: dict,
) -> tuple[Keypoints2DAnnotations, BBox2DAnnotations]:
    keypoints_2d_file = sequence["annotations"]["keypoints_2d"]
    keypoints_2d_file = os.path.join(dataset_dir, keypoints_2d_file)

    with open(keypoints_2d_file, "rb") as f:
        gt_keypoints_2d = Keypoints2DAnnotations.from_dict(orjson.loads(f.read()))

    bboxes_2d_file = sequence["annotations"]["bboxes_2d"]
    bboxes_2d_file = os.path.join(dataset_dir, bboxes_2d_file)

    with open(bboxes_2d_file, "rb") as f:
        gt_bboxes_2d = BBox2DAnnotations.from_dict(orjson.loads(f.read()))

    return gt_keypoints_2d, gt_bboxes_2d


def create_view_inputs(
    dataset_dir: str,
    sequence: dict,
    device: torch.device,
) -> list[ViewInput]:
    cameras = list(sequence["views"].keys())
    views = []
    for camera in cameras:
        images_dir = sequence["views"][camera]["images_dir"]
        fps = sequence["views"][camera]["fps"]
        imgs_paths = sorted(
            glob.glob(
                os.path.join(
                    dataset_dir,
                    images_dir,
                    "*.jpg",
                )
            )
        )
        n_imgs = len(imgs_paths)

        frame_timestamps_local = (torch.arange(n_imgs) / fps).tolist()

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

    return views


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "egohumans_dataset_dir", type=str, help="Path to dataset directory"
    )
    parser.add_argument(
        "egohumans_pred_annotations_dir",
        type=str,
        help="Path to predicted annotations directory",
    )
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    dataset_dir = args.egohumans_dataset_dir
    pred_annotations_dir = args.egohumans_pred_annotations_dir
    output_dir = args.output_dir

    sequences_file = os.path.join(dataset_dir, "egohumans_sequences.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    sequence = next(s for s in sequences if s["sequence_name"] == "legoassemble_001")
    sequence_name = sequence["sequence_name"]

    selected_frames = [0, 300, 600]
    selected_views = ["cam01", "cam05", "cam08"]

    views_inputs = create_view_inputs(dataset_dir, sequence, device)

    gt_kps_2d, gt_bboxes_2d = load_gt_keypoints_and_bboxes_2d(
        dataset_dir=dataset_dir,
        sequence=sequence,
    )

    kps_format = gt_kps_2d.metadata.formats[0]

    for view_input in views_inputs:
        view_id = view_input["view_id"]
        frame_loader = view_input["frame_loader"]
        n_frames = frame_loader.n_frames

        if view_id not in selected_views:
            continue

        for i, frame_idx in enumerate(selected_frames):
            frame = frame_loader.load_frame_at(frame_idx)
            frame_rgb = frame.permute(1, 2, 0).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr_annotated = frame_bgr.copy()

            gt_kps_2d_view = gt_kps_2d.filter_by_view_id(view_id).filter_by_frame_idx(
                frame_idx
            )
            gt_bboxes_2d_view = gt_bboxes_2d.filter_by_view_id(
                view_id
            ).filter_by_frame_idx(frame_idx)

            subjects_ids = gt_kps_2d_view.subjects_ids

            for subject_id in subjects_ids:
                subject_color_rgba = get_subject_color_rgba(subject_id)
                subject_color_rgb = tuple(int(x * 255) for x in subject_color_rgba[:3])
                subject_color_bgr = tuple(reversed(subject_color_rgb))

                subject_kps_2d_view = gt_kps_2d_view.filter_by_subject_id(
                    subject_id
                ).first_or_default()
                subject_bboxes_2d_view = gt_bboxes_2d_view.filter_by_subject_id(
                    subject_id
                ).first_or_default()

                if subject_bboxes_2d_view is not None and subject_kps_2d_view is not None:
                    frame_bgr_annotated = draw_bboxes(
                        frame=frame_bgr_annotated,
                        bboxes=subject_bboxes_2d_view.xyxy.cpu().numpy(),
                        colors_bgr=subject_color_bgr,
                        border_width=20,
                    )

                    frame_bgr_annotated = draw_keypoints(
                        frame=frame_bgr_annotated,
                        keypoints=subject_kps_2d_view.xy.cpu().numpy(),
                        colors_bgr=subject_color_bgr,
                        connectivity=kps_format.keypoints_connectivity,
                        connectivity_colors_bgr=subject_color_bgr,
                        keypoints_radius=20,
                        keypoints_border_width=10,
                        bone_width=20
                    )

            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_dir, f"{view_id}_{i}_annotated.jpg"),
                frame_bgr_annotated,
            )
            cv2.imwrite(
                os.path.join(output_dir, f"{view_id}_{i}.jpg"),
                frame_bgr,
            )
            print(f"Saved frame {frame_idx} for view {view_id}")

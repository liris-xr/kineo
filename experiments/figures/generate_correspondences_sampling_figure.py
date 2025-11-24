import torch
import orjson
import os
import cv2
from tqdm import tqdm
import pickle
import matplotlib
import numpy as np

matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.calibration_points_pairs import CalibrationPointsAnnotations
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader


def load_kps_2d_annotations(
    pred_annotations_dir: str,
    sequence: dict,
) -> Keypoints2DAnnotations:
    kps_2d_file = os.path.join(
        pred_annotations_dir,
        sequence["sequence_name"],
        "keypoints_2d.pkl",
    )
    with open(kps_2d_file, "rb") as f:
        kps_2d = Keypoints2DAnnotations.from_dict(pickle.load(f))
    return kps_2d


def load_calibration_points_annotations(
    pred_annotations_dir: str,
    sequence: dict,
) -> CalibrationPointsAnnotations:
    calibration_points_file = os.path.join(
        pred_annotations_dir,
        sequence["sequence_name"],
        "calibration_points.pkl",
    )

    with open(calibration_points_file, "rb") as f:
        calibration_points = CalibrationPointsAnnotations.from_dict(pickle.load(f))
    return calibration_points


def create_view_inputs(
    dataset_dir: str,
    sequence: dict,
    device: torch.device,
) -> list[ViewInput]:
    cameras = list(sequence["views"].keys())
    views = []
    for camera in cameras:
        video_path = os.path.join(dataset_dir, sequence["views"][camera]["video_path"])
        selected_frames = sequence["views"][camera]["selected_frames"]
        selected_frames_start = int(selected_frames["start"])
        selected_frames_stop = int(selected_frames["stop"])
        selected_frames_step = int(selected_frames["step"])
        selected_frames = range(
            selected_frames_start, selected_frames_stop, selected_frames_step
        )

        views.append(
            ViewInput(
                view_id=camera,
                frame_loader=VideoLoader(
                    video_path=video_path,
                    selected_frames=selected_frames,
                    device=device,
                ),
                audio_loader=None,
            )
        )

    return views


def draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    lineType: int = cv2.LINE_AA,
    dash_length: float = 10,
) -> None:
    # Convert points to numpy arrays
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)

    # Compute the Euclidean distance
    dist = np.linalg.norm(pt2 - pt1)

    # Compute unit vector in direction of the line
    vector = (pt2 - pt1) / dist

    # Loop and draw dashes
    for i in np.arange(0, dist, dash_length * 2):
        start = pt1 + i * vector
        end = pt1 + (i + dash_length) * vector
        if np.linalg.norm(end - pt1) > dist:
            end = pt2
        cv2.line(
            img,
            pt1=tuple(start.astype(int)),
            pt2=tuple(end.astype(int)),
            color=color,
            thickness=thickness,
            lineType=lineType,
        )


def draw_matches(
    view1_frame_bgr_np: np.ndarray,
    view2_frame_bgr_np: np.ndarray,
    matches: np.ndarray,
    score_thr: float = 0.75,
    match_point_color: tuple[int, int, int] | None = None,
    match_line_color: tuple[int, int, int] | None = None,
    match_line_thickness: int = 2,
    match_point_radius: int = 5,
    match_point_thickness: int | None = None,
) -> np.ndarray:
    """
    Draw matches between two views.

    Args:
        view1_frame_rgb: RGB image of the first view.
        view2_frame_rgb: RGB image of the second view.
        matches: Tensor of shape (n_matches, 5) containing the matches (x1, y1, x2, y2, score).
    """
    min_height = min(view1_frame_bgr_np.shape[0], view2_frame_bgr_np.shape[0])
    dy_view1 = view1_frame_bgr_np.shape[0] - min_height
    dy_view2 = view2_frame_bgr_np.shape[0] - min_height

    if dy_view1 > 0:
        view1_frame_bgr_np = view1_frame_bgr_np[:-dy_view1, :, :]
    if dy_view2 > 0:
        view2_frame_bgr_np = view2_frame_bgr_np[:-dy_view2, :, :]

    n_matches = matches.shape[0]
    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2] + view1_frame_bgr_np.shape[1]
    y2 = matches[:, 3]
    scores = matches[:, 4]

    combined_img = np.concatenate([view1_frame_bgr_np, view2_frame_bgr_np], axis=1)

    for i in range(n_matches):
        if match_line_color is None:
            match_line_color_i = tuple(
                cv2.applyColorMap(
                    src=np.array([scores[i] * 255], dtype=np.uint8),
                    colormap=cv2.COLORMAP_VIRIDIS,
                ).reshape(3)
            )
            match_line_color_i = tuple(map(int, match_line_color_i))
        else:
            match_line_color_i = match_line_color

        if match_point_color is None:
            match_point_color_i = tuple(
                cv2.applyColorMap(
                    src=np.array([scores[i] * 255], dtype=np.uint8),
                    colormap=cv2.COLORMAP_VIRIDIS,
                ).reshape(3)
            )
            match_point_color_i = tuple(map(int, match_point_color_i))
        else:
            match_point_color_i = match_point_color

        cv2.circle(
            img=combined_img,
            center=(int(x1[i]), int(y1[i])),
            radius=match_point_radius,
            color=match_point_color_i,
            lineType=cv2.LINE_AA,
            thickness=(-1 if scores[i] > score_thr else 1)
            if not match_point_thickness
            else match_point_thickness,
        )
        cv2.circle(
            img=combined_img,
            center=(int(x2[i]), int(y2[i])),
            radius=match_point_radius,
            color=match_point_color_i,
            lineType=cv2.LINE_AA,
            thickness=(-1 if scores[i] > score_thr else 1)
            if not match_point_thickness
            else match_point_thickness,
        )
        if scores[i] > score_thr:
            cv2.line(
                img=combined_img,
                pt1=(int(x1[i]), int(y1[i])),
                pt2=(int(x2[i]), int(y2[i])),
                color=match_line_color_i,
                thickness=match_line_thickness,
                lineType=cv2.LINE_AA,
            )
        else:
            draw_dashed_line(
                img=combined_img,
                pt1=(int(x1[i]), int(y1[i])),
                pt2=(int(x2[i]), int(y2[i])),
                color=match_line_color_i,
                thickness=match_line_thickness,
                lineType=cv2.LINE_AA,
                dash_length=10,
            )

    return combined_img


def export_single_frame_matches(
    dataset_dir: str,
    pred_annotations_dir: str,
    sequence: dict,
    frame_idx: int,
    view1_id: str,
    view2_id: str,
    output_dir: str,
):
    sequence_name = sequence["sequence_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    views_inputs = create_view_inputs(dataset_dir, sequence, device)
    view_id_to_idx = {
        view_input["view_id"]: i for i, view_input in enumerate(views_inputs)
    }

    kps_2d = load_kps_2d_annotations(pred_annotations_dir, sequence)

    view1 = views_inputs[view_id_to_idx[view1_id]]
    view2 = views_inputs[view_id_to_idx[view2_id]]

    view1_frame_rgb = view1["frame_loader"].load_frame_at(frame_idx)
    view2_frame_rgb = view2["frame_loader"].load_frame_at(frame_idx)
    view1_id = view1["view_id"]
    view2_id = view2["view_id"]

    kps_view1 = (
        kps_2d.filter_by_view_id(view1_id)
        .filter_by_frame_idx(frame_idx)
        .first_or_default()
    )
    kps_view1_xy = kps_view1.xy
    kps_view1_scores = kps_view1.scores

    kps_view2 = (
        kps_2d.filter_by_view_id(view2_id)
        .filter_by_frame_idx(frame_idx)
        .first_or_default()
    )
    kps_view2_xy = kps_view2.xy
    kps_view2_scores = kps_view2.scores

    match_score = torch.sqrt(kps_view1_scores * kps_view2_scores)

    view1_frame_bgr_np = (
        view1_frame_rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
    )
    view2_frame_bgr_np = (
        view2_frame_rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
    )

    matches = torch.cat([kps_view1_xy, kps_view2_xy, match_score.unsqueeze(-1)], dim=-1)

    matches_img = draw_matches(
        view1_frame_bgr_np=view1_frame_bgr_np,
        view2_frame_bgr_np=view2_frame_bgr_np,
        matches=matches.cpu().numpy(),
        score_thr=0.75,
    )
    # Crop image
    matches_img = matches_img[
        131:680,
        563:1669,
    ]

    _, ax = plt.subplots(figsize=(6, 4))
    _ = ax.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
        label="Confidence Score",
    )

    plt.savefig(
        os.path.join(
            output_dir,
            f"single_frame_matches_{sequence_name}_{view1_id}_{view2_id}_{frame_idx}.pdf",
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def avg_frame(view: ViewInput) -> torch.Tensor:
    h, w = view["frame_loader"].resolution_hw
    frame_avg = torch.zeros((3, h, w), device=view["frame_loader"].device)
    frame_loader = view["frame_loader"]

    for frame_idx in tqdm(
        range(frame_loader.n_frames),
        desc="Averaging frames",
        leave=False,
    ):
        frame_avg += frame_loader.load_frame_at(frame_idx) / 255.0

    frame_avg /= frame_loader.n_frames
    frame_avg = (frame_avg * 255.0).to(torch.uint8)
    return frame_avg


def export_all_matches(
    dataset_dir: str,
    pred_annotations_dir: str,
    sequence: dict,
    view1_id: str,
    view2_id: str,
    output_dir: str,
):
    sequence_name = sequence["sequence_name"]

    views_inputs = create_view_inputs(dataset_dir, sequence, device)
    view_id_to_idx = {
        view_input["view_id"]: i for i, view_input in enumerate(views_inputs)
    }

    calibration_points = load_calibration_points_annotations(
        pred_annotations_dir, sequence
    )

    view1 = views_inputs[view_id_to_idx[view1_id]]
    view2 = views_inputs[view_id_to_idx[view2_id]]

    view1_avg_frame = avg_frame(view1)
    view2_avg_frame = avg_frame(view2)

    calib_ann = next(
        a
        for a in calibration_points
        if (a.view1_id == view1_id and a.view2_id == view2_id)
    )

    view1_avg_frame_bgr_np = (
        view1_avg_frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
    )
    view2_avg_frame_bgr_np = (
        view2_avg_frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
    )

    matches = torch.cat(
        [
            calib_ann.points1,
            calib_ann.points2,
            torch.ones((calib_ann.points1.shape[0], 1)),
        ],
        dim=-1,
    ).float()

    matches_img = draw_matches(
        view1_frame_bgr_np=view1_avg_frame_bgr_np,
        view2_frame_bgr_np=view2_avg_frame_bgr_np,
        matches=matches.cpu().numpy(),
        match_line_color=(186, 220, 88)[::-1],
        match_line_thickness=1,
        match_point_color=(106, 176, 76)[::-1],
        match_point_thickness=1,
    )

    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.savefig(
        os.path.join(
            output_dir, f"all_matches_{sequence_name}_{view1_id}_{view2_id}.pdf"
        ),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("h36m_dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument(
        "h36m_pred_annotations_dir",
        type=str,
        help="Path to predicted annotations directory",
    )
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    dataset_dir = args.h36m_dataset_dir
    pred_annotations_dir = args.h36m_pred_annotations_dir
    output_dir = args.output_dir

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    sequences = [s for s in sequences if s["split"] == "val"]

    export_single_frame_matches(
        dataset_dir=dataset_dir,
        pred_annotations_dir=pred_annotations_dir,
        sequence=sequences[0],
        frame_idx=50,
        view1_id="60457274",
        view2_id="55011271",
        output_dir=output_dir,
    )

    export_all_matches(
        dataset_dir=dataset_dir,
        pred_annotations_dir=pred_annotations_dir,
        sequence=next(s for s in sequences if s["sequence_name"] == "S9_Walking 1"),
        view1_id="60457274",
        view2_id="55011271",
        output_dir=output_dir,
    )

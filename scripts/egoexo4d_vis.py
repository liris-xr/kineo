import orjson
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from PIL import ImageDraw
from tqdm import tqdm

palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)


def get_body_metadata():
    keypoints_map = [
        {"label": "Nose", "id": "fee3cbd2", "color": "#f77189"},
        {"label": "Left-eye", "id": "ab12de34", "color": "#d58c32"},
        {"label": "Right-eye", "id": "7f2g1h6k", "color": "#a4a031"},
        {"label": "Left-ear", "id": "mn0pqrst", "color": "#50b131"},
        {"label": "Right-ear", "id": "yz89wx76", "color": "#34ae91"},
        {"label": "Left-shoulder", "id": "5a4b3c2d", "color": "#37abb5"},
        {"label": "Right-shoulder", "id": "e1f2g3h4", "color": "#3ba3ec"},
        {"label": "Left-elbow", "id": "6i7j8k9l", "color": "#bb83f4"},
        {"label": "Right-elbow", "id": "uv0wxy12", "color": "#f564d4"},
        {"label": "Left-wrist", "id": "3z4ab5cd", "color": "#2fd4aa"},
        {"label": "Right-wrist", "id": "efgh6789", "color": "#94d14f"},
        {"label": "Left-hip", "id": "ijklmnop", "color": "#b3d32c"},
        {"label": "Right-hip", "id": "qrstuvwx", "color": "#f9b530"},
        {"label": "Left-knee", "id": "yz012345", "color": "#83f483"},
        {"label": "Right-knee", "id": "6bc7defg", "color": "#32d58c"},
        {"label": "Left-ankle", "id": "hijk8lmn", "color": "#3ba3ec"},
        {"label": "Right-ankle", "id": "opqrs1tu", "color": "#f564d4"},
    ]

    # pyre-ignore
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]
    return keypoints_map, skeleton, pose_kpt_color


def _load_body_annotations(
    dataset_dirpath: str, take: dict[str, Any]
) -> dict[str, Any]:
    body_annotations_filepath = os.path.join(
        dataset_dirpath,
        "annotations",
        "ego_pose",
        "val",
        "body",
        "annotation",
        f"{take['take_uid']}.json",
    )

    with open(body_annotations_filepath, "rb") as f:
        body_annotations = orjson.loads(f.read())
    return body_annotations


def _load_gopro_calibs(dataset_dirpath: str) -> dict[str, Any]:
    camera_calib_filepath = os.path.join(
        dataset_dirpath, "takes", take["take_name"], "trajectory", "gopro_calibs.csv"
    )
    with open(camera_calib_filepath, "r") as f:
        df = pd.read_csv(f)
    data = df.to_dict(orient="records")
    return {row["cam_uid"]: row for row in data}


def _load_camera_annotations(
    dataset_dirpath: str, take: dict[str, Any]
) -> dict[str, Any]:
    camera_annotations_filepath = os.path.join(
        dataset_dirpath,
        "annotations",
        "ego_pose",
        "val",
        "camera_pose",
        f"{take['take_uid']}.json",
    )
    with open(camera_annotations_filepath, "rb") as f:
        camera_annotations = orjson.loads(f.read())
    del camera_annotations["metadata"]
    return camera_annotations


def get_coords(annot):
    pts = dict()
    for k in annot:
        atype = 1
        if annot[k]["placement"] == "auto":
            atype = 0
        pts[k] = [annot[k]["x"], annot[k]["y"], atype]
    return pts


def draw_skeleton(img, all_pts, skeleton, thickness=5):
    for item in skeleton:
        left_index = item[0] - 1
        right_index = item[1] - 1
        left_pt = all_pts[left_index]
        right_pt = all_pts[right_index]
        if len(left_pt) == 0 or len(right_pt) == 0:
            continue
        left_pt = (int(left_pt[0]), int(left_pt[1]))
        right_pt = (int(right_pt[0]), int(right_pt[1]))
        cv2.line(img, left_pt, right_pt, (255, 255, 255), thickness=thickness)


def draw_cross(img, x, y, color, thickness=3):
    # Circle parameters
    center = (int(x), int(y))  # Center of the cross
    cross_length = 10  # Half-length of the cross arms
    # Calculate the end points of the cross
    left_point = (center[0] - cross_length, center[1])
    right_point = (center[0] + cross_length, center[1])
    top_point = (center[0], center[1] - cross_length)
    bottom_point = (center[0], center[1] + cross_length)

    # Draw the horizontal line
    cv2.line(img, left_point, right_point, color, thickness=thickness)
    # Draw the vertical line
    cv2.line(img, top_point, bottom_point, color, thickness=thickness)


def draw_circle(img, x, y, color, radius=4, thickness=2):
    # Circle parameters
    center = (int(x), int(y))  # Center of the circle
    # Draw the circle with a black outline
    cv2.circle(img, center, radius, color, thickness=thickness)


def draw_label(img, x, y, color, label, fontScale=0.5, thickness=1):
    # Circle parameters
    center = (int(x + 20), int(y - 20))  # Center of the circle
    cv2.putText(
        img,
        label,
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
    )


if __name__ == "__main__":

    dataset_dirpath = "F:/Datasets/EgoExo4D"
    takes_filepath = os.path.join(dataset_dirpath, "takes.json")

    with open(takes_filepath, "rb") as f:
        takes = orjson.loads(f.read())

    for take in takes:
        name = take["take_name"]
        uid = take["take_uid"]
        root_dir = take["root_dir"]

        if name != "cmu_soccer06_3":
            continue

        videos_dir = os.path.join(dataset_dirpath, root_dir, "frame_aligned_videos")

        body_annotations = _load_body_annotations(dataset_dirpath, take)
        camera_annotations = _load_camera_annotations(dataset_dirpath, take)
        gopro_calibs = _load_gopro_calibs(dataset_dirpath)

        cameras_data = {}

        for camera_name, camera_annotation in camera_annotations.items():

            if camera_name not in gopro_calibs:
                continue

            gopro_calib = gopro_calibs[camera_name]
            K = np.array(
                [
                    [gopro_calib["intrinsics_0"], 0, gopro_calib["intrinsics_2"]],
                    [0, gopro_calib["intrinsics_1"], gopro_calib["intrinsics_3"]],
                    [0, 0, 1],
                ]
            )

            video_filepath = os.path.join(videos_dir, f"{camera_name}.mp4")

            cameras_data[camera_name] = {
                "K": K,
                "new_K": np.asarray(camera_annotation["camera_intrinsics"]),
                "Rt": np.asarray(camera_annotation["camera_extrinsics"]),
                "dist_coeffs": np.asarray(camera_annotation["distortion_coeffs"]),
                "cap": cv2.VideoCapture(video_filepath),
            }

        for camera_name, camera_data in cameras_data.items():
            cap: cv2.VideoCapture = camera_data["cap"]

            writer = cv2.VideoWriter(
                f"{take['take_name']}_{camera_name}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )

            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            keypoints_map, skeleton, pose_kpt_color = get_body_metadata()

            start_frame = 1423
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in tqdm(range(start_frame, n_frames)):
                ret, frame = cap.read()

                frame_idx_str = str(i)

                if frame_idx_str in body_annotations:

                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                        camera_data["K"],
                        camera_data["dist_coeffs"],
                        np.eye(3),
                        camera_data["new_K"],
                        (frame.shape[1], frame.shape[0]),
                        cv2.CV_16SC2,
                    )
                    frame = cv2.remap(
                        frame,
                        map1,
                        map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )

                    ann = body_annotations[frame_idx_str][0]["annotation2D"][camera_name]
                    pts = get_coords(ann)

                    all_pts = []
                    for _, keypoints in enumerate(keypoints_map):
                        kpname = keypoints["label"].lower()

                        if kpname in pts:
                            x, y = pts[kpname][0], pts[kpname][1]
                            all_pts.append((x, y))
                        else:
                            all_pts.append(())

                    draw_skeleton(frame, all_pts, skeleton)

                    for index, keypoints in enumerate(keypoints_map):
                        kpname = keypoints["label"].lower()
                        if kpname in pts:
                            x, y, pt_type = pts[kpname][0], pts[kpname][1], pts[kpname][2]
                            color = tuple(pose_kpt_color[index])
                            color = (int(color[0]), int(color[1]), int(color[2]))

                            if pt_type == 1:
                                draw_circle(frame, x, y, color)
                            else:
                                draw_cross(frame, x, y, color)
                            draw_label(frame, x, y, color, kpname)

                    writer.write(frame)
                    # plt.imshow(frame)
                    # plt.axis("off")
                    # plt.show()

            cap.release()
            writer.release()

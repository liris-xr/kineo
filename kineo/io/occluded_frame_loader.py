# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import xml.etree.ElementTree
import zlib
from typing import Sequence

import cv2
import numpy as np
import PIL.Image
import torch

from kineo.io.frame_sequence_loader import FrameSequenceLoader

# Port of https://github.com/laurabragagnolo/human3.6m-occluded
N_OCCLUDERS_PER_VIEW = 2


def load_occluders(pascal_voc_dir: str) -> list[np.ndarray]:
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    annotations_dir = os.path.join(pascal_voc_dir, "Annotations")
    for annotation_name in sorted(os.listdir(annotations_dir)):
        xml_root = xml.etree.ElementTree.parse(
            os.path.join(annotations_dir, annotation_name)
        ).getroot()
        if xml_root.find("segmented").text == "0":
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall("object")):
            is_person = obj.find("name").text == "person"
            is_difficult = obj.find("difficult").text != "0"
            is_truncated = obj.find("truncated").text != "0"
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find("bndbox")
                box = [
                    int(bndbox.find(s).text) for s in ("xmin", "ymin", "xmax", "ymax")
                ]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        image_name = xml_root.find("filename").text
        image = np.asarray(
            PIL.Image.open(os.path.join(pascal_voc_dir, "JPEGImages", image_name))
        )
        labels = np.asarray(
            PIL.Image.open(
                os.path.join(
                    pascal_voc_dir,
                    "SegmentationObject",
                    image_name.replace("jpg", "png"),
                )
            )
        )

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(
                np.uint8
            ) * 255
            if cv2.countNonZero(object_mask) < 500:
                continue

            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            # No BGR swap as frames are already RGB
            object_with_mask = np.concatenate(
                [image[ymin:ymax, xmin:xmax], object_mask[..., np.newaxis]], axis=-1
            )
            occluders.append(_resize_by_factor(object_with_mask, 0.5))

    return occluders


def occlude_with_objects(
    image: np.ndarray,
    occluders: Sequence[np.ndarray],
    bbox_xywh: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    result = image.copy()
    x, y, width, height = bbox_xywh
    person_min_side = min(width, height)

    for _ in range(N_OCCLUDERS_PER_VIEW):
        occluder = occluders[rng.integers(len(occluders))]
        random_scale_factor = rng.uniform(0.5, 1.0)
        occluder_min_side = min(occluder.shape[1], occluder.shape[0])
        occluder = _resize_by_factor(
            occluder, random_scale_factor * person_min_side / occluder_min_side
        )
        center = (rng.uniform(x, x + width), rng.uniform(y, y + height))
        _paste_over(occluder, result, center)

    return result


def _paste_over(
    im_src: np.ndarray, im_dst: np.ndarray, center: tuple[float, float]
) -> None:
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1] : end_dst[1], start_dst[0] : end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1] : end_src[1], start_src[0] : end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    im_dst[start_dst[1] : end_dst[1], start_dst[0] : end_dst[0]] = (
        alpha * color_src + (1 - alpha) * region_dst
    )


def _resize_by_factor(image: np.ndarray, factor: float) -> np.ndarray:
    new_size = tuple(
        np.round(np.array([image.shape[1], image.shape[0]]) * factor).astype(int)
    )
    interpolation = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, new_size, interpolation=interpolation)


class OccludedFrameLoader(FrameSequenceLoader):
    """Loader occluding the person of a view where a box is known."""

    def __init__(
        self,
        loader: FrameSequenceLoader,
        occluders: Sequence[np.ndarray],
        bboxes_xywh: dict[int, Sequence[float]],
        sequence_name: str,
        view_idx: int,
        n_views: int,
        n_occluded_views: int,
        seed: int,
    ):
        super().__init__(loader.device)
        self._loader = loader
        self._occluders = occluders
        self._bboxes_xywh = bboxes_xywh
        self._sequence_key = zlib.crc32(sequence_name.encode("utf-8"))
        self._view_idx = view_idx
        self._n_views = n_views
        self._n_occluded_views = n_occluded_views
        self._seed = seed

    def _occlude(self, frame_index: int, frame: torch.Tensor) -> torch.Tensor:
        bbox = self._bboxes_xywh.get(frame_index)
        if bbox is None:
            return frame

        frame_key = [self._seed, self._sequence_key, frame_index]
        occluded_views = np.random.default_rng(frame_key).choice(
            self._n_views, self._n_occluded_views, replace=False
        )
        if self._view_idx not in occluded_views:
            return frame

        image = frame.permute(1, 2, 0).cpu().numpy()
        occluded = occlude_with_objects(
            image,
            self._occluders,
            bbox,
            np.random.default_rng(frame_key + [self._view_idx]),
        )
        return torch.from_numpy(occluded).permute(2, 0, 1).to(frame.device)

    def load_frame_at(self, frame_index: int) -> torch.Tensor:
        return self._occlude(frame_index, self._loader.load_frame_at(frame_index))

    def load_frames_at(
        self, frame_indices: Sequence | range | slice | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(frame_indices, slice):
            frame_indices = range(*frame_indices.indices(self.n_frames))
        frame_indices = [int(i) for i in torch.as_tensor(frame_indices).tolist()]
        frames = self._loader.load_frames_at(frame_indices)
        return torch.stack(
            [self._occlude(i, frame) for i, frame in zip(frame_indices, frames)]
        )

    @property
    def frame_timestamps_local(self) -> torch.Tensor:
        return self._loader.frame_timestamps_local

    @property
    def resolution_hw(self) -> tuple[int, int]:
        return self._loader.resolution_hw

    @property
    def n_frames(self) -> int:
        return self._loader.n_frames

    def close(self):
        self._loader.close()

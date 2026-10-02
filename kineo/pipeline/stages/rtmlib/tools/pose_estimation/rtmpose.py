from typing import List, Tuple

import numpy as np

import torch

from . import torch_processings
from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine



class RTMPose(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 use_tensorrt: bool = False,
                 fp16: bool = False,
                 engine_cache_dir: str = None,
                 strict_providers: bool = False):
        super().__init__(onnx_model,
                         model_input_size,
                         mean,
                         std,
                         backend,
                         device,
                         use_tensorrt=use_tensorrt,
                         fp16=fp16,
                         engine_cache_dir=engine_cache_dir,
                         strict_providers=strict_providers)
        self.to_openpose = to_openpose
        # ONNX output shapes are symbolic, so learn them once
        self._simcc_widths = None

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img.copy() - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores

    def forward_gpu_batch(
        self, frames: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> list[tuple]:
        input_size_wh = tuple(self.model_input_size)
        device = frames[0].device

        mean = std = None
        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
            std = torch.as_tensor(self.std, dtype=torch.float32, device=device)

        crops = []
        centers = []
        scales = []
        counts = []
        for frame, frame_boxes in zip(frames, boxes):
            if frame_boxes.numel() == 0:
                counts.append(0)
                continue
            view_centers, view_scales = torch_processings.bboxes_to_center_scale(
                frame_boxes, input_size_wh
            )
            crops.append(
                torch_processings.crop_and_normalize(
                    frame, view_centers, view_scales, input_size_wh, mean, std
                )
            )
            centers.append(view_centers)
            scales.append(view_scales)
            counts.append(view_centers.shape[0])

        if not crops:
            return [
                (torch.zeros((0, 0, 2), device=device), torch.zeros((0, 0), device=device))
                for _ in frames
            ]

        batch = torch.cat(crops, dim=0).contiguous()
        all_centers = torch.cat(centers, dim=0)
        all_scales = torch.cat(scales, dim=0)

        output_shapes = None
        if self._simcc_widths is not None:
            n_keypoints, width_x, width_y = self._simcc_widths
            output_shapes = [
                (batch.shape[0], n_keypoints, width_x),
                (batch.shape[0], n_keypoints, width_y),
            ]
        simcc_x, simcc_y = self.inference_gpu(batch, output_shapes)
        if self._simcc_widths is None:
            self._simcc_widths = (
                simcc_x.shape[1],
                simcc_x.shape[2],
                simcc_y.shape[2],
            )

        keypoints, scores = torch_processings.decode_simcc(
            simcc_x, simcc_y, all_centers, all_scales, input_size_wh
        )

        results = []
        offset = 0
        for count in counts:
            if count == 0:
                results.append(
                    (
                        torch.zeros((0, keypoints.shape[1], 2), device=device),
                        torch.zeros((0, keypoints.shape[1]), device=device),
                    )
                )
                continue
            results.append(
                (keypoints[offset : offset + count], scores[offset : offset + count])
            )
            offset += count

        return results

# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2
from typing import Callable
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
import torch

from sam2.sam2_generic_video_predictor import (
    SAM2GenericVideoPredictor,
    SAM2GenericVideoPredictorState,
)
from sam2.modeling.sam2_prompt import SAM2Prompt

from kineo.io.frame_sequence_loader import FrameSequenceLoader

@dataclass
class Frame:
    frame_idx: int
    img: torch.Tensor
    img_embeddings: list[torch.Tensor]
    img_pos_embeddings: list[torch.Tensor]


class SAMVideoWidget:
    def __init__(
        self,
        frame_sequence_loader: FrameSequenceLoader,
        sam2_video_predictor: SAM2GenericVideoPredictor,
        sam2_video_predictor_state: SAM2GenericVideoPredictorState,
        continue_callback: Callable[[SAMVideoWidget], None],
        bbox_prompt_min_area_px: int = 25,
    ):
        self._continue_callback = continue_callback
        self._reader = frame_sequence_loader

        self._video_width = self._reader.resolution_hw[1]
        self._video_height = self._reader.resolution_hw[0]
        self._total_frames = self._reader.n_frames

        self._sam2_video_predictor = sam2_video_predictor
        self._sam2_video_predictor_state = sam2_video_predictor_state

        self._current_obj_id = 0
        self._point_distance_threshold_px = 100

        # Clicks associated to each frame.
        self._clicks: dict[int, list[Click]] = defaultdict(list)
        self._bboxes: dict[int, list[BoundingBox]] = defaultdict(list)

        self._press_x = None
        self._press_y = None
        self._last_valid_x = None
        self._last_valid_y = None
        self._is_pressing = False
        self._is_dragging = False

        self._bbox_x1 = None
        self._bbox_y1 = None
        self._bbox_x2 = None
        self._bbox_y2 = None

        self._is_alt_pressed = False
        self._is_ctrl_pressed = False
        self._is_shift_pressed = False

        self._bbox_prompt_min_area_px = bbox_prompt_min_area_px

        self._current_frame_idx = 0
        self._current_frame_cache: Frame | None = None

        self._fig = plt.figure()
        gs = gridspec.GridSpec(
            3,
            3,
            height_ratios=[9, 1, 1],
            width_ratios=[8, 1, 1],
            hspace=0.1,
            wspace=0.1,
        )
        self._ax_image = self._fig.add_subplot(gs[0, :])
        self._ax_image.axis("off")
        self._image = self._ax_image.imshow(
            self.current_frame.img.permute(1, 2, 0).cpu().numpy()
        )
        self._mask_overlay = self._ax_image.imshow(
            np.zeros((self._video_height, self._video_width, 4), dtype=np.uint8)
        )

        self._ax_instruction_text = self._ax_image.text(
            0, 1,
            "Click: add positive point\nAlt+Click: add negative point\nShift+Click: remove point/bbox\nDrag: add bbox",
            transform=self._ax_image.transAxes,
            fontsize=8,
            color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
            verticalalignment='top',
            horizontalalignment='left'
        )

        self._bbox_rect = self._ax_image.add_patch(
            plt.Rectangle(
                (0, 0),
                0,
                0,
                fill=True,
                facecolor=(52 / 255, 152 / 255, 219 / 255, 0.2),
                edgecolor=(52 / 255, 152 / 255, 219 / 255, 1),
                linewidth=1,
            )
        )

        self._ax_slider = self._fig.add_subplot(gs[1, 0])
        self._slider = Slider(
            self._ax_slider, "Frame", 0, self._total_frames - 1, valinit=0, valstep=1
        )
        self._slider.on_changed(self._on_slider_change)
        self._slider.label.set_visible(False)
        self._slider.valtext.set_visible(False)

        self._ax_prev = self._fig.add_subplot(gs[1, 1])
        self._btn_prev = Button(self._ax_prev, "<")
        self._btn_prev.on_clicked(self._on_prev_button)

        self._ax_next = self._fig.add_subplot(gs[1, 2])
        self._btn_next = Button(self._ax_next, ">")
        self._btn_next.on_clicked(self._on_next_button)

        self._ax_continue = self._fig.add_subplot(gs[2, :])
        self._btn_continue = Button(self._ax_continue, "Continue")
        self._btn_continue.on_clicked(self._on_continue)

        self._button_release_cid = self._fig.canvas.mpl_connect(
            "button_release_event", self._on_mouse_release
        )
        self._motion_notify_cid = self._fig.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_move
        )
        self._button_press_cid = self._fig.canvas.mpl_connect(
            "button_press_event", self._on_mouse_press
        )
        self._key_press_cid = self._fig.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )
        self._key_release_cid = self._fig.canvas.mpl_connect(
            "key_release_event", self._on_key_release
        )

    def set_title(self, title: str):
        self._fig.canvas.manager.set_window_title(title)

    def close(self):
        plt.figure(self._fig.number)
        plt.close(self._fig)

    def _jump_to_frame(self, frame_idx: int):
        if frame_idx != self._current_frame_idx:
            for bbox in self._bboxes[self._current_frame_idx]:
                bbox.ui_rect.set_visible(False)
            for click in self._clicks[self._current_frame_idx]:
                click.ui_point.set_visible(False)

            self._current_frame_idx = frame_idx
            self._image.set_data(self.current_frame.img.permute(1, 2, 0).cpu().numpy())

            self._update_mask()

            for bbox in self._bboxes[frame_idx]:
                bbox.ui_rect.set_visible(True)
            for click in self._clicks[frame_idx]:
                click.ui_point.set_visible(True)
            self._fig.canvas.draw_idle()

    def _clear_conditional_memory(self, frame_idx: int):
        self._sam2_video_predictor_state.memory_bank.clear_conditional_memories_in_frame(
            frame_idx=frame_idx
        )

    def _on_continue(self, event):
        self._continue_callback(self)

    def _update_mask(self):

        obj_id = self._current_obj_id

        clicks_obj = [
            c for c in self._clicks[self._current_frame_idx] if c.obj_id == obj_id
        ]
        bboxes_obj = [
            b for b in self._bboxes[self._current_frame_idx] if b.obj_id == obj_id
        ]

        if len(clicks_obj) > 0:
            points_coords = torch.as_tensor(
                [[click.x, click.y] for click in clicks_obj]
            )
            points_labels = torch.as_tensor([click.click_type for click in clicks_obj])
        else:
            points_coords = None
            points_labels = None

        if len(bboxes_obj) > 0:
            bboxes = torch.as_tensor(
                [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes_obj]
            )
        else:
            bboxes = None

        if points_coords is None and bboxes is None:
            prompt = None
        else:
            prompt = SAM2Prompt(
                obj_id=self._current_obj_id,
                points_coords=points_coords,
                points_labels=points_labels,
                boxes=bboxes,
            )

        frame = self.current_frame.img / 255.0
        frame = frame.to(self._sam2_video_predictor.device)

        if prompt is None:
            self._clear_conditional_memory(self._current_frame_idx)

        result = self._sam2_video_predictor.forward_embeddings(
            state=self._sam2_video_predictor_state,
            frame_idx=self._current_frame_idx,
            img_embeddings=self.current_frame.img_embeddings,
            img_pos_embeddings=self.current_frame.img_pos_embeddings,
            prompts=[prompt] if prompt is not None else [],
            create_memory=prompt is not None,
        )

        if obj_id in result:
            tmp_generator = torch.Generator()
            tmp_generator.manual_seed(obj_id)

            mask = result[obj_id].best_mask_logits > 0

            obj_color = torch.cat(
                [torch.rand(3, generator=tmp_generator), torch.tensor([0.6])], dim=0
            ).to(mask.device)

            mask = mask.reshape(
                self._video_height, self._video_width, 1
            ) * obj_color.reshape(1, 1, -1)
            mask = mask.cpu().numpy()
            self._mask_overlay.set_data(mask)
        else:
            self._mask_overlay.set_data(
                np.zeros((self._video_height, self._video_width, 4), dtype=np.uint8)
            )

    def _on_slider_change(self, val):
        self._jump_to_frame(int(val))

    def _on_prev_button(self, event):
        # Find the closest frame index to the left that has a bbox or a click.
        for i in range(self._current_frame_idx - 1, -1, -1):
            if len(self._bboxes[i]) > 0 or len(self._clicks[i]) > 0:
                self._slider.set_val(i)
                break

    def _on_next_button(self, event):
        # Find the closest frame index to the right that has a bbox or a click.
        for i in range(self._current_frame_idx + 1, self._total_frames):
            if len(self._bboxes[i]) > 0 or len(self._clicks[i]) > 0:
                self._slider.set_val(i)
                break

    def _on_drag_start(self, x: float, y: float):
        self._bbox_rect.set_visible(True)
        self._bbox_x1 = int(x)
        self._bbox_y1 = int(y)
        self._fig.canvas.draw_idle()

    def _on_drag_stop(self, x: float, y: float):
        self._bbox_x2 = int(x)
        self._bbox_y2 = int(y)

        min_x = clamp(min(self._bbox_x1, self._bbox_x2), 0, self._video_width)
        min_y = clamp(min(self._bbox_y1, self._bbox_y2), 0, self._video_height)
        max_x = clamp(max(self._bbox_x1, self._bbox_x2), 0, self._video_width)
        max_y = clamp(max(self._bbox_y1, self._bbox_y2), 0, self._video_height)

        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        bbox_area = bbox_width * bbox_height

        if bbox_area > self._bbox_prompt_min_area_px:
            # Create a new bounding box.
            ui_rect = plt.Rectangle(
                (min_x, min_y),
                bbox_width,
                bbox_height,
                fill=True,
                facecolor=(0, 1, 0, 0.2),
                edgecolor=(0, 1, 0, 1),
                linewidth=1,
            )
            self._ax_image.add_patch(ui_rect)
            self._bboxes[self._current_frame_idx].append(
                BoundingBox(
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    self._current_obj_id,
                    ui_rect,
                )
            )
            self._update_mask()

        self._bbox_rect.set_visible(False)
        self._fig.canvas.draw_idle()

    def _on_drag_move(self, x: float, y: float):
        self._bbox_x2 = int(x)
        self._bbox_y2 = int(y)

        min_x = clamp(min(self._bbox_x1, self._bbox_x2), 0, self._video_width)
        min_y = clamp(min(self._bbox_y1, self._bbox_y2), 0, self._video_height)
        max_x = clamp(max(self._bbox_x1, self._bbox_x2), 0, self._video_width)
        max_y = clamp(max(self._bbox_y1, self._bbox_y2), 0, self._video_height)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        self._bbox_rect.set_xy((min_x, min_y))
        self._bbox_rect.set_width(bbox_width)
        self._bbox_rect.set_height(bbox_height)
        self._bbox_rect.set_visible(bbox_width > 0 and bbox_height > 0)
        self._fig.canvas.draw_idle()

    def _on_mouse_click(self, x: float, y: float):
        x, y = int(x), int(y)
        x = max(0, min(x, self._video_width))
        y = max(0, min(y, self._video_height))

        # Delete mode.
        if self._is_shift_pressed:
            for bbox in self._bboxes[self._current_frame_idx]:
                if bbox.contains(x, y):
                    self._bboxes[self._current_frame_idx].remove(bbox)
                    bbox.ui_rect.remove()
                    self._update_mask()
                    self._fig.canvas.draw_idle()
                    return
            for click in self._clicks[self._current_frame_idx]:
                if click.is_close(x, y, self._point_distance_threshold_px):
                    self._clicks[self._current_frame_idx].remove(click)
                    click.ui_point.remove()
                    self._update_mask()
                    self._fig.canvas.draw_idle()
                    return
        # Negative click.
        elif self._is_alt_pressed:
            ui_point = plt.Circle(
                (x, y),
                radius=10,
                fill=True,
                facecolor=(1, 0, 0, 0.5),
                edgecolor=(1, 0, 0, 1),
            )
            self._ax_image.add_patch(ui_point)
            self._clicks[self._current_frame_idx].append(
                Click(x, y, 0, self._current_obj_id, ui_point)
            )
            self._update_mask()
            self._fig.canvas.draw_idle()
        # Positive click.
        else:
            ui_point = plt.Circle(
                (x, y),
                radius=10,
                fill=True,
                facecolor=(0, 1, 0, 0.5),
                edgecolor=(0, 1, 0, 1),
            )
            self._ax_image.add_patch(ui_point)
            self._clicks[self._current_frame_idx].append(
                Click(x, y, 1, self._current_obj_id, ui_point)
            )
            self._update_mask()
            self._fig.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key == "alt":
            self._is_alt_pressed = True
        if event.key == "control":
            self._is_ctrl_pressed = True
        if event.key == "shift":
            self._is_shift_pressed = True

    def _on_key_release(self, event):
        if event.key == "alt":
            self._is_alt_pressed = False
        if event.key == "control":
            self._is_ctrl_pressed = False
        if event.key == "shift":
            self._is_shift_pressed = False

    def _on_mouse_press(self, event):
        if (
            event.inaxes == self._ax_image
            and event.xdata is not None
            and event.ydata is not None
        ):
            self._is_pressing = True
            self._press_x = event.xdata
            self._press_y = event.ydata

    def _on_mouse_move(self, event):

        if event.inaxes == self._ax_image:
            self._last_valid_x = event.xdata
            self._last_valid_y = event.ydata

        if (
            self._is_pressing
            and event.inaxes == self._ax_image
            and not self._is_dragging
        ):
            self._is_dragging = True
            self._on_drag_start(event.xdata, event.ydata)

        if self._is_dragging and event.inaxes == self._ax_image:
            self._on_drag_move(
                x=event.xdata,
                y=event.ydata,
            )

    def _on_mouse_release(self, event):
        if event.inaxes == self._ax_image:
            mouse_x = event.xdata
            mouse_y = event.ydata
        else:
            mouse_x = self._last_valid_x
            mouse_y = self._last_valid_y

        if self._is_dragging:
            self._is_dragging = False
            self._on_drag_stop(mouse_x, mouse_y)
        elif self._is_pressing:
            self._on_mouse_click(mouse_x, mouse_y)

        self._is_pressing = False
        self._is_dragging = False

    @property
    def current_frame(self) -> Frame:
        if (
            self._current_frame_cache is not None
            and self._current_frame_cache.frame_idx == self._current_frame_idx
        ):
            return self._current_frame_cache

        frame_pt = self._reader.load_frame_at(self._current_frame_idx)

        img_embeddings, img_pos_embeddings = self._sam2_video_predictor.encode_image(
            frame_pt.to(self._sam2_video_predictor.device)
        )

        self._current_frame_cache = Frame(
            self._current_frame_idx, frame_pt, img_embeddings, img_pos_embeddings
        )
        return self._current_frame_cache


@dataclass
class Click:
    x: int
    y: int
    click_type: int
    obj_id: int

    ui_point: plt.Circle

    def __post_init__(self):
        if self.click_type not in [0, 1]:
            raise ValueError(
                f"Expected click_type to be 1 (positive) or 0 (negative), got {self.click_type}"
            )

    def is_close(self, x: int, y: int, distance_threshold_px: int = 10) -> bool:
        dist = sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        return dist <= distance_threshold_px


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    obj_id: int

    ui_rect: plt.Rectangle

    def __post_init__(self):
        if self.x1 > self.x2:
            raise ValueError(f"Expected x1 <= x2, got {self.x1} > {self.x2}")
        if self.y1 > self.y2:
            raise ValueError(f"Expected y1 <= y2, got {self.y1} > {self.y2}")

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def clamp(x: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(x, max_val))


if __name__ == "__main__":
    from sam2.build_sam import build_sam2_generic_video_predictor
    import matplotlib
    matplotlib.use("TkAgg")

    sam2_video_predictor = build_sam2_generic_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "./checkpoints/sam2.1_hiera_small.pt",
        device="cuda",
    )

    video_path = "E:/Datasets/Guedelon/Carriere Luc/gopro1.mp4"
    video_reader = cv2.VideoCapture(video_path)
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sam2_video_predictor_state = SAM2GenericVideoPredictorState(
        video_hw=(video_height, video_width)
    )

    def continue_callback(widget: SAMVideoWidget):
        print(sam2_video_predictor_state.memory_bank.count_conditional_memories())
        print(sam2_video_predictor_state.memory_bank.count_non_conditional_memories())
        widget.close()

    widget = SAMVideoWidget(
        video_reader=video_reader,
        sam2_video_predictor=sam2_video_predictor,
        sam2_video_predictor_state=sam2_video_predictor_state,
        continue_callback=continue_callback,
    )
    widget.set_title(f"Video {video_path}")
    plt.show()

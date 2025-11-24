# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch

from kineo.visualization.colormap import Colormap
from aitviewer.renderables.point_clouds import PointClouds


def create_keypoints_sequence(
    points: torch.Tensor,
    points_scores: torch.Tensor | None = None,
    cmap: str | Colormap = "viridis",
    name: str = "PointCloud",
    points_visibility_threshold: float = -1,
) -> PointClouds:
    assert points.ndim == 3, (
        f"Expected points to be of shape (F, K, 3), got {points.shape}"
    )

    if points_scores is not None:
        assert points_scores.ndim == 2, (
            f"Expected points_scores to be of shape (F, K), got {points_scores.shape}"
        )

    n_frames = points.shape[0]
    n_points = points.shape[1]

    if isinstance(cmap, str):
        cmap = Colormap(cmap)

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if points_scores is None:
        colors = None
    else:
        points_scores = points_scores.reshape(-1)
        visible = points_scores > points_visibility_threshold
        colors = cmap.forward(points_scores)
        colors[~visible, 3] = 0
        points_scores = points_scores.reshape(n_frames, n_points)
        colors = colors.reshape(n_frames, n_points, 4).cpu().numpy()

    pcd = PointClouds(points=points.reshape(n_frames, -1, 3), colors=colors)
    pcd.name = name
    return pcd

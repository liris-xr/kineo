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
from typing import Sequence


class VideoParameters(torch.nn.Module):
    """
    Video parameters for optimization.

    This class contains video parameters that are either fixed (like frame timestamps) or optimizable parameters:

    Fixed parameters:
    - Frame timestamps (in seconds)
    - Number of frames

    Optimizable parameters:
    - Global time offset (in seconds). This is the offset to apply to the frame timestamps expressed in local time to convert them to global time.
    - Rolling shutter readout speed (in seconds). This is the time it takes for the sensor to read from the top to the bottom of the frame.
    """

    def __init__(
        self,
        n_frames: int,
        frame_timestamps: torch.Tensor | Sequence[float],
        batch_dims: torch.Size = torch.Size([]),
        device: torch.device = "cpu",
        requires_grad: bool = False,
    ):
        super(VideoParameters, self).__init__()
        self.batch_dims = batch_dims
        self.device = device

        frame_timestamps = torch.as_tensor(
            frame_timestamps,
            device=device,
            dtype=torch.float32,
        )

        assert frame_timestamps.shape == (
            *batch_dims,
            n_frames,
        ) or frame_timestamps.shape == (n_frames,)

        # Frame timestamps (in seconds) in local time.
        self._frame_timestamps = frame_timestamps.expand(*batch_dims, -1)
        self._n_frames = n_frames

        # Time offset relative to global time (in seconds).
        self._global_time_offset = torch.nn.Parameter(
            torch.zeros(
                (*batch_dims, 1),
                device=device,
                dtype=torch.float32,
                requires_grad=requires_grad,
            )
        )

        # The total time it takes for the sensor to read from the top to the bottom of the frame (in seconds).
        self._rolling_shutter_readout_speed = torch.nn.Parameter(
            torch.zeros(
                (*batch_dims, 1),
                device=device,
                dtype=torch.float32,
                requires_grad=requires_grad,
            )
        )

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_timestamps(self) -> torch.Tensor:
        return self._frame_timestamps

    @property
    def global_time_offset(self) -> torch.Tensor:
        return self._global_time_offset

    @property
    def rolling_shutter_readout_speed(self) -> torch.Tensor:
        return self._rolling_shutter_readout_speed

    @global_time_offset.setter
    def global_time_offset(self, global_time_offset: torch.Tensor) -> None:
        assert global_time_offset.shape == self._global_time_offset.shape
        self._global_time_offset.data = global_time_offset

    @rolling_shutter_readout_speed.setter
    def rolling_shutter_readout_speed(
        self, rolling_shutter_readout_speed: torch.Tensor
    ) -> None:
        assert (
            rolling_shutter_readout_speed.shape
            == self._rolling_shutter_readout_speed.shape
        )
        self._rolling_shutter_readout_speed.data = rolling_shutter_readout_speed

    def set_optimized_parameters(
        self,
        global_time_offset: bool | None = None,
        rolling_shutter_readout_speed: bool | None = None,
    ) -> None:
        """
        Enable or disable optimization of video parameters.

        Args:
            global_time_offset: Whether to optimize global time offset.
            rolling_shutter_readout_speed: Whether to optimize rolling shutter readout speed.
        """
        if global_time_offset is not None:
            self._global_time_offset.requires_grad = global_time_offset
        if rolling_shutter_readout_speed is not None:
            self._rolling_shutter_readout_speed.requires_grad = (
                rolling_shutter_readout_speed
            )

    @property
    def optimized_parameters(self) -> list[torch.Tensor]:
        """
        Get the optimized parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def named_optimized_parameters(self) -> dict[str, torch.Tensor]:
        """
        Get the optimized parameters with their names.
        """
        params = {}

        if self._global_time_offset.requires_grad:
            params["global_time_offset"] = self._global_time_offset
        if self._rolling_shutter_readout_speed.requires_grad:
            params["rolling_shutter_readout_speed"] = (
                self._rolling_shutter_readout_speed
            )
        return params

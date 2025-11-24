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

import torch
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
import numpy as np
import cv2
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.stage_timing import StageTiming, StageTimingsAnnotations
from time import perf_counter
import builtins
from typing import TypeVar, Generic

RuntimeConfigType = TypeVar("RuntimeConfigType")


class PipelineStage(ABC, Generic[RuntimeConfigType]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: RuntimeConfigType | None = None,
        dynamic_runtime_cfg: dict[str, RuntimeConfigType] | None = None,
    ):
        self._name = name
        self._order = order
        self._runtime_cfg = runtime_cfg
        self._dynamic_runtime_cfg = dynamic_runtime_cfg

    @property
    def name(self) -> str:
        return self._name

    @property
    def order(self) -> int:
        return self._order

    @property
    def runtime_cfg(self) -> RuntimeConfigType:
        return self._runtime_cfg

    @property
    def dynamic_runtime_cfg(self) -> dict[str, RuntimeConfigType] | None:
        return self._dynamic_runtime_cfg

    @abstractmethod
    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: RuntimeConfigType,
    ) -> None:
        raise NotImplementedError


class Pipeline:
    def __init__(
        self,
        stages: list[PipelineStage],
        seed: int = 19,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.stages = sorted(stages, key=lambda x: (x.order, x.name))
        self.device = device
        self.seed = seed

    @staticmethod
    def build_pipeline_from_config(
        cfg: str | Path | DictConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> Pipeline:
        if isinstance(cfg, str | Path):
            cfg = OmegaConf.load(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError("cfg must be a string, Path, or DictConfig")

        stages = [
            hydra.utils.instantiate(stage)
            for stage in cfg.pipeline.stages.values()
            if stage is not None
        ]

        for stage in stages:
            if not isinstance(stage, PipelineStage):
                raise ValueError("Stage is not an instance of PipelineStage")

        pipeline = Pipeline(
            stages=stages,
            device=device,
            seed=cfg.pipeline.seed,
        )
        return pipeline

    def run(
        self,
        sequence_name: str,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
    ) -> dict[str, Annotations]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        cv2.setRNGSeed(self.seed)

        annotations = annotations.copy()

        old_print = builtins.print

        def tqdm_print(*args, **kwargs):
            sep = kwargs.pop("sep", " ")
            end = kwargs.pop("end", "\n")
            msg = sep.join(map(str, args)) + end
            tqdm.write(msg, end="")  # tqdm.write supports end=

        builtins.print = tqdm_print

        pbar = tqdm(
            enumerate(self.stages),
            desc="Running pipeline stages",
            total=len(self.stages),
            leave=False,
        )

        all_stages_timings: list[StageTiming] = []

        for stage_idx, stage in pbar:
            pbar.set_postfix(stage_name=stage.name)

            stage_name = stage.name
            stage_cfg = stage.runtime_cfg

            if (
                stage.dynamic_runtime_cfg is not None
                and sequence_name in stage.dynamic_runtime_cfg
            ):
                stage_cfg = stage.dynamic_runtime_cfg[sequence_name]

            torch.cuda.synchronize()
            start_time = perf_counter()
            try:
                stage.forward(
                    sequence_name=sequence_name,
                    pipeline=self,
                    views=views,
                    annotations=annotations,
                    gt_annotations=gt_annotations,
                    runtime_cfg=stage_cfg,
                )
            except Exception as e:
                print(f"Error in stage {stage_name}: {e}")
                raise e
            torch.cuda.synchronize()
            end_time = perf_counter()

            all_stages_timings.append(
                StageTiming(
                    stage_name=stage_name,
                    stage_idx=stage_idx,
                    duration_seconds=end_time - start_time,
                )
            )

            # Update stage timings annotations every time so that they will be included for export
            annotations["stage_timings"] = StageTimingsAnnotations(
                annotations=all_stages_timings
            )

        pbar.close()
        builtins.print = old_print
        return annotations

    def __str__(self) -> str:
        """
        Return a nicely formatted string representation of the pipeline stages.
        """
        lines = ["\nPipeline Configuration:", "=" * 50]

        for i, stage in enumerate(self.stages):
            lines.append(f"\n[{i}] {stage.name}")

            # Print runtime config parameters
            if stage.runtime_cfg is not None:
                lines.append("  Runtime Configuration:")
                for key, value in vars(stage.runtime_cfg).items():
                    lines.append(f"    {key}: {value}")

            # Print dynamic runtime configs if they exist
            if stage.dynamic_runtime_cfg is not None:
                lines.append("  Dynamic Runtime Configurations:")
                for seq_name, cfg in stage.dynamic_runtime_cfg.items():
                    lines.append(f"    Sequence '{seq_name}':")
                    for key, value in vars(cfg).items():
                        lines.append(f"      {key}: {value}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

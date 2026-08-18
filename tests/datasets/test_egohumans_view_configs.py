import glob
import os

import pytest
from omegaconf import OmegaConf

CONFIGS_GLOB = "configs/experiments/benchmarks/*_hsfm[0-9].yaml"


def _view_configs():
    paths = sorted(glob.glob(CONFIGS_GLOB))
    assert paths, f"no fixed-view benchmark configs matched {CONFIGS_GLOB}"
    return paths


@pytest.mark.parametrize("config_path", _view_configs())
def test_every_sequence_gets_the_configured_view_count(config_path):
    n_views = int(os.path.basename(config_path).split("_hsfm")[1].split(".")[0])
    selection = OmegaConf.load(config_path).camera_selection

    for sequence_name, view_ids in selection.items():
        assert len(view_ids) == n_views, f"{sequence_name} has {len(view_ids)} views"
        assert len(set(view_ids)) == n_views, f"{sequence_name} repeats a view"


@pytest.mark.parametrize("config_path", _view_configs())
def test_output_and_cache_roots_are_setting_specific(config_path):
    suffix = os.path.basename(config_path).split("_")[-1].removesuffix(".yaml")
    cfg = OmegaConf.load(config_path)

    # Read raw so the oc.env interpolation is not resolved here.
    for key in ("output_root_dir", "cache_root_dir"):
        assert OmegaConf.to_container(cfg, resolve=False)[key].endswith(suffix), (
            f"{key} in {config_path} would collide with another setting's run"
        )

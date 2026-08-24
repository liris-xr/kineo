import glob
import os
import re

import pytest
from omegaconf import OmegaConf

CONFIGS_GLOB = "configs/experiments/benchmarks/*_[0-9]views*.yaml"


def _view_configs():
    paths = sorted(glob.glob(CONFIGS_GLOB))
    assert paths, f"no fixed-view benchmark configs matched {CONFIGS_GLOB}"
    return paths


def _stem(config_path: str) -> str:
    return os.path.basename(config_path).removesuffix(".yaml")


def _view_count(config_path: str) -> int:
    return int(re.search(r"_(\d+)views", _stem(config_path)).group(1))


def _view_setting(config_path: str) -> str:
    """The `<n>views` part, shared by a base config and its variants."""
    return re.search(r"_(\d+views)", _stem(config_path)).group(1)


@pytest.mark.parametrize("config_path", _view_configs())
def test_every_sequence_gets_the_configured_view_count(config_path):
    n_views = _view_count(config_path)
    selection = OmegaConf.load(config_path).camera_selection

    for sequence_name, view_ids in selection.items():
        assert len(view_ids) == n_views, f"{sequence_name} has {len(view_ids)} views"
        assert len(set(view_ids)) == n_views, f"{sequence_name} repeats a view"


@pytest.mark.parametrize("config_path", _view_configs())
def test_output_root_is_config_specific(config_path):
    # Read raw so the oc.env interpolation is not resolved here.
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    assert raw["output_root_dir"].endswith(_stem(config_path)), (
        f"output_root_dir in {config_path} would collide with another run"
    )


@pytest.mark.parametrize("config_path", _view_configs())
def test_cache_root_belongs_to_this_view_setting(config_path):
    # Variants that only change a bundle-adjustment setting share their base
    # config's cache on purpose - the cached stages (MoGe intrinsics, NLF
    # keypoints) do not read it, so the outputs are identical. Sharing across
    # view counts would be a real bug, so pin the `<n>views` part only.
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    setting = _view_setting(config_path)

    assert raw["cache_root_dir"].endswith(setting), (
        f"cache_root_dir in {config_path} does not belong to {setting}"
    )

import glob
import os
import re

import pytest
from omegaconf import OmegaConf

CONFIGS_GLOB = "configs/experiments/ablation_study/n_views/*_[0-9]*.yaml"
BENCHMARK = (
    "configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estDk1k2.yaml"
)


def _view_configs():
    paths = sorted(glob.glob(CONFIGS_GLOB))
    assert paths, f"no fixed-view benchmark configs matched {CONFIGS_GLOB}"
    return paths


def _stem(config_path: str) -> str:
    return os.path.basename(config_path).removesuffix(".yaml")


def _view_count(config_path: str) -> int:
    return int(re.search(r"_n_views_(\d+)", _stem(config_path)).group(1))


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
def test_cache_root_is_shared_with_the_benchmark(config_path):
    # The cache holds one file per view, so a 2-view run and the full-rig
    # benchmark read the same directory: the first fills in the views it needs
    # and the other reuses them. A view count with its own directory would
    # re-run MoGe and NLF over views that are already on disk.
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    benchmark = OmegaConf.to_container(OmegaConf.load(BENCHMARK), resolve=False)

    assert raw["cache_root_dir"] == benchmark["cache_root_dir"], (
        f"cache_root_dir in {config_path} does not match the benchmark, so the "
        "per-view cache cannot be shared with it"
    )

from kineo.datasets.egohumans.configs.default import _C as default_cfg
from kineo.datasets.egohumans.configs.default import update_config
import os

CONFIG_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str):
    cfg = default_cfg.clone()
    update_config(cfg, config_path)
    return cfg

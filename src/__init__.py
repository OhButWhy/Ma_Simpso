"""
Simpsons Character Classification Package.

Модели и утилиты для классификации персонажей из Симпсонов.
"""

__version__ = "1.0.0"

# Импортируем основные компоненты для удобного доступа
from .model import SimpleCNN, create_model
from .data_utils import (
    SimpsonsDataset,
    create_dataloaders,
    build_transforms,
    split_samples,
    collect_samples,
)
from . import config
from .config import (
    set_seed,
    get_config_dict,
    load_config_from_json,
    apply_config_from_json,
)

__all__ = [
    'SimpleCNN',
    'create_model',
    'SimpsonsDataset',
    'create_dataloaders',
    'build_transforms',
    'split_samples',
    'collect_samples',
    'config',
    'set_seed',
    'get_config_dict',
    'load_config_from_json',
    'apply_config_from_json',
]

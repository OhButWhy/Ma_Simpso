"""Конфигурация гиперпараметров для обучения модели."""

import json
import random
from pathlib import Path

import numpy as np
import torch

INPUT_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Фиксирует random seed для воспроизводимости результатов.

    Args:
        seed: Значение seed для генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_dict() -> dict:
    """Возвращает словарь с текущей конфигурацией."""
    return {
        "input_size": INPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "random_seed": RANDOM_SEED,
    }


def load_config_from_json(config_path: str | Path) -> dict:
    """
    Загружает конфигурацию из JSON файла.

    Args:
        config_path: Путь к training_config.json

    Returns:
        Словарь с полной информацией о конфигурации и окружении
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return config_data


def apply_config_from_json(config_path: str | Path) -> None:
    """
    Применяет параметры из JSON файла к глобальным переменным.

    Внимание: Модифицирует глобальные переменные модуля!

    Args:
        config_path: Путь к training_config.json
    """
    import sys

    config_data = load_config_from_json(config_path)
    hyperparams = config_data.get("hyperparameters", {})

    # Получаем текущий модуль
    current_module = sys.modules[__name__]

    # Применяем гиперпараметры
    if "input_size" in hyperparams:
        setattr(current_module, "INPUT_SIZE", hyperparams["input_size"])
    if "batch_size" in hyperparams:
        setattr(current_module, "BATCH_SIZE", hyperparams["batch_size"])
    if "learning_rate" in hyperparams:
        setattr(current_module, "LEARNING_RATE", hyperparams["learning_rate"])
    if "num_epochs" in hyperparams:
        setattr(current_module, "NUM_EPOCHS", hyperparams["num_epochs"])
    if "random_seed" in hyperparams:
        setattr(current_module, "RANDOM_SEED", hyperparams["random_seed"])
    if "val_split" in hyperparams:
        setattr(current_module, "VAL_SPLIT", hyperparams["val_split"])
    if "test_split" in hyperparams:
        setattr(current_module, "TEST_SPLIT", hyperparams["test_split"])

"""
Скрипт для просмотра и анализа сохраненной конфигурации обучения.

Использование:
    python scripts/show_config.py                  # Показать последнюю конфигурацию
    python scripts/show_config.py path/to/config.json  # Показать конкретный файл
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в path для импорта src как пакета
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import load_config_from_json  # noqa: E402


def print_config(config_path: Path) -> None:
    """Красиво выводит конфигурацию."""
    try:
        config_data = load_config_from_json(config_path)
    except FileNotFoundError:
        print(f"Файл конфигурации не найден: {config_path}")
        return

    print("=" * 60)
    print(f"Конфигурация: {config_path}")
    print("=" * 60)

    # Информация о времени и окружении
    if "timestamp" in config_data:
        print(f"\nTimestamp: {config_data['timestamp']}")

    if "environment" in config_data:
        env = config_data["environment"]
        print("\nOkruzhenie (Environment):")
        print(f"   PyTorch version: {env.get('torch_version', 'N/A')}")
        print(f"   CUDA available: {env.get('cuda_available', False)}")
        if env.get("cuda_version"):
            print(f"   CUDA version: {env.get('cuda_version')}")

    # Гиперпараметры
    if "hyperparameters" in config_data:
        hp = config_data["hyperparameters"]
        print("\nHyperparameters:")
        print(f"   Input size: {hp.get('input_size', 'N/A')}")
        print(f"   Batch size: {hp.get('batch_size', 'N/A')}")
        print(f"   Learning rate: {hp.get('learning_rate', 'N/A')}")
        print(f"   Num epochs: {hp.get('num_epochs', 'N/A')}")
        print(f"   Val split: {hp.get('val_split', 'N/A')}")
        print(f"   Test split: {hp.get('test_split', 'N/A')}")
        print(f"   Random seed: {hp.get('random_seed', 'N/A')}")

    # Результаты обучения
    metrics = {
        k: v for k, v in config_data.items()
        if k not in ["timestamp", "hyperparameters", "environment"]
    }

    if metrics:
        print("\nTraining Results:")
        for key, value in metrics.items():
            formatted_key = key.replace("_", " ").title()
            print(f"   {formatted_key}: {value}")

    print("\n" + "=" * 60)


def main():
    """Главная функция."""
    if len(sys.argv) > 1:
        # Пользователь передал путь к конфигурации
        config_path = Path(sys.argv[1])
    else:
        # Используем последнюю сохраненную конфигурацию
        config_path = Path("src/results/training_config.json")

    if not config_path.exists():
        print(f"Файл не найден: {config_path}")
        print("\nUsage:")
        print("  python scripts/show_config.py")
        print("  python scripts/show_config.py path/to/config.json")
        return

    print_config(config_path)


if __name__ == "__main__":
    main()

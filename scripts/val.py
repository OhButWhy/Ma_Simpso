"""
Скрипт валидации обученной модели.

Загружает сохраненные веса и оценивает модель на валидационном наборе
без повторного обучения (быстро).

Использование:
    python scripts/val.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в path для импорта src как пакета
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src import config, create_dataloaders, create_model, set_seed  # noqa: E402


def validate(
    model: nn.Module,
    val_loader,
    device,
) -> tuple[float, float]:
    """Валидация модели без обновления весов.

    Args:
        model: нейросеть в режиме eval
        val_loader: DataLoader с валидационными данными
        device: устройство (CPU или GPU)

    Returns:
        (средняя потеря, точность на валидации в процентах)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main():
    """Главная функция валидации."""

    # Фиксируем random seed
    set_seed(config.RANDOM_SEED)

    print("=" * 60)
    print("Валидация: классификация персонажей Симпсонов")
    print("=" * 60)

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство: {device}")

    # Загружаем датасет
    print("\nЗагрузка датасета...")
    dataset_info = create_dataloaders(
        dataset_dir="src/dataset",
        input_size=config.INPUT_SIZE,
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=42,
        num_workers=0,
    )

    val_loader = dataset_info["val_loader"]
    num_classes = dataset_info["num_classes"]

    print(f"Валидационные примеры: {dataset_info['val_size']}")
    print(f"Классов: {num_classes}")

    # Создаем модель
    print("\nСоздание модели...")
    model = create_model(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE
    )
    model = model.to(device)

    # Загружаем сохраненные веса
    checkpoint_path = Path("src/models/best_model.pt")
    if not checkpoint_path.exists():
        print(f"Ошибка: модель не найдена в {checkpoint_path}")
        print("Сначала запустите train.py для обучения модели")
        return

    print(f"\nЗагрузка весов из: {checkpoint_path}")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print("Веса загружены успешно")

    # Валидация
    print("\nОценка на валидационном наборе...")
    val_loss, val_acc = validate(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("=" * 60)
    print(f"Точность на валидации: {val_acc:.2f}%")
    print(f"Loss на валидации: {val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

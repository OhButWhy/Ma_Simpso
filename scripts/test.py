"""
Скрипт тестирования обученной модели.

Загружает сохраненные веса и оценивает модель на тестовом наборе.
Выводит полные метрики: accuracy, confusion matrix и т.д.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Добавляем корень проекта в path для импорта src как пакета
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from src import config, create_dataloaders, create_model, set_seed  # noqa: E402


def test_model(
    model: nn.Module,
    test_loader,
    class_names: list[str],
    device,
) -> dict:
    """Тестирование модели с подробными метриками.

    Args:
        model: нейросеть
        test_loader: DataLoader с тестовыми данными
        class_names: список имен классов
        device: устройство

    Returns:
        словарь с метриками
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()

            all_predictions.extend(predicted)
            all_labels.extend(labels)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Вычисляем метрики
    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Используем только классы, которые есть в тестовом наборе
    unique_labels = np.unique(all_labels)
    target_names = [class_names[i] for i in unique_labels]

    class_report = classification_report(
        all_labels,
        all_predictions,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "confusion_matrix": conf_matrix,
        "class_report": class_report,
    }


def save_confusion_matrix_text(
    conf_matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Сохраняет полную матрицу ошибок в текстовый файл."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Полная матрица ошибок\n")
        file.write("=" * 60 + "\n")
        file.write("Порядок классов (индекс: имя):\n")
        for index, class_name in enumerate(class_names):
            file.write(f"{index}: {class_name}\n")

        file.write("\nМатрица ошибок:\n")
        file.write(np.array2string(conf_matrix, max_line_width=200))
        file.write("\n")


def save_test_results(
    accuracy: float,
    class_report: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Сохраняет ключевые результаты тестового прогона."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "test_metrics.json"
    report_path = output_dir / "classification_report.txt"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with report_path.open("w", encoding="utf-8") as file:
        file.write(class_report)

    return metrics_path, report_path


def main():
    # Фиксируем random seed
    set_seed(config.RANDOM_SEED)

    print("=" * 60)
    print("Тестирование: классификация персонажей Симпсонов")
    print("=" * 60)

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

    test_loader = dataset_info["test_loader"]
    class_names = dataset_info["class_names"]
    num_classes = dataset_info["num_classes"]

    print(f"Тестовые примеры: {dataset_info['test_size']}")
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

    # Тестирование
    print("\nТестирование модели...")
    results = test_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
    )

    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    acc = results["accuracy"]
    print(f"\nОбщая точность: {acc:.4f} ({acc*100:.2f}%)")

    print("\nОтчет по классам:")
    print("-" * 60)
    print(results["class_report"])

    print("\nMatrица ошибок (первые 10x10 классов):")
    print("-" * 60)
    print(results["confusion_matrix"][:10, :10])

    results_dir = Path("src/results")
    confusion_matrix_path = results_dir / "confusion_matrix.txt"
    save_confusion_matrix_text(
        conf_matrix=results["confusion_matrix"],
        class_names=class_names,
        output_path=confusion_matrix_path,
    )
    metrics_path, report_path = save_test_results(
        accuracy=acc,
        class_report=results["class_report"],
        output_dir=results_dir,
    )
    print(f"\nПолная матрица ошибок сохранена: {confusion_matrix_path}")
    print(f"Метрики теста сохранены: {metrics_path}")
    print(f"Classification report сохранён: {report_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

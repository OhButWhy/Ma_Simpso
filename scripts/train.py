"""
Скрипт обучения модели классификации персонажей Симпсонов.

Использует:
- config.py для параметров
- data_utils.py для загрузки датасета
- model.py для архитектуры
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта в path для импорта src как пакета
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import (
    config,
    create_dataloaders,
    create_model,
    get_config_dict,
    set_seed,
)  # noqa: E402


def save_history_csv(
    history: list[dict],
    output_path: Path,
) -> None:
    """Сохраняет историю обучения в CSV файл."""
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(
    history: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Строит и сохраняет графики loss и accuracy."""
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    train_accs = [row["train_acc"] for row in history]
    val_accs = [row["val_acc"] for row in history]

    loss_plot_path = output_dir / "loss_curve.png"
    acc_plot_path = output_dir / "accuracy_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()

    return loss_plot_path, acc_plot_path


def save_config(
    output_path: Path,
    extra_info: dict | None = None,
) -> None:
    """Сохраняет конфигурацию запуска в JSON.

    Args:
        output_path: Путь для сохранения JSON файла
        extra_info: Дополнительная информация (опционально)
    """
    # Безопасное получение версии CUDA
    cuda_version = None
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda  # type: ignore
        except AttributeError:
            cuda_version = None

    config_data = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": get_config_dict(),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": cuda_version,
        },
    }

    if extra_info:
        config_data.update(extra_info)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
) -> tuple[float, float]:
    """Обучение модели на одну эпоху.

    Args:
        model: нейросеть
        train_loader: DataLoader с обучающими данными
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (CPU или GPU)

    Returns:
        (средняя потеря, точность на обучении)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Обучение",
        leave=False
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Прямой проход
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device,
) -> tuple[float, float]:
    """Валидация модели (без обновления весов).

    Args:
        model: нейросеть
        val_loader: DataLoader с валидационными данными
        criterion: функция потерь
        device: устройство (CPU или GPU)

    Returns:
        (средняя потеря, точность на валидации)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()  # не удалятть штраф 500 рублей
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main():
    """Главная функция обучения."""

    # Фиксируем random seed для воспроизводимости
    set_seed(config.RANDOM_SEED)

    print("=" * 60)
    print("Обучение: классификация персонажей Симпсонов")
    print("=" * 60)
    print(f"Random seed: {config.RANDOM_SEED}")

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Загружаем датасет (только train и val, без test!)
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

    train_loader = dataset_info["train_loader"]
    val_loader = dataset_info["val_loader"]
    num_classes = dataset_info["num_classes"]

    print("Датасет загружен:")
    print(f"  Классов: {num_classes}")
    print(f"  Обучающие примеры: {dataset_info['train_size']}")
    print(f"  Валидационные примеры: {dataset_info['val_size']}")

    # Создаем модель
    print("\nСоздание модели...")
    model = create_model(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Модель создана ({total_params:,} параметров)")

    # Оптимизация
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=5e-5,
    )

    # Директория для весов
    models_dir = Path("src/models")
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = models_dir / "best_model.pt"

    # Директория для артефактов обучения
    results_dir = Path("src/results")
    results_dir.mkdir(exist_ok=True)
    history_csv_path = results_dir / "training_history.csv"

    # Обучение
    print(f"\nНачинаем обучение ({config.NUM_EPOCHS} эпох)...")
    print("-" * 60)

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Эпоха {epoch}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [*] Лучшая модель сохранена! (Val Acc: {val_acc:.2f}%)")

    print("-" * 60)
    print("\nОбучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Веса сохранены: {checkpoint_path}")

    # Сохраняем артефакты
    save_history_csv(history=history, output_path=history_csv_path)
    loss_plot_path, acc_plot_path = plot_history(
        history=history,
        output_dir=results_dir,
    )

    config_path = results_dir / "training_config.json"
    save_config(
        output_path=config_path,
        extra_info={
            "best_val_accuracy": best_val_acc,
            "total_epochs_trained": config.NUM_EPOCHS,
            "num_classes": num_classes,
            "train_size": dataset_info["train_size"],
            "val_size": dataset_info["val_size"],
        },
    )

    print(f"История обучения сохранена: {history_csv_path}")
    print(f"График loss сохранен: {loss_plot_path}")
    print(f"График accuracy сохранен: {acc_plot_path}")
    print(f"Конфигурация сохранена: {config_path}")


if __name__ == "__main__":
    main()

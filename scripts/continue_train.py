"""
Скрипт для продолжения обучения уже обученной модели.

Загружает сохранённые веса и конфигурацию, продолжает обучение
на заданное количество дополнительных эпох.

Использование:
    python scripts/continue_train.py --epochs 5
    python scripts/continue_train.py --epochs 10 --lr 0.0003
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
    set_seed,
)  # noqa: E402


def load_history_csv(history_path: Path) -> list[dict]:
    """Загружает историю обучения из CSV."""
    history = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                })
    return history


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


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    """Обучение модели на одну эпоху."""
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

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

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
    """Валидация модели."""
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

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main(
    num_epochs: int = 5,
    learning_rate: float | None = None,
):
    """Продолжить обучение."""
    set_seed(config.RANDOM_SEED)

    print("=" * 60)
    print("Продолжение обучения модели")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

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
        use_weighted_sampler=config.USE_WEIGHTED_SAMPLER,
    )

    train_loader = dataset_info["train_loader"]
    val_loader = dataset_info["val_loader"]
    num_classes = dataset_info["num_classes"]

    print(f"Датасет загружен: {dataset_info['train_size']} train, "
          f"{dataset_info['val_size']} val")

    # Создаём модель
    print("\nСоздание модели...")
    model = create_model(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        dropout_rate=config.DROPOUT_RATE,
    )
    model = model.to(device)

    # Загружаем веса
    checkpoint_path = Path("src/models/best_model.pt")
    if not checkpoint_path.exists():
        print("Ошибка: файл с весами не найден!")
        return

    print(f"Загрузка весов из {checkpoint_path}...")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

    # Настраиваем обучение
    current_lr = learning_rate or config.LEARNING_RATE
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=current_lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
    )

    print(f"Learning rate: {current_lr}")
    print(f"Dropout rate: {config.DROPOUT_RATE}")

    # Загружаем историю
    history_path = Path("src/results/training_history.csv")
    history = load_history_csv(history_path)
    start_epoch = len(history) + 1

    print(f"\nПродолжаем обучение с эпохи {start_epoch} "
          f"на {num_epochs} эпох...")
    print("-" * 60)

    best_val_acc = max(
        [h["val_acc"] for h in history],
        default=0.0
    )
    best_epoch = start_epoch - 1
    epochs_without_improvement = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.GRAD_CLIP_NORM,
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        tqdm.write(
            f"Эпоха {epoch}/{start_epoch + num_epochs - 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        scheduler.step(val_acc)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(
                f"  [*] Лучшая модель сохранена! (Val Acc: {val_acc:.2f}%)"
            )
        else:
            epochs_without_improvement += 1

    print("-" * 60)
    print("\nОбучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Лучшая эпоха: {best_epoch}")

    # Сохраняем историю и графики
    results_dir = Path("src/results")
    save_history_csv(history=history, output_path=history_path)
    loss_plot_path, acc_plot_path = plot_history(
        history=history,
        output_dir=results_dir,
    )

    print(f"История обновлена: {history_path}")
    print(f"График loss сохранён: {loss_plot_path}")
    print(f"График accuracy сохранён: {acc_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Продолжить обучение иснованной модели"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Количество дополнительных эпох для обучения",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Новый learning rate (если не указан, используется конфиг)",
    )
    args = parser.parse_args()

    main(num_epochs=args.epochs, learning_rate=args.lr)

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SimpsonsDataset(Dataset):
    """Пользовательский датасет для изображений Симпсонов.
    Каждый элемент - это кортеж (изображение, метка класса)."""
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        transform=None,
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _is_image_file(file_path: Path) -> bool:
    """Проверяет, является ли файл изображением по расширению."""
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def collect_samples(
    dataset_dir: str | Path,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Сканирует директорию с данными и собирает пути
    к изображениям и их метки классов.
    Ожидается, что структура папок следующая:
    dataset_dir/
        class_0/
            img1.jpg
            img2.jpg
            ...
        class_1/
            img3.jpg
            img4.jpg
            ...
        ..."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_names: List[str] = []
    all_samples: List[Tuple[Path, int]] = []

    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        image_files = sorted(
            [
                path
                # Генератор всех файлов в папке и подпапках класса
                for path in class_dir.rglob("*")
                if path.is_file() and _is_image_file(path)
            ]
        )

        if not image_files:
            continue

        class_index = len(class_names)
        class_names.append(class_dir.name)
        # extend добавляет к списку все элементы из другого списка
        # в данном случае - кортежи (путь к изображению, индекс класса)
        all_samples.extend((image_path, class_index)
                           for image_path in image_files)

    if not class_names:
        raise ValueError("No non-empty class folders with images were found.")

    return all_samples, class_names


def build_transforms(input_size: int):
    """Создает трансформации для обучения и валидации/тестирования.
    Для обучения используются аугментации,
    для валидации/тестирования - только ресайз и нормализация."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    return train_transform, eval_transform


def split_samples(
    samples: Sequence[Tuple[Path, int]],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
]:
    """Разбивает список образцов на обучающую,
    валидационную и тестовую выборки.
    Разбиение выполняется по классам,
    чтобы сохранить пропорции классов в каждой выборке."""
    if not (0 <= val_split < 1 and 0 <= test_split < 1):
        raise ValueError("val_split and test_split must be in range [0, 1).")

    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be < 1.")

    rng = random.Random(seed)

    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for sample in samples:
        _, label = sample
        by_class.setdefault(label, []).append(sample)

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for label_samples in by_class.values():
        label_samples = label_samples.copy()
        rng.shuffle(label_samples)

        n_samples = len(label_samples)
        n_val = int(n_samples * val_split)
        n_test = int(n_samples * test_split)

        if n_samples - (n_val + n_test) < 1:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        val_part = label_samples[:n_val]
        test_part = label_samples[n_val: n_val + n_test]
        train_part = label_samples[n_val + n_test:]

        train_samples.extend(train_part)
        val_samples.extend(val_part)
        test_samples.extend(test_part)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def create_dataloaders(
    dataset_dir: str | Path,
    input_size: int,
    batch_size: int,
    val_split: float,
    test_split: float,
    seed: int = 42,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
):
    samples, class_names = collect_samples(dataset_dir)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    train_transform, eval_transform = build_transforms(input_size=input_size)

    train_dataset = SimpsonsDataset(train_samples, transform=train_transform)
    val_dataset = SimpsonsDataset(val_samples, transform=eval_transform)
    test_dataset = SimpsonsDataset(test_samples, transform=eval_transform)

    # pin_memory ускоряет передачу данных на GPU, если он доступен
    pin_memory = torch.cuda.is_available()

    train_sampler = None
    if use_weighted_sampler:
        class_counts: Dict[int, int] = {}
        for _, label in train_samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Увеличиваем вероятность выбора редких классов.
        sample_weights = [1.0 / class_counts[label] for _, label in train_samples]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

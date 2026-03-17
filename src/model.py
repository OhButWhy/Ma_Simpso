import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Сверточная нейронная сеть для классификации персонажей.

    Архитектура:
    - 2 блока свертки (Conv -> BatchNorm -> ReLU -> MaxPool)
    - Дополнительный MaxPool для сохранения spatial-scale
    - Полносвязные слои с Dropout
    - Выходной слой для классификации
    """

    def __init__(self, num_classes: int, input_size: int = 128):
        super().__init__()

        # Блок 1: 3 -> 64 канала
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB изображение
            out_channels=64,    # 64 фильтра
            kernel_size=3,      # размер фильтра 3x3
            padding=1           # padding чтобы размер не менялся
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        # MaxPool уменьшает размер изображения в 2 раза
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 2: 64 -> 128 каналов
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Третий MaxPool без дополнительной свертки
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Вычисляем размер после сверток
        # input_size -> /2 -> /2 -> /2 = input_size // 8
        feature_size = (input_size // 8) * (input_size // 8) * 128

        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Регуляризация: отключаем 50% нейронов
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """Прямой проход через сеть.

        Args:
            x: тензор изображений [batch_size, 3, height, width]

        Returns:
            логиты для каждого класса [batch_size, num_classes]
        """
        # Сверточные блоки
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(x)

        # Полносвязные слои
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def create_model(num_classes: int, input_size: int = 128) -> SimpleCNN:
    """Создает и возвращает модель для классификации.

    Args:
        num_classes: количество классов для предсказания
        input_size: размер входного изображения (по одной стороне)

    Returns:
        Инициализированная модель
    """
    model = SimpleCNN(num_classes=num_classes, input_size=input_size)
    return model


if __name__ == "__main__":
    # Тестирование архитектуры
    model = create_model(num_classes=43, input_size=128)

    # Создаем тестовый тензор (батч из 4 изображений)
    test_input = torch.randn(4, 3, 128, 128)

    # Прогоняем через модель
    output = model(test_input)

    print(f"Размер входа: {test_input.shape}")
    print(f"Размер выхода: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Количество параметров: {total_params:,}")

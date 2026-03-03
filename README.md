# Ma_Simpso

Классификация персонажей Симпсонов с использованием сверточной нейронной сети на PyTorch.

## Описание

Проект решает задачу многоклассовой классификации изображений (42 класса персонажей). Модель обучается на датасете Simpsons Characters и оценивается на отложенных валидационной и тестовой выборках.

## Архитектура и параметры

**Модель:** `SimpleCNN`

- 3 сверточных блока (32 → 64 → 128)
- MaxPooling после каждого блока
- 2 полносвязных слоя и Dropout 0.5
- Количество параметров: 8,492,906

**Гиперпараметры:**

- Размер входа: 128×128
- Batch size: 32
- Optimizer: Adam (`lr=0.001`, `weight_decay=1e-4`)
- Функция потерь: CrossEntropyLoss
- Эпохи: 15
- Разбиение: 70/15/15 (train/val/test)

## Структура репозитория

```text
Ma_Simpso/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── model.py
│   ├── dataset/
│   ├── models/
│   └── results/
├── scripts/
│   ├── train.py
│   ├── val.py
│   └── test.py
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

или (через `uv`):

```bash
uv pip sync uv.lock
```

### Dev-зависимости

Для разработки с форматтерами и линтерами:

```bash
pip install -e ".[dev]"
```

## Качество кода

### Pre-commit хуки

Проект использует pre-commit для автоматической проверки кода перед коммитом:

- **black** — форматирование кода
- **isort** — сортировка импортов
- **flake8** — проверка стиля и ошибок

Установка:

```bash
pre-commit install
```

Ручной запуск проверок:

```bash
pre-commit run --all-files
```

### Воспроизводимость

Зафиксирован `RANDOM_SEED = 42` в [src/config.py](src/config.py). Функция `set_seed()` устанавливает seed для:

- `random`, `numpy.random`
- `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.deterministic = True`

Все скрипты ([train.py](scripts/train.py), [val.py](scripts/val.py), [test.py](scripts/test.py)) вызывают `set_seed()` в начале выполнения.

## Датасет

Ссылка: [Kaggle Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/data)

Ожидаемая структура:

```text
src/dataset/
├── abraham_grampa_simpson/
├── agnes_skinner/
├── bart_simpson/
└── ...
```

Внутри каждой папки класса — изображения (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

## Как воспроизвести результаты

1. Обучение:

```bash
python scripts/train.py
```

1. Валидация (без переобучения):

```bash
python scripts/val.py
```

1. Финальное тестирование:

```bash
python scripts/test.py
```

## Разбиение данных

Разбиение выполняется автоматически внутри `src/data_utils.py` из общего `src/dataset/`:

- Train: 70%
- Validation: 15%
- Test: 15%

Используется стратифицированное разбиение по классам (сохранение пропорций классов).

## Артефакты, которые сохраняются

После запуска `python scripts/train.py`:

- Веса лучшей модели: `src/models/best_model.pt`
- История обучения (по эпохам): `src/results/training_history.csv`
- График потерь: `src/results/loss_curve.png`
- График точности: `src/results/accuracy_curve.png`
- Конфигурация запуска: `src/results/training_config.json` (гиперпараметры, seed, версии библиотек, timestamp)

После запуска `python scripts/test.py`:

- Итоговые метрики выводятся в консоль (overall accuracy, classification report, confusion matrix)
- Полная матрица ошибок сохраняется в текстовый файл: `src/results/confusion_matrix.txt`

## Метрики (текущий запуск)

| Метрика              | Значение |
| -------------------- | -------- |
| Точность на тесте    | 85.28%   |
| Точность валидации   | 85.00%   |
| Точность обучения    | 93.24%   |
| Precision (взвешен.) | 0.86     |
| Recall (взвешен.)    | 0.85     |
| F1-score (взвешен.)  | 0.85     |

## Краткий анализ результатов

- Увеличение accuracy происходит стабильно до 14-й эпохи.
- Лучший чекпоинт сохраняется по максимальной `val_acc`.
- Разница между train и val метриками умеренная, явного сильного переобучения нет.
- Сложнее всего распознаются редкие классы с малым числом изображений.

## Требования

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- scikit-learn >= 1.0
- NumPy >= 1.20
- Pillow >= 8.0
- tqdm >= 4.60

## Лицензия

Учебный проект.

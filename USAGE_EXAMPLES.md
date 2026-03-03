# Usage Examples for Ma_Simpso

## Working with Saved Configuration

This document demonstrates how to use the saved `training_config.json` file.

### Basic Usage Scenarios

1. View training parameters
2. Reproduce training with same parameters
3. Compare multiple experiments
4. Log and report results

## Example 1: Load and View Configuration

```python
from pathlib import Path
from src import load_config_from_json

config_path = Path("src/results/training_config.json")
config_data = load_config_from_json(config_path)

print("Training Configuration:")
print(f"Timestamp: {config_data['timestamp']}")
print(f"PyTorch version: {config_data['environment']['torch_version']}")
print(f"Hyperparameters: {config_data['hyperparameters']}")
print(f"Best validation accuracy: {config_data['best_val_accuracy']:.2%}")
```

## Example 2: Apply Configuration Parameters Globally

```python
from src import apply_config_from_json, config

apply_config_from_json("src/results/training_config.json")

print(f"Batch size: {config.BATCH_SIZE}")
print(f"Learning rate: {config.LEARNING_RATE}")
print(f"Random seed: {config.RANDOM_SEED}")
```

## Example 3: Fix Random Seed from Configuration

```python
from pathlib import Path
from src import load_config_from_json, set_seed

config_path = Path("src/results/training_config.json")
config_data = load_config_from_json(config_path)

seed = config_data['hyperparameters']['random_seed']
set_seed(seed)

print(f"Seed fixed to: {seed}")
```

## Example 4: Compare Multiple Configurations

```python
from pathlib import Path
from src import load_config_from_json

configs = [
    Path("src/results/training_config.json"),
]

for cfg_path in configs:
    if cfg_path.exists():
        cfg = load_config_from_json(cfg_path)
        hp = cfg['hyperparameters']
        acc = cfg.get('best_val_accuracy', 'N/A')
        print(f"\n{cfg_path.name}:")
        print(f"  LR: {hp['learning_rate']}, Epochs: {hp['num_epochs']}")
```

## Example 5: Export Configuration to Markdown Report

```python
from pathlib import Path
from src import load_config_from_json

def export_config_to_markdown(config_path: Path, output_path: Path) -> None:
    """Export configuration to a markdown report."""
    config_data = load_config_from_json(config_path)
    hp = config_data['hyperparameters']
    env = config_data['environment']

    md_content = f"""# Training Configuration Report

## Execution Time

- **Timestamp**: {config_data['timestamp']}

## Hyperparameters

- **Input Size**: {hp['input_size']}
- **Batch Size**: {hp['batch_size']}
- **Learning Rate**: {hp['learning_rate']}
- **Epochs**: {hp['num_epochs']}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Report saved to: {output_path}")
```

## Viewing Configuration via Script

Use the `show_config.py` script to view configuration:

```bash
python scripts/show_config.py
python scripts/show_config.py path/to/config.json
```

## API Reference

### load_config_from_json

Load configuration from JSON file.

```python
from src import load_config_from_json

config_data = load_config_from_json("src/results/training_config.json")
```

### apply_config_from_json

Apply parameters from JSON file to global configuration.

```python
from src import apply_config_from_json

apply_config_from_json("src/results/training_config.json")
```

### set_seed

Fix random seed for reproducibility.

```python
from src import set_seed

set_seed(42)
```

### get_config_dict

Get current configuration as dictionary.

```python
from src import get_config_dict

config = get_config_dict()
print(config)
```

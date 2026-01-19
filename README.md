# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code
Hylke Wierda

## Quick Start

1. Connect to Snellius:
```bash
ssh user@snellius.surf.nl
```
2. Load modules:
```bash
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
```
3. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
4. Install dependencies:
```bash
pip install torch torchvision matplotlib


## Project Structure

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```

# calippo
**CALI**bration of **P**hotonics **P**r**O**cessors


## Installation
The package can be installed using `pip` (Python version `3.11` or above is suggested)

```bash
python3 -m venv .venvs/cal
source .venvs/cal/bin/activate
pip install git+ssh://git@github.com/ricalbr/calippo.git
```

## Usage

Use-case example:

```python
from calippo.calibration import Trainer, TrainingConfig, TrainingPipeline, load_data
from calippo.model import Clements
from calippo.tools import get_device
import torch

# Use the GPU if CUDA is available, otherwise fall back to the CPU
device = get_device()
if device.type == "cuda":
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
else:
    print(f"Using device: {device}")

processor = Clements(
    n_modes=6,
    non_linear=False,
    unbalanced_beam_splitters=True,
    device=device,
)

cfg = TrainingConfig()  # use default training settings

# Data Loading
train_bundle, test_bundle = load_data(
    data_filepath="20261117_PIC6810.pkl",
    train_ratio=cfg.train_ratio,
    device=device,
)

# Training
trainer = Trainer(processor, cfg)
pipeline = TrainingPipeline(trainer, cfg)
try:
    pipeline.run_multi_start(train_bundle, test_bundle)
except KeyboardInterrupt:
    print("Ctrl-C program interrupted.\n")

# Export best model weights
processor.export_calibration("calibration.pkl", state_dict=pipeline.best_weights)```

## Issues

To request features or report bugs open an issue [here](https://github.com/ricalbr/calippo/issues)

---

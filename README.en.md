# RGC-LLM Engineering Prototype

RGC-LLM (Resonant Graph Calculus LLM) is a research-driven engineering prototype. It is not a production LLM.

## Project layout

```text
rgc-llm-prototype/
├── configs/
│   └── default.yaml
├── data/
│   ├── train_samples.json
│   └── train_samples.jsonl
├── src/rgc_llm/
│   ├── cli/
│   ├── core/
│   ├── modules/
│   ├── training/
│   ├── utils/
│   └── model.py
├── tests/
├── README.md
├── README.en.md
└── README.zh-CN.md
```

## Installation

```bash
cd rgc-llm-prototype
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Data formats

### JSONL
Each line is one object.

```json
{"input": "Explain why latency rises.", "target": "Represent latency, bottlenecks, and retries."}
```

### JSON
Supported forms:

```json
[
  {"input": "...", "target": "..."}
]
```

or:

```json
{
  "samples": [
    {"input": "...", "target": "..."}
  ]
}
```

The field names are configurable via:

- `data.text_field`
- `data.target_field`

## Training

Default config already points to `data/train_samples.jsonl`.

```bash
rgc-train fit --config configs/default.yaml
```

Outputs are written to `training.output_dir`, defaulting to:

```text
outputs/default/
```

Artifacts include:

- `epoch_1.pt`, `epoch_2.pt`, ...
- `final.pt`

### Use your own JSON file

Edit `configs/default.yaml`:

```yaml
data:
  train_file: data/my_train.jsonl
  file_format: auto
  text_field: input
  target_field: target
  toy_samples: []
```

Then train again:

```bash
rgc-train fit --config configs/default.yaml
```

## Inference

```bash
rgc-infer run \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --text "Analyze the causal structure of supplier delays and launch risk."
```

Optional flags:

- `--deep true|false`
- `--urgency 0.0..1.0`

## Checkpoint format

Training now saves a structured checkpoint payload:

```python
{
  "format": "rgc-llm-checkpoint",
  "created_at": "...",
  "epoch": 2,
  "model_config": {...},
  "app_config": {...},
  "state_dict": {...},
  "extra": {...}
}
```

## Model extraction

### Export a pure state dict

```bash
rgc-export state-dict \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output exports/state_dict.pt
```

### Export individual modules

```bash
rgc-export modules \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output-dir exports/modules
```

You can also restrict exported modules:

```bash
rgc-export modules \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output-dir exports/modules \
  --include encoder --include decoder
```

### Inspect a checkpoint

```bash
rgc-export inspect --checkpoint outputs/default/final.pt
```

It prints metadata, tensor count, total parameter count, and per-tensor shapes.

## Python usage

```python
import torch
from rgc_llm.config import AppConfig
from rgc_llm.model import RGCLLM
from rgc_llm.utils.checkpointing import load_checkpoint_into_model

cfg = AppConfig.from_yaml("configs/default.yaml")
model = RGCLLM(cfg.model)
load_checkpoint_into_model(model, "outputs/default/final.pt")
model.eval()

with torch.no_grad():
    out = model("Compare wind and solar under storage constraints.", deep=True)

print(out.text)
```



## Next steps

- add validation and evaluation datasets
- add logging backends such as TensorBoard or Weights & Biases
- add TorchScript / ONNX export experiments
- replace the heuristic event encoder with a stronger neural parser
- add sparse batching for larger graphs

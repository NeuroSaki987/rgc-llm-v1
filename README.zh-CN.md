# RGC-LLM 工程版原型

RGC-LLM（Resonant Graph Calculus LLM）是一个**研究驱动的工程原型**。它不是生产级大模型。


## 项目结构

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

## 安装

```bash
cd rgc-llm-prototype
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 数据格式

### JSONL
每一行都是一条样本：

```json
{"input": "解释为什么时延会上升。", "target": "应表示时延、瓶颈和重试之间的关系。"}
```

### JSON
支持两种结构：

```json
[
  {"input": "...", "target": "..."}
]
```

或：

```json
{
  "samples": [
    {"input": "...", "target": "..."}
  ]
}
```

字段名可通过配置修改：

- `data.text_field`
- `data.target_field`

## 训练

默认配置已经指向 `data/train_samples.jsonl`：

```bash
rgc-train fit --config configs/default.yaml
```

训练输出目录由 `training.output_dir` 控制，默认是：

```text
outputs/default/
```

会生成：

- `epoch_1.pt`、`epoch_2.pt` 等周期 checkpoint
- `final.pt` 最终模型

### 使用你自己的 JSON / JSONL 数据

修改 `configs/default.yaml`：

```yaml
data:
  train_file: data/my_train.jsonl
  file_format: auto
  text_field: input
  target_field: target
  toy_samples: []
```

然后重新训练：

```bash
rgc-train fit --config configs/default.yaml
```

## 推理

```bash
rgc-infer run \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --text "分析供应链延迟如何影响产品上线风险。"
```

可选参数：

- `--deep true|false`：是否启用深度推理
- `--urgency 0.0..1.0`：紧急度，越高越偏向快速路径

## Checkpoint 格式

训练会保存结构化 checkpoint：

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

## 模型提取

### 导出纯 `state_dict`

```bash
rgc-export state-dict \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output exports/state_dict.pt
```

### 导出子模块

```bash
rgc-export modules \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output-dir exports/modules
```

也可以只导出指定模块：

```bash
rgc-export modules \
  --config configs/default.yaml \
  --checkpoint outputs/default/final.pt \
  --output-dir exports/modules \
  --include encoder --include decoder
```

### 检查 checkpoint 内容

```bash
rgc-export inspect --checkpoint outputs/default/final.pt
```

会输出：

- checkpoint 元信息
- tensor 数量
- 总参数量
- 每个参数张量的名称、shape、dtype

## Python 方式加载模型

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
    out = model("比较风能和太阳能在储能受限下的适配性。", deep=True)

print(out.text)
```


## 嗯

- 增加验证集与评测脚本
- 接入 TensorBoard 或 Weights & Biases
- 补 TorchScript / ONNX 导出实验
- 用更强的神经语义解析器替换当前事件编码器
- 为更大图结构补稀疏 batch 支持

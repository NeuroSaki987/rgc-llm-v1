# RGC-LLM Engineering Prototype / RGC-LLM ~原型机~ 原型姬

English: see [README.en.md](README.en.md)  
中文：见 [README.zh-CN.md](README.zh-CN.md)
数学：我不会latex，稍微等着，以后再补

- 这是一个大语言模型的工程模型，不可用于生产，其可行性尚待验证
- 暂且起名rgc_llm，以后要是发展起来了就换个高大上的名字，我不知道叫他什么
- 我可没有什么语料可以训练这个模型，感兴趣的的自己拿走看看。
- 代码能跑，自己研究；设计的很优雅，跑起来一坨屎

## Quick start / 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

rgc-train fit --config configs/default.yaml
rgc-infer run --config configs/default.yaml --checkpoint outputs/default/final.pt --text "Explain the causal chain behind rising latency."
rgc-export inspect --checkpoint outputs/default/final.pt
```

## CLI overview / 命令总览

```bash
rgc-train fit --config configs/default.yaml
rgc-infer run --config configs/default.yaml --text "..." --checkpoint outputs/default/final.pt
rgc-export state-dict --config configs/default.yaml --checkpoint outputs/default/final.pt --output exports/state_dict.pt
rgc-export modules --config configs/default.yaml --checkpoint outputs/default/final.pt --output-dir exports/modules
rgc-export inspect --checkpoint outputs/default/final.pt
```

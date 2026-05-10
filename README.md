# 🔪 ONNX Model Surgery

<div align="center">

**Visual inspection, pruning, patching, and optimization toolkit for ONNX models.**

[![CI](https://github.com/Luv-Goel/onnx-model-surgery/actions/workflows/ci.yml/badge.svg)](https://github.com/Luv-Goel/onnx-model-surgery/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue?logo=python)](https://github.com/Luv-Goel/onnx-model-surgery)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Luv-Goel/onnx-model-surgery?style=social)](https://github.com/Luv-Goel/onnx-model-surgery/stargazers)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-orange)](https://pypi.org/project/onnx-model-surgery/)

</div>

No API keys. No cloud. Just your model, a terminal, and some surgical tools.

```bash
pip install onnx-model-surgery
onnx-surgery info model.onnx
onnx-surgery graph model.onnx
onnx-surgery prune model.onnx --op-types Dropout Identity
```

---

## What is this?

ONNX is great for interoperability, but once you've got a `.onnx` file, what do you actually do with it? 

Most people either:
- Load it and pray it runs
- Reach for Netron (great for viewing, not editing)
- Write one-off Python scripts that they lose

This project bridges that gap. Think of it as **a small workbench for dissecting, inspecting, and repairing ONNX models** — without needing to spin up a full ML framework or hunt for a GPU.

### What you can do

| Command | What it does |
|---------|-------------|
| Command | What it does |
|---------|-------------|
| `onnx-surgery info model.onnx` | Full summary: ops, shapes, parameters, graph structure |
| `onnx-surgery info --shapes model.onnx` | Summary + computational estimates |
| `onnx-surgery graph model.onnx` | ASCII visualization of the graph topology |
| `onnx-surgery stats model.onnx` | Operator type frequency table |
| `onnx-surgery flops model.onnx` | FLOPs, MACs, and parameter count estimation |
| `onnx-surgery prune model.onnx --op-types Dropout Identity` | Remove specific node types |
| `onnx-surgery strip model.onnx -o clean.onnx` | Strip unused weights + fold identity nodes |
| `onnx-surgery simplify model.onnx -o simple.onnx` | Graph simplification (constant folding + identity removal) |
| `onnx-surgery diff a.onnx b.onnx` | Structural comparison of two models |
| `onnx-surgery extract model.onnx --from X --to Y -o sub.onnx` | Extract subgraph between tensors |
| `onnx-surgery rename model.onnx --map old:new -o renamed.onnx` | Bulk rename tensors |
| `onnx-surgery report model.onnx -o report.html` | Generate standalone HTML report |
| `onnx-surgery validate model.onnx` | Run the official ONNX checker |
| `onnx-surgery json model.onnx` | Export everything as JSON for scripting |

### As a library

```python
from onnx_surgery import load_model, model_summary, ascii_graph
from onnx_surgery.tools.prune import prune_nodes

model = load_model("model.onnx")
print(model_summary(model))

# Remove all Dropout and Identity nodes
cleaned = prune_nodes(model, op_types=["Dropout", "Identity"])

# See what changed
print(ascii_graph(cleaned))
onnx.save(cleaned, "clean_model.onnx")
```

---

## Why would you need this?

A few real-world scenarios:

- **You trained with Dropout and now you want to export for inference** — Dropout is useless at inference time, but ONNX keeps it around. One command strips it.
- **Some tool generated a monster model with hundreds of Identity nodes** — Identity passes-through are harmless but make debugging impossible when reading the graph. Nuke `em.
- **You need to compare two versions of a model** — `onnx-surgery json` gives you a clean diff target.
- **You're debugging shape mismatches** — the inspect command shows every tensor shape across every node in one scrollable view.
- **You want to shave off unused weights before deployment** — strip removes initializers that aren't connected to anything.

---

## Installation

```bash
pip install onnx numpy rich       # core
pip install graphviz               # optional: pretty graph rendering
```

Then either:
```bash
pip install onnx-model-surgery     # from PyPI
# or
git clone https://github.com/Luv-Goel/onnx-model-surgery
cd onnx-model-surgery && pip install -e .
```

---

## Dependencies

**Core** (always needed):
- `onnx` — the official package
- `numpy` — everyone's friend
- `rich` — pretty terminal output

**Optional:**
- `graphviz` — if you want the DOT graph export

Zero cloud dependencies. Zero API keys. Zero GPU.

---

## Example workflow

```bash
# Download a model
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

# Look at what's inside
onnx-surgery info resnet50-v2-7.onnx

# Strip training-only ops
onnx-surgery prune resnet50-v2-7.onnx --op-types Dropout --output cleaned.onnx

# Remove unused weights
onnx-surgery strip cleaned.onnx --optimize extended --output final.onnx

# Check it's valid
onnx-surgery validate final.onnx
```

---

## Project structure

```
onnx-model-surgery/
├── onnx_surgery/
│   ├── core/
│   │   ├── model_loader.py    # Load & parse ONNX files
│   │   ├── graph.py           # Directed graph representation + traversal
│   │   └── visualization.py   # ASCII + Graphviz rendering
│   ├── tools/
│   │   ├── prune.py           # Node removal, weight stripping
│   │   ├── patch.py           # Operation replacement, tensor renaming
│   │   ├── inspect.py         # Detailed inspection + JSON export
│   │   ├── export.py          # Save, validate, optimize
│   │   ├── flops.py           # FLOPs and parameter estimation
│   │   ├── diff.py            # Structural model comparison
│   │   ├── extract.py         # Subgraph extraction
│   │   ├── simplify.py        # Graph simplification
│   │   └── report.py          # HTML report generation
│   └── cli/
│       └── main.py            # Click-free argparse CLI
├── tests/
│   └── test_core.py
└── examples/
    └── basic_surgery.py
```

---

## What this is NOT

- **Not Netron** — this is a terminal tool for surgery, not a GUI for browsing. Netron is better for visual browsing; this is better for programmatic batch work.
- **Not an optimization framework** — there's no kernel fusion or quantization here. This is about graph-level surgery.
- **Not a training framework** — if you want to train models, there are better tools. This is for post-training cleanup and inspection.

---

## License

MIT. Do whatever.

---

*Built because staring at raw ONNX protobufs is nobody's idea of a good time.*

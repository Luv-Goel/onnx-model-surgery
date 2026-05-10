# ONNX Model Surgery 🏥

<div align="center">

[![Version](https://img.shields.io/pypi/v/onnx-model-surgery?color=blue)](https://pypi.org/project/onnx-model-surgery/)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/Luv-Goel/onnx-model-surgery/actions/workflows/ci.yml/badge.svg)](https://github.com/Luv-Goel/onnx-model-surgery/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Luv-Goel/onnx-model-surgery/branch/main/graph/badge.svg)](https://codecov.io/gh/Luv-Goel/onnx-model-surgery)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-blueviolet)]()
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pypi/dm/onnx-model-surgery?color=orange)]()

**Visual ONNX model inspection, pruning, patching, quantization, and optimization toolkit. 15+ CLI commands for production model surgery.**

</div>

---

## Features

- **Model inspection** — Detailed model info, graph visualization, node-level statistics
- **Pruning** — Remove nodes, inputs, outputs, and unused branches
- **Strip** — Remove training metadata, doc strings, and non-essential data
- **Validation** — Check model integrity, shape consistency, and runtime errors
- **Analysis** — FLOP counting, parameter counting, tensor shape analysis
- **Diff** — Compare two models and show structural changes
- **Extract** — Extract subgraphs by node name or pattern
- **Simplify** — Fold constants, fuse operations, remove identity nodes
- **Rename** — Batch rename nodes, inputs, and outputs
- **Report** — Generate comprehensive HTML analysis reports
- **Quantization** — FP16 half-precision and INT8 dynamic/static quantization
- **Diff Report** — Interactive HTML diff visualization between model versions
- **JSON export** — Full model metadata as JSON

## Quick Start

```bash
pip install onnx-model-surgery

# Show model info
oms info model.onnx

# Print model graph
oms graph model.onnx

# Model statistics
oms stats model.onnx

# Validate model
oms validate model.onnx

# Count FLOPs
oms flops model.onnx

# Prune unused nodes
oms prune model.onnx --output pruned.onnx

# Diff two models
oms diff model-v1.onnx model-v2.onnx

# Strip training metadata
oms strip model.onnx --output stripped.onnx

# Extract subgraph
oms extract model.onnx --nodes "Conv_3,Relu_3,Conv_4" --output subgraph.onnx

# Rename nodes
oms rename model.onnx --map "Conv_3:conv3,Relu_3:relu3" --output renamed.onnx

# Generate report
oms report model.onnx --output report.html

# JSON export
oms json model.onnx --output model.json

# --- v0.3.0+ ---

# FP16 quantization (pure ONNX, no extra deps)
oms quantize model.onnx --mode fp16 --output model_fp16.onnx

# INT8 dynamic quantization (requires onnxruntime)
oms quantize model.onnx --mode int8 --output model_int8.onnx

# INT8 static quantization (requires calibration data)
oms quantize model.onnx --mode int8-static --calibrate inputs.npz --output model_int8.onnx

# Interactive HTML diff report
oms diff-report model-v1.onnx model-v2.onnx --output diff.html
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `oms info [model]` | Display model metadata |
| `oms graph [model]` | Visual graph representation |
| `oms stats [model]` | Node-level statistics |
| `oms prune [model]` | Remove unused nodes |
| `oms strip [model]` | Remove training metadata |
| `oms validate [model]` | Model integrity check |
| `oms json [model]` | Export as JSON |
| `oms flops [model]` | FLOPs estimation |
| `oms diff [a] [b]` | Compare two models |
| `oms extract [model]` | Subgraph extraction |
| `oms simplify [model]` | Graph simplification |
| `oms report [model]` | HTML analysis report |
| `oms rename [model]` | Batch node renaming |
| `oms quantize [model]` | FP16 / INT8 quantization |
| `oms diff-report [a] [b]` | Interactive HTML diff report |

## Quantization Guide

### FP16 (Half-Precision)

Reduces model size by ~50% with minimal accuracy loss. Works on any ONNX model without external dependencies.

```bash
# Basic FP16 conversion
oms quantize model.onnx --mode fp16 --output model_fp16.onnx

# Using the Python API
from onnx_surgery.tools.quantize import convert_to_fp16
from onnx_surgery.core import load_model

model = load_model("model.onnx")
fp16_model = convert_to_fp16(model)
onnx.save(fp16_model, "model_fp16.onnx")
```

### INT8 Dynamic Quantization

Quantizes weights to INT8 with dynamic activation ranges. No calibration data needed.

```bash
# Requires onnxruntime
pip install onnx-model-surgery[quant]

oms quantize model.onnx --mode int8 --output model_int8.onnx
```

### INT8 Static Quantization

Quantizes both weights and activations using representative calibration data. Highest compression ratio.

```bash
oms quantize model.onnx --mode int8-static \
  --calibrate calibration_data.npz \
  --output model_quant.onnx
```

## Diff Report

Compare two ONNX models visually:

```bash
# Generate an interactive HTML diff report
oms diff-report model_v1.onnx model_v2.onnx --output diff.html

# Using the Python API
from onnx_surgery.tools.diff_report import generate_diff_html
from onnx_surgery.core import load_model

a = load_model("v1.onnx")
b = load_model("v2.onnx")
html = generate_diff_html(a, b, title="v1 vs v2")
Path("diff.html").write_text(html)
```

The report shows:
- Summary cards with node/input/output/parameter deltas
- Colour-coded added / removed / changed node tables
- Operator distribution bar charts
- Input / output shape changes

## Python API

```python
from onnx_surgery.core import load_model, model_summary, SurgeryGraph, ascii_graph
from onnx_surgery.tools.prune import prune_nodes, strip_initializers
from onnx_surgery.tools.export import validate, optimize
from onnx_surgery.tools.inspect import inspect, to_json
from onnx_surgery.tools.flops import estimate_flops
from onnx_surgery.tools.diff import diff
from onnx_surgery.tools.extract import extract_subgraph
from onnx_surgery.tools.simplify import simplify
from onnx_surgery.tools.report import generate_html_report
from onnx_surgery.tools.quantize import convert_to_fp16, quantize_int8, quantize_model_file
from onnx_surgery.tools.diff_report import generate_diff_html

model = load_model("model.onnx")
print(inspect(model, detailed=True))
```

## Architecture

```
onnx-model-surgery/
├── onnx_surgery/
│   ├── cli/
│   │   ├── main.py           # CLI entry point (15 commands)
│   │   └── __init__.py
│   ├── core/
│   │   ├── graph.py          # SurgeryGraph — traversable ONNX graph
│   │   ├── model_loader.py   # load_model, model_summary, list_* utils
│   │   ├── visualization.py  # ASCII graph, Graphviz DOT, op_stats
│   │   └── __init__.py
│   ├── tools/
│   │   ├── prune.py          # Node/initializer pruning
│   │   ├── patch.py          # Tensor renaming, attribute patching
│   │   ├── inspect.py        # Pretty-print and JSON export
│   │   ├── export.py         # Save, validate, optimize
│   │   ├── flops.py          # FLOPs & parameter estimation
│   │   ├── diff.py           # Structural model comparison
│   │   ├── extract.py        # Subgraph extraction
│   │   ├── simplify.py       # Constant folding, BN fusion
│   │   ├── report.py         # HTML analysis report generator
│   │   ├── quantize.py       # INT8/FP16 quantization
│   │   ├── diff_report.py    # Interactive HTML diff report
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   └── test_core.py          # 12+ tests, CI-passing
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).

# ONNX Model Surgery ðŸ¥

<div align="center">

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-12%2F12-passing-brightgreen)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-blueviolet)]()

**Visual ONNX model inspection, pruning, patching, and optimization toolkit. 13 CLI commands for everything model surgery.**

</div>

---

## Features

- **Model inspection** â€” Detailed model info, graph visualization, node-level statistics
- **Pruning** â€” Remove nodes, inputs, outputs, and unused branches
- **Strip** â€” Remove training metadata, doc strings, and non-essential data
- **Validation** â€” Check model integrity, shape consistency, and runtime errors
- **Analysis** â€” FLOP counting, parameter counting, tensor shape analysis
- **Diff** â€” Compare two models and show structural changes
- **Extract** â€” Extract subgraphs by node name or pattern
- **Simplify** â€” Fold constants, fuse operations, remove identity nodes
- **Rename** â€” Batch rename nodes, inputs, and outputs
- **Report** â€” Generate comprehensive HTML analysis reports
- **JSON export** â€” Full model metadata as JSON

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

## Architecture

```
onnx-model-surgery/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ oms/
â”‚   â”‚   â”œâ”€â”€ cli.py          # CLI entry point
â”‚   â”‚   â”œâ”€â”€ commands/       # 13 command modules
â”‚   â”‚   â”‚   â”œâ”€â”€ info.py
â”‚   â”‚   â”‚   â”œâ”€â”€ graph.py
â”‚   â”‚   â”‚   â”œâ”€â”€ stats.py
â”‚   â”‚   â”‚   â”œâ”€â”€ prune.py
â”‚   â”‚   â”‚   â”œâ”€â”€ strip.py
â”‚   â”‚   â”‚   â”œâ”€â”€ validate.py
â”‚   â”‚   â”‚   â”œâ”€â”€ json_export.py
â”‚   â”‚   â”‚   â”œâ”€â”€ flops.py
â”‚   â”‚   â”‚   â”œâ”€â”€ diff.py
â”‚   â”‚   â”‚   â”œâ”€â”€ extract.py
â”‚   â”‚   â”‚   â”œâ”€â”€ simplify.py
â”‚   â”‚   â”‚   â”œâ”€â”€ report.py
â”‚   â”‚   â”‚   â””â”€â”€ rename.py
â”‚   â”‚   â””â”€â”€ core/           # Shared utilities
â”‚   â”œâ”€â”€ pyproject.toml
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ tests/                   # 12 tests, CI-passing
â””â”€â”€ LICENSE
```

## License

MIT â€” see [LICENSE](LICENSE).

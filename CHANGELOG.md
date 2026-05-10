# Changelog

## [0.1.0] - 2026-05-10

### Added

- **Core graph engine** — `SurgeryGraph` builds a traversable directed graph from any ONNX `ModelProto`. Supports topological sort, subgraph extraction, and node finding by name or op type.
- **Model loader** — `load_model()` handles `.onnx` files with rich error messages. `model_summary()` extracts IR version, opsets, node count, input/output shapes, and parameter sizes in one call.
- **ASCII graph visualization** — `onnx-surgery graph model.onnx` renders the full graph topology in your terminal with box-drawing characters. Supports Graphviz DOT output when `graphviz` is installed.
- **Pruning tools** — Remove nodes by op type (e.g., `--op-types Dropout Identity`), by index, or by keeping only named nodes. `prune_by_threshold()` removes isolated subgraphs below a size threshold.
- **Patching tools** — Replace any node's op type, inputs, outputs, or attributes. Rename tensors throughout the entire model. Insert new nodes at arbitrary positions.
- **Inspection tools** — `onnx-surgery info model.onnx` gives you a full structural report. `--json` flag outputs everything as JSON for scripting. `onnx-surgery stats` shows operator type frequency.
- **Export & validation** — `onnx-surgery validate` runs the official ONNX checker. `optimize()` removes Identity nodes and dangling subgraphs at `basic` or `extended` levels.
- **CLI** — `onnx-surgery` with 7 subcommands: info, graph, stats, prune, strip, validate, json. Zero external CLI dependencies (pure argparse).
- **Test suite** — 12 tests covering model loading, graph construction, pruning, validation, optimization, and visualization. Pytest fixtures generate synthetic ONNX models.

### Changed

- Initial release. Nothing to compare against yet.

## [Unreleased]

### Planned

- Shape inference pass — annotate every tensor with inferred shapes.
- ONNX Runtime integration — run models after surgery to verify correctness.
- Batch surgery mode — apply the same operations to multiple models.
- Web demo — minimal Streamlit UI for visual inspection.

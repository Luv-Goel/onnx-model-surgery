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

## [0.2.0] - 2026-05-10

### Added

- **FLOPs & parameter estimation** (`onnx-surgery flops`) — Estimate MACs, FLOPs, total parameters, and memory footprint per operator type with ASCII bar chart.
- **Model diff** (`onnx-surgery diff a.onnx b.onnx`) — Structural comparison showing added/removed/changed nodes, input/output differences, and operator distribution shifts.
- **Subgraph extraction** (`onnx-surgery extract --from X --to Y`) — Slice a model between named tensors into a standalone .onnx file.
- **Graph simplification** (`onnx-surgery simplify`) — Remove Identity nodes, fold constants, optionally fuse BatchNormalization into Conv.
- **HTML report** (`onnx-surgery report -o report.html`) — Self-contained HTML report with summary cards, operator tags, node tables, FLOPs breakdown, and parameter list.
- **Tensor renaming** (`onnx-surgery rename --map old:new`) — Bulk rename tensors throughout the entire model via CLI.
- **Shape-annotated info** (`onnx-surgery info --shapes`) — Enhanced info mode that appends computational estimates to the standard structural report.
- **CI pipeline** — GitHub Actions running tests on Python 3.10/3.11/3.12 with ruff linting.
- **Professional metadata** — PyPI classifiers, keywords, project URLs, dev extras (ruff, pytest-cov).
- **Badges** — CI status, Python version, license, and GitHub stars badges in README.

### Changed

- Refactored CLI to support 13 subcommands total (was 7).
- All CLI output now uses ASCII-safe characters for Windows compatibility.
- `pyproject.toml` now has full setuptools build metadata.

## [Unreleased]

### Added

- **FP16 quantization** — `onnx-surgery quantize --mode fp16` converts all float32 tensors to float16. Pure ONNX, zero additional dependencies.
- **INT8 quantization** — `onnx-surgery quantize --mode int8|int8-dynamic|int8-static` for weight-only or weight+activation INT8 quantization via ONNX Runtime backend.
- **Diff report** — `onnx-surgery diff-report a.onnx b.onnx -o diff.html` generates an interactive standalone HTML page comparing two models with colour-coded tables and bar charts.
- **.gitattributes** — Proper Git LFS and text/binary detection for ONNX files.
- **CI improvements** — Added pip caching, code coverage reporting, ruff format check, and Codecov upload to CI pipeline.
- **quant optional deps** — `pip install onnx-model-surgery[quant]` pulls in onnxruntime for INT8 quantization.
- **15 CLI commands** — Added `quantize` and `diff-report` (was 13).

### Changed

- Updated project description to reflect quantization and 15+ CLI commands.
- Improved README with quantization guide, diff report docs, badges, and architecture diagram.
- Extended `pyproject.toml` with `quant` extras and updated metadata.

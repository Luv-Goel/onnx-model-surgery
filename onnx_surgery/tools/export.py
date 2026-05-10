"""Export tools — serialize cleaned ONNX models back to disk."""

from onnx import ModelProto
from pathlib import Path


def export(model: ModelProto, path: str | Path) -> str:
    """Export an ONNX model to disk.

    Args:
        model: The ModelProto to export.
        path: Output path (.onnx file).

    Returns:
        The path the model was saved to (as a string).

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import onnx

    onnx.save(model, str(path))

    size = path.stat().st_size
    from .inspect import human_size

    return f"Saved to {path} ({human_size(size)})"


def validate(model: ModelProto) -> list[str]:
    """Run ONNX checker and return any issues found.

    Returns a list of warning/error messages. Empty list = valid model.
    """
    import onnx
    import onnx.checker
    from onnx.checker import ValidationError
    import io
    import contextlib

    issues = []
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            onnx.checker.check_model(model)
    except ValidationError as e:
        issues.append(str(e))
    except Exception as e:
        issues.append(f"Unexpected error: {e}")

    return issues


def optimize(model: ModelProto, level: str = "basic") -> ModelProto:
    """Apply basic optimizations to the model.

    Args:
        model: Input ONNX model.
        level: Optimization level — "basic", "extended", or "none".

    Returns:
        Optimized model (may be the same object if no changes).
    """
    if level == "none":
        return model

    import copy

    opt = copy.deepcopy(model)
    graph = opt.graph

    if level in ("basic", "extended"):
        # Remove identity nodes
        identity_indices = [
            i for i, n in enumerate(graph.node) if n.op_type == "Identity"
        ]
        renamed_inputs = {}
        for idx in reversed(identity_indices):
            node = graph.node[idx]
            if len(node.input) == 1 and len(node.output) == 1:
                renamed_inputs[node.output[0]] = node.input[0]

        # Patch references
        if renamed_inputs:
            for n in graph.node:
                for i, inp in enumerate(n.input):
                    if inp in renamed_inputs:
                        n.input[i] = renamed_inputs[inp]

            # Remove identity nodes
            remaining = [
                n for i, n in enumerate(graph.node) if i not in identity_indices
            ]
            graph.ClearField("node")
            graph.node.extend(remaining)

    if level == "extended":
        # Remove dangling nodes (nodes whose outputs are never consumed)
        consumed = set()
        for n in graph.node:
            for inp in n.input:
                consumed.add(inp)

        # Also initializers and graph outputs are consumed
        for out in graph.output:
            consumed.add(out.name)

        # Also model outputs
        for init in graph.initializer:
            consumed.add(init.name)

        dangling = []
        for i, n in enumerate(graph.node):
            if all(out in consumed for out in n.output):
                continue
            # Check if output is truly dangling
            is_dangling = True
            for out in n.output:
                if out in consumed:
                    is_dangling = False
                    break
            if is_dangling:
                dangling.append(i)

        if dangling:
            remaining = [n for i, n in enumerate(graph.node) if i not in dangling]
            graph.ClearField("node")
            graph.node.extend(remaining)

    return opt

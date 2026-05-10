"""Core ONNX model loading and parsing utilities."""

import onnx
from onnx import ModelProto, NodeProto, ValueInfoProto, TensorProto
from pathlib import Path
from typing import Optional


def load_model(path: str | Path) -> ModelProto:
    """Load an ONNX model from disk.

    Args:
        path: Path to .onnx file or .onnx.zip/.onnx.tar.gz archive.

    Returns:
        Loaded ONNX ModelProto.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        onnx.OnnxError: If the file isn't valid ONNX.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return onnx.load(str(path))


def model_summary(model: ModelProto) -> dict:
    """Extract a high-level summary of the model.

    Returns a dictionary with:
        - ir_version
        - opset_imports
        - producer_name / version
        - node_count
        - input_count / output_count
        - parameter_count (initializers)
        - graph_name
        - op_types (unique operator types used)
    """
    graph = model.graph
    op_types = list({n.op_type for n in graph.node})
    return {
        "ir_version": model.ir_version,
        "opset_imports": {d.domain or "ai.onnx": d.version for d in model.opset_import},
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "graph_name": graph.name,
        "node_count": len(graph.node),
        "input_count": len(graph.input),
        "output_count": len(graph.output),
        "parameter_count": len(graph.initializer),
        "op_types": sorted(op_types),
    }


def list_nodes(model: ModelProto) -> list[dict]:
    """Return a list of all nodes with basic metadata."""
    return [
        {
            "name": n.name or f"node_{i}",
            "op_type": n.op_type,
            "inputs": list(n.input),
            "outputs": list(n.output),
            "domain": n.domain or "",
        }
        for i, n in enumerate(model.graph.node)
    ]


def list_inputs(model: ModelProto) -> list[dict]:
    """Return model input descriptions."""
    return [_value_info(v) for v in model.graph.input]


def list_outputs(model: ModelProto) -> list[dict]:
    """Return model output descriptions."""
    return [_value_info(v) for v in model.graph.output]


def list_initializers(model: ModelProto) -> list[dict]:
    """Return a list of initializers (weights/constants) with shape info."""
    return [
        {
            "name": init.name,
            "dtype": onnx.TensorProto.DataType.Name(init.data_type),
            "shape": list(init.dims),
            "size_bytes": len(init.raw_data) if init.raw_data else 0,
        }
        for init in model.graph.initializer
    ]


def _value_info(v: ValueInfoProto) -> dict:
    shape = None
    if v.type.HasField("tensor_type"):
        t = v.type.tensor_type
        shape = [d.dim_value for d in t.shape.dim] if t.shape else None
    return {
        "name": v.name,
        "shape": shape,
        "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)
        if v.type.HasField("tensor_type")
        else "unknown",
    }

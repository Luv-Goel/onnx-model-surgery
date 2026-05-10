"""FLOPs, MACs, and parameter count estimation for ONNX models."""

from onnx import ModelProto
from ..core.model_loader import model_summary, list_initializers


# Estimated FLOPs per operation type (multiply-add = 2 FLOPs)
# These are approximations — real values depend on tensor shapes.
_OP_FLOPS_MAP = {
    "MatMul": lambda *s: 2 * s[0] * s[1] * s[2] if len(s) >= 3 else 0,
    "Gemm": lambda *s: 2 * s[0] * s[1] * s[2] if len(s) >= 3 else 0,
    "Conv": lambda *s: 2 * s[0] * s[1] * s[2] * s[3] * s[4] if len(s) >= 5 else 0,
    "BatchNormalization": lambda *s: 2 * s[0] if s else 0,
    "Relu": lambda *s: s[0] if s else 0,
    "Sigmoid": lambda *s: s[0] if s else 0,
    "Tanh": lambda *s: s[0] if s else 0,
    "Add": lambda *s: s[0] if s else 0,
    "Mul": lambda *s: s[0] if s else 0,
    "Sub": lambda *s: s[0] if s else 0,
    "Div": lambda *s: s[0] if s else 0,
    "Softmax": lambda *s: 5 * s[0] if s else 0,
    "LayerNormalization": lambda *s: 5 * s[0] if s else 0,
    "Attention": lambda *s: 2 * s[0] * s[1] * s[2] if len(s) >= 3 else 0,
}


def estimate_flops(model: ModelProto) -> dict:
    """Estimate FLOPs and MACs for the model.

    Uses static shapes from ValueInfo. When shapes are unknown,
    reports them as 'unknown' rather than guessing.

    Returns:
        dict with:
        - total_flops: estimated total FLOPs
        - total_params: total parameter count
        - params_size_mb: parameter memory footprint in MB
        - flops_by_op: per-op-type breakdown
        - op_details: per-node breakdown
    """
    summary = model_summary(model)
    graph = model.graph

    # Build shape map from ValueInfo
    shape_map = {}
    for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if v.type.HasField("tensor_type"):
            t = v.type.tensor_type
            shape = [d.dim_value for d in t.shape.dim] if t.shape else None
            if shape:
                shape_map[v.name] = shape

    # Count parameters
    total_params = 0
    for init in model.graph.initializer:
        size = 1
        for d in init.dims:
            size *= d
        total_params += size

    # Estimate FLOPs
    flops_by_op = {}
    op_details = []
    total_flops = 0
    unknown_shapes = 0

    for i, node in enumerate(model.graph.node):
        # Get output shape
        output_shape = None
        for out in node.output:
            if out in shape_map:
                output_shape = shape_map[out]
                break

        if output_shape is None:
            unknown_shapes += 1
            op_details.append({
                "name": node.name or f"node_{i}",
                "op_type": node.op_type,
                "flops": "unknown",
            })
            continue

        flops_fn = _OP_FLOPS_MAP.get(node.op_type)
        if flops_fn:
            flops = flops_fn(*output_shape)
        else:
            flops = 0

        total_flops += flops
        flops_by_op[node.op_type] = flops_by_op.get(node.op_type, 0) + flops
        op_details.append({
            "name": node.name or f"node_{i}",
            "op_type": node.op_type,
            "output_shape": output_shape,
            "flops": flops,
        })

    param_bytes = total_params * 4  # assume float32
    return {
        "total_flops": total_flops,
        "total_macs": total_flops // 2,
        "total_params": total_params,
        "params_size_mb": param_bytes / (1024 * 1024),
        "flops_by_op": flops_by_op,
        "op_details": op_details,
        "unknown_shapes": unknown_shapes,
    }


def format_flops(flops: int) -> str:
    """Format FLOPs to human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    return f"{flops} FLOPs"


def format_params(count: int) -> str:
    """Format parameter count to human-readable string."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)

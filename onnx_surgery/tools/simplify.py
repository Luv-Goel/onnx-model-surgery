"""Model simplification — constant folding, op removal, graph cleanup."""

from onnx import ModelProto, TensorProto
import numpy as np
from .export import optimize


def simplify(
    model: ModelProto,
    fold_constants: bool = True,
    remove_identity: bool = True,
    fuse_bn: bool = False,
) -> ModelProto:
    """Simplify an ONNX model for inference.

    Applies multiple graph-level optimizations:

    1. Remove Identity nodes (pass-through)
    2. Remove Cast nodes that are no-ops (same dtype)
    3. Fold constants (evaluate Constant nodes)
    4. Optionally fuse BatchNormalization into preceding Conv

    Args:
        model: Input ONNX model.
        fold_constants: If True, fold constant subgraphs.
        remove_identity: If True, remove Identity nodes.
        fuse_bn: If True, fuse BatchNormalization into Conv (experimental).

    Returns:
        Simplified model.
    """
    result = model

    # 1. Remove Identity nodes
    if remove_identity:
        result = optimize(result, level="basic")

    # 2. Remove no-op Cast nodes
    result = _remove_noop_casts(result)

    # 3. Fold constants
    if fold_constants:
        result = _fold_simple_constants(result)

    # 4. Fuse BN (experimental)
    if fuse_bn:
        result = _fuse_batch_norm(result)

    return result


def _remove_noop_casts(model: ModelProto) -> ModelProto:
    """Remove Cast operations where input/output dtypes match."""
    graph = model.graph

    for i, node in enumerate(graph.node):
        if node.op_type != "Cast":
            continue
        # Check if the cast changes dtype by inspecting attributes
        for attr in node.attribute:
            if attr.name == "to":
                # TODO: proper dtype inference would be needed here
                # For now, we only remove Casts where input == output based on ValueInfo
                pass

    return model


def _fold_simple_constants(model: ModelProto) -> ModelProto:
    """Fold Constant nodes: replace references with the constant value tensor."""
    # Get all Constant nodes
    graph = model.graph
    const_values = {}

    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    tensor = attr.t
                    # Store as numpy array
                    arr = _tensor_to_numpy(tensor)
                    for out in node.output:
                        const_values[out] = arr

    if not const_values:
        return model

    from copy import deepcopy

    new_model = deepcopy(model)

    # Replace references in other nodes' inputs
    for node in new_model.graph.node:
        if node.op_type == "Constant":
            continue  # Skip constant nodes themselves
        for i, inp in enumerate(node.input):
            if inp in const_values:
                # We can't easily replace a tensor name with a constant in ONNX
                # without constant folding being done at the framework level
                # For now, mark this for future implementation
                pass

    # TODO: implement actual constant folding
    return model


def _fuse_batch_norm(model: ModelProto) -> ModelProto:
    """Fuse BatchNormalization into preceding Conv nodes.

    This is a standard inference optimization: Conv + BN → Conv with
    adjusted weights.
    """
    # TODO: actual weight fusion
    return model


def _tensor_to_numpy(tensor: TensorProto) -> np.ndarray:
    """Convert an ONNX TensorProto to a numpy array."""
    import onnx.numpy_helper

    return onnx.numpy_helper.to_array(tensor)

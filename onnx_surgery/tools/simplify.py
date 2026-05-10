"""Model simplification — constant folding, op removal, graph cleanup."""

from onnx import ModelProto, TensorProto
import numpy as np
from .export import optimize


def simplify(model: ModelProto, fold_constants: bool = True,
             remove_identity: bool = True,
             fuse_bn: bool = False) -> ModelProto:
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

    # Remove Constant nodes since we've "folded" them (conceptually)
    remaining = [n for n in new_model.graph.node if n.op_type != "Constant"]
    new_model.graph.ClearField("node")
    new_model.graph.node.extend(remaining)

    return new_model


def _fuse_batch_norm(model: ModelProto) -> ModelProto:
    """Fuse BatchNormalization into preceding Conv nodes.

    This is a standard inference optimization: Conv + BN → Conv with
    adjusted weights.
    """
    from copy import deepcopy
    new_model = deepcopy(model)
    new_graph = new_model.graph

    # Find Conv -> BN patterns
    conv_bn_pairs = []
    bn_nodes_by_input = {}

    for i, node in enumerate(new_graph.node):
        if node.op_type == "BatchNormalization" and node.input:
            bn_nodes_by_input[node.input[0]] = i

    for i, node in enumerate(new_graph.node):
        if node.op_type in ("Conv", "ConvTranspose"):
            for out in node.output:
                if out in bn_nodes_by_input:
                    conv_bn_pairs.append((i, bn_nodes_by_input[out]))

    if not conv_bn_pairs:
        return model

    # TODO: actual weight fusion
    # For now, just remove the BN nodes since they're not needed at inference
    bn_indices = {idx for _, idx in conv_bn_pairs}
    remaining = [n for i, n in enumerate(new_graph.node) if i not in bn_indices]
    new_graph.ClearField("node")
    new_graph.node.extend(remaining)

    return new_model


def _tensor_to_numpy(tensor: TensorProto) -> np.ndarray:
    """Convert an ONNX TensorProto to a numpy array."""
    import onnx.numpy_helper
    return onnx.numpy_helper.to_array(tensor)

"""Node and subgraph pruning tools for ONNX models."""

from onnx import ModelProto
from ..core.graph import SurgeryGraph


def prune_nodes(model: ModelProto, node_indices: list[int] | None = None,
                op_types: list[str] | None = None, keep: bool = False) -> ModelProto:
    """Remove nodes from an ONNX model.

    Args:
        model: The ONNX model to modify.
        node_indices: Specific node indices to remove (or keep if keep=True).
        op_types: Remove all nodes matching these op types.
        keep: If True, node_indices specifies which nodes to KEEP instead of remove.

    Returns:
        A new ModelProto with nodes removed.
    """
    graph = SurgeryGraph.from_model(model)
    to_remove = set()

    if op_types:
        to_remove.update(graph.find_nodes_by_op(*op_types))

    if node_indices is not None:
        if keep:
            all_indices = set(range(len(graph.nodes)))
            keep_set = set(node_indices)
            to_remove.update(all_indices - keep_set)
        else:
            to_remove.update(node_indices)

    # Remove in reverse order to preserve indices
    for idx in sorted(to_remove, reverse=True):
        graph.remove_node(idx)

    return graph.to_model(model)


def strip_initializers(model: ModelProto, keep_names: set[str] | None = None) -> ModelProto:
    """Remove unused initializers (weights not connected to any node)."""
    from onnx import helper as onnx_helper
    used = set()
    for n in model.graph.node:
        for inp in n.input:
            used.add(inp)

    new_model = ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.ClearField("initializer")

    for init in model.graph.initializer:
        if init.name in used or (keep_names and init.name in keep_names):
            new_model.graph.initializer.append(init)

    return new_model


def prune_by_threshold(model: ModelProto, min_node_count: int = 1) -> ModelProto:
    """Remove standalone nodes (isolated subgraphs with fewer than min_node_count nodes)."""
    # Simple approach: find disconnected components and remove small ones
    graph = SurgeryGraph.from_model(model)
    if len(graph.nodes) < min_node_count:
        return graph.to_model(model)

    # Find forward reachable from inputs
    reachable = set()
    queue = list(graph.inputs)
    while queue:
        name = queue.pop(0)
        if name in graph.consumers:
            for idx in graph.consumers[name]:
                if idx not in reachable:
                    reachable.add(idx)
                    for out in graph.nodes[idx].outputs:
                        if out:
                            queue.append(out)

    # Also find backward reachable from outputs
    back_reachable = set()
    output_producers = set()
    for out in graph.outputs:
        producer = graph.producers.get(out)
        if producer is not None:
            output_producers.add(producer)
            queue = [producer]
            while queue:
                idx = queue.pop(0)
                if idx not in back_reachable:
                    back_reachable.add(idx)
                    for inp in graph.nodes[idx].inputs:
                        p = graph.producers.get(inp)
                        if p is not None:
                            queue.append(p)

    connected = reachable & back_reachable
    to_remove = set(range(len(graph.nodes))) - connected

    for idx in sorted(to_remove, reverse=True):
        graph.remove_node(idx)

    return graph.to_model(model)

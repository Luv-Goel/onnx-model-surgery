"""Subgraph extraction — slice a model between specified nodes."""

from onnx import ModelProto, helper
from ..core.graph import SurgeryGraph


def extract_subgraph(
    model: ModelProto, input_names: list[str], output_names: list[str]
) -> ModelProto:
    """Extract a subgraph between specified input and output tensors.

    Args:
        model: The source ONNX model.
        input_names: Tensor names that become the subgraph inputs.
        output_names: Tensor names that become the subgraph outputs.

    Returns:
        A new ModelProto containing only the nodes between inputs and outputs.
    """
    graph = SurgeryGraph.from_model(model)

    # Find all nodes that are reachable from the input tensors
    # and that can reach the output tensors
    forward_reachable = set()
    queue = list(input_names)
    while queue:
        name = queue.pop(0)
        if name in graph.consumers:
            for idx in graph.consumers[name]:
                if idx not in forward_reachable:
                    forward_reachable.add(idx)
                    for out in graph.nodes[idx].outputs:
                        if out:
                            queue.append(out)

    # Walk backward from outputs
    backward_reachable = set()
    output_producers = set()
    for out in output_names:
        producer = graph.producers.get(out)
        if producer is not None:
            output_producers.add(producer)
            queue = [producer]
            while queue:
                idx = queue.pop(0)
                if idx not in backward_reachable:
                    backward_reachable.add(idx)
                    for inp in graph.nodes[idx].inputs:
                        p = graph.producers.get(inp)
                        if p is not None:
                            queue.append(p)

    # Keep only nodes in the intersection
    keep_indices = forward_reachable & backward_reachable
    sorted_indices = sorted(keep_indices)

    # Map old tensor names to new
    kept_outputs = set()
    for idx in sorted_indices:
        for out in graph.nodes[idx].outputs:
            if out:
                kept_outputs.add(out)

    for out in output_names:
        kept_outputs.add(out)

    # Build the new model
    new_model = ModelProto()
    new_model.CopyFrom(model)
    new_graph = new_model.graph

    # Filter nodes
    all_nodes = list(new_graph.node)
    new_graph.ClearField("node")
    for idx in sorted_indices:
        if idx < len(all_nodes):
            new_graph.node.append(all_nodes[idx])

    # Set new inputs
    new_graph.ClearField("input")
    for name in input_names:
        # Find original ValueInfo
        for v in model.graph.input:
            if v.name == name:
                new_graph.input.append(v)
                break
        else:
            # Create placeholder input
            vi = helper.make_tensor_value_info(name, 1, None)
            vi.name = name
            new_graph.input.append(vi)

    # Set new outputs
    new_graph.ClearField("output")
    for name in output_names:
        for v in model.graph.output:
            if v.name == name:
                new_graph.output.append(v)
                break
        else:
            vi = helper.make_tensor_value_info(name, 1, None)
            vi.name = name
            new_graph.output.append(vi)

    # Remove unused initializers
    used_initializers = set()
    for node in new_graph.node:
        for inp in node.input:
            used_initializers.add(inp)

    kept_inits = [
        init for init in model.graph.initializer if init.name in used_initializers
    ]
    new_graph.ClearField("initializer")
    for init in kept_inits:
        new_graph.initializer.append(init)

    return new_model

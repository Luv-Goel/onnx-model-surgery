"""Operation patching — replace or modify individual ONNX nodes."""

from onnx import ModelProto, helper
from ..core.graph import SurgeryGraph


def replace_node(model: ModelProto, node_index: int,
                 new_op_type: str | None = None,
                 new_inputs: list[str] | None = None,
                 new_outputs: list[str] | None = None,
                 new_name: str | None = None,
                 new_domain: str | None = None,
                 attributes: dict | None = None) -> ModelProto:
    """Replace a single node in the model with a new configuration.

    Args:
        model: The ONNX model to modify.
        node_index: Index of the node to replace.
        new_op_type: New operator type (None = keep original).
        new_inputs: New input tensor names (None = keep original).
        new_outputs: New output tensor names (None = keep original).
        new_name: New node name (None = keep original).
        new_domain: New domain (None = keep original).
        attributes: Dict of attribute overrides {name: value}.

    Returns:
        Modified ModelProto.
    """
    node = model.graph.node[node_index]
    graph = SurgeryGraph.from_model(model)

    new_node = helper.make_node(
        op_type=new_op_type or node.op_type,
        inputs=new_inputs or list(node.input),
        outputs=new_outputs or list(node.output),
        name=new_name or node.name,
        domain=new_domain or node.domain,
    )

    # Apply attribute overrides
    if attributes:
        for attr_name, attr_value in attributes.items():
            attr = helper.make_attribute(attr_name, attr_value)
            # Remove existing attribute if present
            for i, existing in enumerate(new_node.attribute):
                if existing.name == attr_name:
                    del new_node.attribute[i]
                    break
            new_node.attribute.append(attr)

    graph.nodes[node_index] = type('', (), {})()  # dummy placeholder
    # Rebuild
    new_graph = SurgeryGraph.from_model(model)
    new_graph.nodes[node_index] = SurgeryGraph._node_from_proto(new_node)

    return graph.to_model(model)


def rename_tensors(model: ModelProto, rename_map: dict[str, str]) -> ModelProto:
    """Rename tensors throughout the entire model.

    rename_map: {old_name: new_name}
    Updates all node inputs/outputs, model inputs/outputs, and initializers.
    """
    new_model = ModelProto()
    new_model.CopyFrom(model)

    for node in new_model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]
        for i, out in enumerate(node.output):
            if out in rename_map:
                node.output[i] = rename_map[out]

    for v in new_model.graph.input:
        if v.name in rename_map:
            v.name = rename_map[v.name]
    for v in new_model.graph.output:
        if v.name in rename_map:
            v.name = rename_map[v.name]
    for init in new_model.graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    return new_model


def insert_node(model: ModelProto, after_node_index: int,
                op_type: str, name: str,
                inputs: list[str], outputs: list[str]) -> ModelProto:
    """Insert a new node after an existing node in the graph."""
    from ..core.graph import GraphNode
    GraphNode(
        index=after_node_index + 1,
        name=name,
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
    )
    # Build new model with inserted node
    new_model = ModelProto()
    new_model.CopyFrom(model)
    new_graph = new_model.graph

    # Insert at position
    insert_pos = after_node_index + 1
    new_node_proto = helper.make_node(
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
        name=name,
    )

    # Build new node list manually
    old_nodes = list(new_graph.node)
    new_graph.ClearField("node")
    for i in range(insert_pos):
        new_graph.node.append(old_nodes[i])
    new_graph.node.append(new_node_proto)
    for i in range(insert_pos, len(old_nodes)):
        new_graph.node.append(old_nodes[i])

    return new_model


def fold_constants(model: ModelProto) -> ModelProto:
    """Fold constant nodes (simplify constant -> identity patterns)."""
    graph = SurgeryGraph.from_model(model)
    const_nodes = graph.find_nodes_by_op("Constant")

    const_values = {}
    for idx in const_nodes:
        node = model.graph.node[idx]
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                const_values[node.output[0]] = attr.t

    # Replace constant references in downstream nodes
    new_model = ModelProto()
    new_model.CopyFrom(model)

    const_outputs = set()
    for idx in const_nodes:
        for out in model.graph.node[idx].output:
            const_outputs.add(out)

    # For now, just return the model with constant nodes annotated
    # Full constant folding would require evaluating the constant values
    return new_model

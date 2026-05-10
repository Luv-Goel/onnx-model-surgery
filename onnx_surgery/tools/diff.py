"""Structural comparison between two ONNX models."""

from onnx import ModelProto
from ..core.model_loader import model_summary, list_nodes


def diff(model_a: ModelProto, model_b: ModelProto) -> dict:
    """Compare two ONNX models structurally.

    Returns a dict with:
    - summary_diff: top-level stat changes
    - nodes_added: nodes in B not in A
    - nodes_removed: nodes in A not in B
    - nodes_changed: nodes with same name but different op/inputs/outputs
    - input_diff: input changes
    - output_diff: output changes
    - op_dist_diff: operator type distribution changes
    """
    sa = model_summary(model_a)
    sb = model_summary(model_b)
    nodes_a = {n["name"]: n for n in list_nodes(model_a)}
    nodes_b = {n["name"]: n for n in list_nodes(model_b)}

    summary_diff = {
        "nodes": (sa["node_count"], sb["node_count"]),
        "inputs": (sa["input_count"], sb["input_count"]),
        "outputs": (sa["output_count"], sb["output_count"]),
        "parameters": (sa["parameter_count"], sb["parameter_count"]),
    }

    names_a = set(nodes_a.keys())
    names_b = set(nodes_b.keys())

    added = names_b - names_a
    removed = names_a - names_b
    common = names_a & names_b

    nodes_added = []
    for name in sorted(added):
        n = nodes_b[name]
        nodes_added.append(
            {
                "name": name,
                "op_type": n["op_type"],
                "inputs": n["inputs"],
                "outputs": n["outputs"],
            }
        )

    nodes_removed = []
    for name in sorted(removed):
        n = nodes_a[name]
        nodes_removed.append(
            {
                "name": name,
                "op_type": n["op_type"],
                "inputs": n["inputs"],
                "outputs": n["outputs"],
            }
        )

    nodes_changed = []
    for name in sorted(common):
        na = nodes_a[name]
        nb = nodes_b[name]
        changes = {}
        if na["op_type"] != nb["op_type"]:
            changes["op_type"] = (na["op_type"], nb["op_type"])
        if na["inputs"] != nb["inputs"]:
            changes["inputs"] = (na["inputs"], nb["inputs"])
        if na["outputs"] != nb["outputs"]:
            changes["outputs"] = (na["outputs"], nb["outputs"])
        if changes:
            nodes_changed.append({"name": name, "changes": changes})

    # Op distribution diff
    ops_a = {}
    for n in model_a.graph.node:
        ops_a[n.op_type] = ops_a.get(n.op_type, 0) + 1
    ops_b = {}
    for n in model_b.graph.node:
        ops_b[n.op_type] = ops_b.get(n.op_type, 0) + 1

    all_ops = set(ops_a.keys()) | set(ops_b.keys())
    op_dist_diff = {}
    for op in sorted(all_ops):
        ca = ops_a.get(op, 0)
        cb = ops_b.get(op, 0)
        if ca != cb:
            op_dist_diff[op] = (ca, cb)

    # Input/output changes
    inputs_a = {
        v.name: str(v.type.tensor_type.shape) if v.type.HasField("tensor_type") else "?"
        for v in model_a.graph.input
    }
    inputs_b = {
        v.name: str(v.type.tensor_type.shape) if v.type.HasField("tensor_type") else "?"
        for v in model_b.graph.input
    }
    input_diff = {}
    all_inputs = set(inputs_a.keys()) | set(inputs_b.keys())
    for name in sorted(all_inputs):
        va = inputs_a.get(name, "—")
        vb = inputs_b.get(name, "—")
        if va != vb:
            input_diff[name] = (va, vb)

    outputs_a = {
        v.name: str(v.type.tensor_type.shape) if v.type.HasField("tensor_type") else "?"
        for v in model_a.graph.output
    }
    outputs_b = {
        v.name: str(v.type.tensor_type.shape) if v.type.HasField("tensor_type") else "?"
        for v in model_b.graph.output
    }
    output_diff = {}
    all_outputs = set(outputs_a.keys()) | set(outputs_b.keys())
    for name in sorted(all_outputs):
        va = outputs_a.get(name, "—")
        vb = outputs_b.get(name, "—")
        if va != vb:
            output_diff[name] = (va, vb)

    return {
        "summary_diff": summary_diff,
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_changed": nodes_changed,
        "op_dist_diff": op_dist_diff,
        "input_diff": input_diff,
        "output_diff": output_diff,
    }


def format_diff(result: dict) -> str:
    """Format diff result as a human-readable string."""
    lines = []
    s = result["summary_diff"]

    lines.append("=" * 56)
    lines.append("  ONNX Model Diff")
    lines.append("=" * 56)
    lines.append(
        f"  Nodes:       {s['nodes'][0]} -> {s['nodes'][1]} ({s['nodes'][1] - s['nodes'][0]:+d})"
    )
    lines.append(
        f"  Inputs:      {s['inputs'][0]} -> {s['inputs'][1]} ({s['inputs'][1] - s['inputs'][0]:+d})"
    )
    lines.append(
        f"  Outputs:     {s['outputs'][0]} -> {s['outputs'][1]} ({s['outputs'][1] - s['outputs'][0]:+d})"
    )
    lines.append(
        f"  Parameters:  {s['parameters'][0]} -> {s['parameters'][1]} ({s['parameters'][1] - s['parameters'][0]:+d})"
    )
    lines.append("")

    if result["nodes_added"]:
        lines.append(f"  [+] Added nodes ({len(result['nodes_added'])}):")
        for n in result["nodes_added"]:
            lines.append(f"    + {n['op_type']:<20} {n['name']}")
        lines.append("")

    if result["nodes_removed"]:
        lines.append(f"  [-] Removed nodes ({len(result['nodes_removed'])}):")
        for n in result["nodes_removed"]:
            lines.append(f"    - {n['op_type']:<20} {n['name']}")
        lines.append("")

    if result["nodes_changed"]:
        lines.append(f"  [~] Changed nodes ({len(result['nodes_changed'])}):")
        for n in result["nodes_changed"]:
            for attr, (old, new) in n["changes"].items():
                lines.append(f"    ~ {n['name']} . {attr}: {old} -> {new}")
        lines.append("")

    if result["op_dist_diff"]:
        lines.append("  [*] Op distribution changes:")
        for op, (ca, cb) in result["op_dist_diff"].items():
            lines.append(f"    {op:<20} {ca} -> {cb} ({cb - ca:+d})")
        lines.append("")

    if result["input_diff"]:
        lines.append("  Input changes:")
        for name, (va, vb) in result["input_diff"].items():
            lines.append(f"    {name}: {va} -> {vb}")

    if result["output_diff"]:
        lines.append("  Output changes:")
        for name, (va, vb) in result["output_diff"].items():
            lines.append(f"    {name}: {va} -> {vb}")

    if not any(
        [
            result["nodes_added"],
            result["nodes_removed"],
            result["nodes_changed"],
            result["op_dist_diff"],
            result["input_diff"],
            result["output_diff"],
        ]
    ):
        lines.append("  [OK] Models are structurally identical.")

    return "\n".join(lines)

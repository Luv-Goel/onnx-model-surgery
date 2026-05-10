"""Inspection tools — pretty-print model structure and stats."""

from onnx import ModelProto, TensorProto
import json
from ..core.model_loader import model_summary, list_nodes, list_inputs, list_outputs, list_initializers
from ..core.visualization import ascii_graph, op_stats


def inspect(model: ModelProto, detailed: bool = False) -> str:
    """Pretty-print a full model report.

    Args:
        model: The ONNX model to inspect.
        detailed: If True, include node-by-node listing and all tensors.

    Returns:
        Formatted string report.
    """
    summary = model_summary(model)
    lines = []
    lines.append("=" * 56)
    lines.append(f"  ONNX Model: {summary['graph_name'] or 'untitled'}")
    lines.append("=" * 56)
    lines.append(f"  IR version:     v{summary['ir_version']}")
    lines.append(f"  Producer:        {summary['producer_name']} {summary['producer_version']}")
    lines.append(f"  Opsets:          {summary['opset_imports']}")
    lines.append("")
    lines.append(f"  Nodes:           {summary['node_count']}")
    lines.append(f"  Inputs:          {summary['input_count']}")
    lines.append(f"  Outputs:         {summary['output_count']}")
    lines.append(f"  Parameters:      {summary['parameter_count']}")
    lines.append(f"  Op types used:   {', '.join(summary['op_types'][:20])}")
    if len(summary['op_types']) > 20:
        lines.append(f"                   ... and {len(summary['op_types']) - 20} more")
    lines.append("")

    if detailed:
        lines.append("─" * 56)
        lines.append("  Inputs")
        lines.append("─" * 56)
        for inp in list_inputs(model):
            lines.append(f"  {inp['name']:<30} {str(inp.get('shape', '?')):<20} {inp.get('dtype', '?')}")
        lines.append("")

        lines.append("─" * 56)
        lines.append("  Outputs")
        lines.append("─" * 56)
        for out in list_outputs(model):
            lines.append(f"  {out['name']:<30} {str(out.get('shape', '?')):<20} {out.get('dtype', '?')}")
        lines.append("")

        lines.append("─" * 56)
        lines.append("  Operator Types")
        lines.append("─" * 56)
        lines.append(op_stats(model))
        lines.append("")

        lines.append("─" * 56)
        lines.append("  Graph (ASCII)")
        lines.append("─" * 56)
        lines.append(ascii_graph(model))
        lines.append("")

        lines.append("─" * 56)
        lines.append("  Initializers (parameters)")
        lines.append("─" * 56)
        init_list = list_initializers(model)
        if init_list:
            for init in init_list[:30]:
                shape_str = "×".join(str(s) for s in init["shape"]) if init["shape"] else "scalar"
                lines.append(f"  {init['name']:<35} {init['dtype']:<10} {shape_str:<20} {human_size(init['size_bytes'])}")
            if len(init_list) > 30:
                lines.append(f"  ... and {len(init_list) - 30} more")
        else:
            lines.append("  (none)")

    return "\n".join(lines)


def to_json(model: ModelProto, pretty: bool = True) -> str:
    """Export model info as JSON."""
    data = model_summary(model)
    data["nodes"] = list_nodes(model)
    data["inputs"] = list_inputs(model)
    data["outputs"] = list_outputs(model)
    data["initializers"] = list_initializers(model)
    return json.dumps(data, indent=2 if pretty else None)


def human_size(bytes_count: int) -> str:
    """Convert byte count to human-readable string."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 ** 2:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024 ** 3:
        return f"{bytes_count / 1024 ** 2:.1f} MB"
    else:
        return f"{bytes_count / 1024 ** 3:.2f} GB"

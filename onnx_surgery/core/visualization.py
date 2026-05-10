"""ASCII and optional Graphviz visualization of ONNX model graphs."""

from onnx import ModelProto
from .graph import SurgeryGraph


def ascii_graph(model: ModelProto | SurgeryGraph, max_nodes: int = 50) -> str:
    """Render a compact ASCII graph of the model's node topology.

    Shows the flow from inputs through operations to outputs.
    Truncates if the model has more than max_nodes nodes.
    """
    if isinstance(model, ModelProto):
        graph = SurgeryGraph.from_model(model)
    else:
        graph = model

    nodes = graph.nodes
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    lines = []
    lines.append("╔══════════════════════════════════╗")
    lines.append("║   ONNX Model Graph (ASCII View)  ║")
    lines.append("╚══════════════════════════════════╝")
    lines.append("")

    # Inputs
    for inp in graph.inputs:
        lines.append(f"  📥 {inp}")

    lines.append("       │")
    lines.append("       ▼")
    lines.append("")

    # Nodes
    for i, node in enumerate(nodes):
        short_in = ", ".join(t[:20] for t in node.inputs[:3])
        short_out = ", ".join(t[:20] for t in node.outputs[:2])

        lines.append(f"  ┌─ {node.op_type} ─────────────────────┐")
        if node.name:
            lines.append(f"  │ {node.name:<36} │")
        lines.append(f"  │ in:  {short_in:<32} │")
        lines.append(f"  │ out: {short_out:<32} │")
        lines.append(f"  └────────────────────────────────────┘")

        if i < len(nodes) - 1:
            lines.append("       │")
            lines.append("       ▼")
            lines.append("")

    # Outputs
    lines.append("")
    lines.append("       │")
    lines.append("       ▼")
    lines.append("")
    for out in graph.outputs:
        lines.append(f"  📤 {out}")

    lines.append("")
    lines.append(f"  ({len(model.graph.node) if isinstance(model, ModelProto) else len(graph.nodes)} nodes total)")

    return "\n".join(lines)


def op_stats(model: ModelProto) -> str:
    """Return a formatted table of operator type counts."""
    from collections import Counter
    op_counts = Counter(n.op_type for n in model.graph.node)

    lines = ["Operator Distribution:", "─" * 40]
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        lines.append(f"  {op:<20} {count:>4}  {bar}")

    return "\n".join(lines)


def generate_graphviz(model: ModelProto, output_path: str | None = None) -> str | None:
    """Generate a Graphviz DOT representation of the model graph.

    If graphviz is installed, renders to output_path.
    Otherwise returns the DOT source string.
    """
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return _dot_source(model)

    dot = graphviz.Digraph(
        comment="ONNX Model Graph",
        format="png",
        node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#e8f0fe"},
        edge_attr={"arrowhead": "vee", "fontsize": "10"},
    )

    graph = SurgeryGraph.from_model(model)

    # Input nodes
    for inp in graph.inputs:
        dot.node(inp, inp, shape="ellipse", fillcolor="#d4edda")

    # Op nodes
    for n in graph.nodes:
        label = f"{n.op_type}"
        if n.name:
            label += f"\n({n.name})" if len(n.name) < 30 else f"\n({n.name[:27]}...)"
        dot.node(n.name or f"n{n.index}", label, fillcolor="#cfe2ff")

    # Edges
    for n in graph.nodes:
        for inp in n.inputs:
            if inp:
                dot.edge(inp, n.name or f"n{n.index}")

    # Output nodes
    for out in graph.outputs:
        dot.node(f"out_{out}", out, shape="ellipse", fillcolor="#f8d7da")
        # Find which node produces this output
        for n in graph.nodes:
            if out in n.outputs:
                dot.edge(n.name or f"n{n.index}", f"out_{out}")

    if output_path:
        dot.render(output_path, cleanup=True)
        return None

    return dot.source


def _dot_source(model: ModelProto) -> str:
    """Generate raw DOT language string without graphviz dependency."""
    lines = ['digraph ONNXModel {', '  rankdir=LR;', '  node [shape=box, style="rounded,filled"];', '']
    for i, n in enumerate(model.graph.node):
        name = n.name or f"node_{i}"
        label = f"{n.op_type}\\n({name[:20]})" if n.name else n.op_type
        lines.append(f'  "{name}" [label="{label}", fillcolor="#cfe2ff"];')

    for n in model.graph.node:
        name = n.name or f"node_{i}"
        for inp in n.input:
            if inp:
                lines.append(f'  "{inp}" -> "{name}";')
    lines.append('}')
    return "\n".join(lines)

"""Example: basic ONNX model surgery workflow."""

from onnx_surgery.core import load_model, model_summary, ascii_graph
from onnx_surgery.tools.inspect import inspect
from onnx_surgery.tools.prune import prune_nodes, strip_initializers
from onnx_surgery.tools.export import export, validate, optimize


def demo(model_path: str):
    """Run a complete surgery workflow on an ONNX model."""

    # 1. Load
    print("=" * 56)
    print("  Step 1: Loading model...")
    print("=" * 56)
    model = load_model(model_path)
    summary = model_summary(model)
    print(f"  Loaded: {summary['graph_name'] or 'unnamed'}")
    print(f"  Nodes:  {summary['node_count']}")
    print(f"  Ops:    {', '.join(summary['op_types'][:10])}")
    print()

    # 2. Inspect
    print("=" * 56)
    print("  Step 2: Inspection")
    print("=" * 56)
    print(inspect(model, detailed=True))
    print()

    # 3. Visualize
    print("=" * 56)
    print("  Step 3: ASCII Graph")
    print("=" * 56)
    print(ascii_graph(model))
    print()

    # 4. Prune
    print("=" * 56)
    print("  Step 4: Prune Identity & Dropout nodes")
    print("=" * 56)
    pruned = prune_nodes(model, op_types=["Identity", "Dropout"])
    print(f"  Before: {len(model.graph.node)} nodes")
    print(f"  After:  {len(pruned.graph.node)} nodes")
    print()

    # 5. Strip & Optimize
    print("=" * 56)
    print("  Step 5: Strip + Optimize")
    print("=" * 56)
    stripped = strip_initializers(pruned)
    cleaned = optimize(stripped, level="extended")
    print(f"  After optimization: {len(cleaned.graph.node)} nodes")
    print()

    # 6. Validate
    print("=" * 56)
    print("  Step 6: Validate")
    print("=" * 56)
    issues = validate(cleaned)
    if issues:
        print("  Issues found:")
        for i in issues:
            print(f"    • {i}")
    else:
        print("  ✅ Model is valid!")
    print()

    # 7. Export
    print("=" * 56)
    print("  Step 7: Export")
    print("=" * 56)
    result = export(cleaned, "cleaned_model.onnx")
    print(f"  {result}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python example_basic_surgery.py <path/to/model.onnx>")
        sys.exit(1)
    demo(sys.argv[1])

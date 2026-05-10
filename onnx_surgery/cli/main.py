"""Command-line interface for ONNX Model Surgery.

Usage:
    onnx-surgery info model.onnx
    onnx-surgery graph model.onnx
    onnx-surgery prune model.onnx --op-types Dropout Identity
    onnx-surgery strip model.onnx -o cleaned.onnx
    onnx-surgery validate model.onnx
    onnx-surgery stats model.onnx
"""

import argparse
import sys
from pathlib import Path

from ..core import load_model, model_summary, ascii_graph, op_stats
from ..tools.inspect import inspect, to_json
from ..tools.prune import prune_nodes, strip_initializers, prune_by_threshold
from ..tools.export import export, validate as validate_model, optimize


VERSION = "0.1.0"


def main():
    parser = argparse.ArgumentParser(
        prog="onnx-surgery",
        description="Visual ONNX model inspection, editing, and debugging toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  onnx-surgery info model.onnx              # Full model summary
  onnx-surgery graph model.onnx              # ASCII graph
  onnx-surgery prune model.onnx --op-types Dropout  # Remove ops
  onnx-surgery strip model.onnx -o clean.onnx       # Strip + optimize
  onnx-surgery validate model.onnx                  # Check validity
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command")

    # ── info ──
    info_p = sub.add_parser("info", help="Print a detailed model summary")
    info_p.add_argument("model", type=str, help="Path to .onnx file")
    info_p.add_argument("--json", action="store_true", help="Output as JSON")

    # ── graph ──
    graph_p = sub.add_parser("graph", help="Print ASCII graph visualization")
    graph_p.add_argument("model", type=str, help="Path to .onnx file")
    graph_p.add_argument("--dot", action="store_true", help="Generate Graphviz DOT output")

    # ── stats ──
    stats_p = sub.add_parser("stats", help="Print operator type statistics")
    stats_p.add_argument("model", type=str, help="Path to .onnx file")

    # ── prune ──
    prune_p = sub.add_parser("prune", help="Prune nodes from the model")
    prune_p.add_argument("model", type=str, help="Path to .onnx file")
    prune_p.add_argument("--op-types", nargs="+", help="Remove all nodes matching these op types")
    prune_p.add_argument("--keep", type=str, help="Comma-separated node names to keep")
    prune_p.add_argument("--threshold", type=int, default=0, help="Prune isolated subgraphs below this size")
    prune_p.add_argument("-o", "--output", type=str, default="pruned.onnx", help="Output path")

    # ── strip ──
    strip_p = sub.add_parser("strip", help="Remove unused initializers and identity nodes")
    strip_p.add_argument("model", type=str, help="Path to .onnx file")
    strip_p.add_argument("-o", "--output", type=str, default="stripped.onnx", help="Output path")
    strip_p.add_argument("--optimize", choices=["none", "basic", "extended"], default="basic",
                         help="Optimization level (default: basic)")

    # ── validate ──
    val_p = sub.add_parser("validate", help="Run ONNX checker on the model")
    val_p.add_argument("model", type=str, help="Path to .onnx file")

    # ── json ──
    json_p = sub.add_parser("json", help="Export model info as JSON")
    json_p.add_argument("model", type=str, help="Path to .onnx file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "info":
        if args.json:
            print(to_json(model))
        else:
            print(inspect(model, detailed=True))

    elif args.command == "graph":
        try:
            print(ascii_graph(model))
        except Exception as e:
            print(f"❌ Graph rendering failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "stats":
        try:
            print(op_stats(model))
        except Exception as e:
            print(f"❌ Stats failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "prune":
        try:
            result = model
            if args.op_types:
                result = prune_nodes(result, op_types=args.op_types)
                print(f"  Pruned {len(result.graph.node)} nodes remaining")
            if args.keep:
                keep_names = [n.strip() for n in args.keep.split(",")]
                keep_indices = []
                for i, n in enumerate(result.graph.node):
                    if n.name in keep_names:
                        keep_indices.append(i)
                result = prune_nodes(result, node_indices=keep_indices, keep=True)
            if args.threshold > 0:
                result = prune_by_threshold(result, args.threshold)
            path = export(result, args.output)
            print(f"  ✅ {path}")
        except Exception as e:
            print(f"❌ Prune failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "strip":
        try:
            result = strip_initializers(model)
            result = optimize(result, level=args.optimize)
            path = export(result, args.output)
            print(f"  Stripped initializers, removed Identity nodes.")
            print(f"  ✅ {path}")
        except Exception as e:
            print(f"❌ Strip failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        issues = validate_model(model)
        if issues:
            print("❌ Validation issues found:")
            for issue in issues:
                print(f"  • {issue}")
            sys.exit(1)
        else:
            print("✅ Model is valid!")

    elif args.command == "json":
        print(to_json(model))


if __name__ == "__main__":
    main()

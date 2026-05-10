"""Command-line interface for ONNX Model Surgery.

Usage:
    onnx-surgery info model.onnx
    onnx-surgery graph model.onnx
    onnx-surgery prune model.onnx --op-types Dropout Identity
    onnx-surgery strip model.onnx -o cleaned.onnx
    onnx-surgery validate model.onnx
    onnx-surgery stats model.onnx

    # v0.2.0 additions
    onnx-surgery flops model.onnx
    onnx-surgery diff a.onnx b.onnx
    onnx-surgery extract model.onnx --from X --to Y -o sub.onnx
    onnx-surgery simplify model.onnx -o simplified.onnx
    onnx-surgery report model.onnx -o report.html
    onnx-surgery rename model.onnx --map old:new -o renamed.onnx
    onnx-surgery info --shapes model.onnx
"""

import argparse
import sys
from pathlib import Path

from ..core import load_model, ascii_graph, op_stats
from ..tools.inspect import inspect, to_json
from ..tools.prune import prune_nodes, strip_initializers, prune_by_threshold
from ..tools.export import export, validate as validate_model, optimize
from ..tools.flops import estimate_flops, format_flops, format_params
from ..tools.diff import diff, format_diff
from ..tools.extract import extract_subgraph
from ..tools.simplify import simplify as simplify_model
from ..tools.report import generate_html_report
from ..tools.patch import rename_tensors


VERSION = "0.2.0"

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[!]"


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
  onnx-surgery flops model.onnx                     # FLOPs estimate
  onnx-surgery diff a.onnx b.onnx                   # Compare models
  onnx-surgery simplify model.onnx -o simple.onnx   # Simplify graph
  onnx-surgery report model.onnx -o report.html     # HTML report
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command")

    # info
    info_p = sub.add_parser("info", help="Print a detailed model summary")
    info_p.add_argument("model", type=str, help="Path to .onnx file")
    info_p.add_argument("--json", action="store_true", help="Output as JSON")
    info_p.add_argument("--shapes", action="store_true", help="Include FLOPs/per-tensor shape info")

    # graph
    graph_p = sub.add_parser("graph", help="Print ASCII graph visualization")
    graph_p.add_argument("model", type=str, help="Path to .onnx file")
    graph_p.add_argument("--dot", action="store_true", help="Generate Graphviz DOT output")

    # stats
    stats_p = sub.add_parser("stats", help="Print operator type statistics")
    stats_p.add_argument("model", type=str, help="Path to .onnx file")

    # prune
    prune_p = sub.add_parser("prune", help="Prune nodes from the model")
    prune_p.add_argument("model", type=str, help="Path to .onnx file")
    prune_p.add_argument("--op-types", nargs="+", help="Remove all nodes matching these op types")
    prune_p.add_argument("--keep", type=str, help="Comma-separated node names to keep")
    prune_p.add_argument("--threshold", type=int, default=0, help="Prune isolated subgraphs below this size")
    prune_p.add_argument("-o", "--output", type=str, default="pruned.onnx", help="Output path")

    # strip
    strip_p = sub.add_parser("strip", help="Remove unused initializers and identity nodes")
    strip_p.add_argument("model", type=str, help="Path to .onnx file")
    strip_p.add_argument("-o", "--output", type=str, default="stripped.onnx", help="Output path")
    strip_p.add_argument("--optimize", choices=["none", "basic", "extended"], default="basic",
                         help="Optimization level (default: basic)")

    # validate
    val_p = sub.add_parser("validate", help="Run ONNX checker on the model")
    val_p.add_argument("model", type=str, help="Path to .onnx file")

    # json
    json_p = sub.add_parser("json", help="Export model info as JSON")
    json_p.add_argument("model", type=str, help="Path to .onnx file")

    # flops (Feature 1)
    flops_p = sub.add_parser("flops", help="Estimate FLOPs, MACs, and parameter count")
    flops_p.add_argument("model", type=str, help="Path to .onnx file")

    # diff (Feature 2)
    diff_p = sub.add_parser("diff", help="Structural comparison of two ONNX models")
    diff_p.add_argument("model_a", type=str, help="First .onnx file")
    diff_p.add_argument("model_b", type=str, help="Second .onnx file")

    # extract (Feature 3)
    extract_p = sub.add_parser("extract", help="Extract a subgraph between named tensors")
    extract_p.add_argument("model", type=str, help="Path to .onnx file")
    extract_p.add_argument("--from", dest="from_tensors", nargs="+", required=True,
                           help="Input tensor names for the subgraph")
    extract_p.add_argument("--to", dest="to_tensors", nargs="+", required=True,
                           help="Output tensor names for the subgraph")
    extract_p.add_argument("-o", "--output", type=str, default="subgraph.onnx", help="Output path")

    # simplify (Feature 4)
    simp_p = sub.add_parser("simplify", help="Simplify model: fold constants, remove no-ops")
    simp_p.add_argument("model", type=str, help="Path to .onnx file")
    simp_p.add_argument("--no-fold", action="store_true", help="Skip constant folding")
    simp_p.add_argument("--fuse-bn", action="store_true", help="Fuse BatchNormalization into Conv")
    simp_p.add_argument("-o", "--output", type=str, default="simplified.onnx", help="Output path")

    # report (Feature 5)
    report_p = sub.add_parser("report", help="Generate a standalone HTML report")
    report_p.add_argument("model", type=str, help="Path to .onnx file")
    report_p.add_argument("-o", "--output", type=str, default="model_report.html", help="Output HTML path")
    report_p.add_argument("--title", type=str, default="ONNX Model Report", help="Report title")

    # rename (Feature 6)
    rename_p = sub.add_parser("rename", help="Bulk rename tensors in a model")
    rename_p.add_argument("model", type=str, help="Path to .onnx file")
    rename_p.add_argument("--map", nargs="+", required=True,
                          help="Rename mappings: old:new old:new ...")
    rename_p.add_argument("-o", "--output", type=str, default="renamed.onnx", help="Output path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # diff takes two models
    if args.command == "diff":
        try:
            model_a = load_model(args.model_a)
            model_b = load_model(args.model_b)
        except FileNotFoundError as e:
            print(f" {FAIL} {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f" {FAIL} Failed to load model: {e}", file=sys.stderr)
            sys.exit(1)

        result = diff(model_a, model_b)
        print(format_diff(result))
        return

    # rename
    if args.command == "rename":
        try:
            model = load_model(args.model)
        except Exception as e:
            print(f" {FAIL} {e}", file=sys.stderr)
            sys.exit(1)

        rename_map = {}
        for mapping in args.map:
            if ":" not in mapping:
                print(f" {FAIL} Invalid mapping: {mapping} (expected old:new)", file=sys.stderr)
                sys.exit(1)
            old, new = mapping.split(":", 1)
            rename_map[old] = new

        result = rename_tensors(model, rename_map)
        path = export(result, args.output)
        print(f"  Renamed {len(rename_map)} tensors.")
        print(f"  {OK} {path}")
        return

    # Load model for remaining commands
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f" {FAIL} {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f" {FAIL} Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "info":
        if args.json:
            print(to_json(model))
        elif args.shapes:
            flops_data = estimate_flops(model)
            print(inspect(model, detailed=True))
            print("-" * 56)
            print("  Computational Estimate")
            print("-" * 56)
            print(f"  Total FLOPs:  {format_flops(flops_data['total_flops'])}")
            print(f"  Total MACs:   {format_flops(flops_data['total_macs'])}")
            print(f"  Parameters:   {format_params(flops_data['total_params'])} ({flops_data['params_size_mb']:.1f} MB)")
            if flops_data['unknown_shapes']:
                print(f"  {WARN} {flops_data['unknown_shapes']} nodes have unknown shapes")
        else:
            print(inspect(model, detailed=True))

    elif args.command == "graph":
        try:
            print(ascii_graph(model))
        except Exception as e:
            print(f" {FAIL} Graph rendering failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "stats":
        try:
            print(op_stats(model))
        except Exception as e:
            print(f" {FAIL} Stats failed: {e}", file=sys.stderr)
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
            print(f"  {OK} {path}")
        except Exception as e:
            print(f" {FAIL} Prune failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "strip":
        try:
            result = strip_initializers(model)
            result = optimize(result, level=args.optimize)
            path = export(result, args.output)
            print("  Stripped initializers, removed Identity nodes.")
            print(f"  {OK} {path}")
        except Exception as e:
            print(f" {FAIL} Strip failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        issues = validate_model(model)
        if issues:
            print(f" {FAIL} Validation issues found:")
            for issue in issues:
                print(f"    - {issue}")
            sys.exit(1)
        else:
            print(f" {OK} Model is valid!")

    elif args.command == "json":
        print(to_json(model))

    elif args.command == "flops":
        flops_data = estimate_flops(model)
        print("=" * 56)
        print("  FLOPs & Parameter Estimation")
        print("=" * 56)
        print(f"  Total FLOPs:            {format_flops(flops_data['total_flops'])}")
        print(f"  Total MACs:             {format_flops(flops_data['total_macs'])}")
        print(f"  Total Parameters:       {format_params(flops_data['total_params'])}")
        print(f"  Parameter Memory:       {flops_data['params_size_mb']:.1f} MB (float32)")
        print("")
        if flops_data['flops_by_op']:
            print("  Per-Operator FLOPs:")
            max_flops = max(flops_data['flops_by_op'].values()) if flops_data['flops_by_op'] else 1
            for op, count in sorted(flops_data['flops_by_op'].items(), key=lambda x: -x[1]):
                bar = "#" * min(int(count / max_flops * 30) if max_flops else 0, 30)
                print(f"    {op:<20} {format_flops(count):<12} {bar}")
        if flops_data['unknown_shapes']:
            print(f"\n  {WARN} {flops_data['unknown_shapes']} node(s) have unknown shapes - results may be incomplete")

    elif args.command == "extract":
        try:
            result = extract_subgraph(model, args.from_tensors, args.to_tensors)
            path = export(result, args.output)
            from_tensors = ", ".join(args.from_tensors)
            to_tensors = ", ".join(args.to_tensors)
            print(f"  Extracted subgraph: {from_tensors} -> {to_tensors}")
            print(f"  Nodes: {len(result.graph.node)}")
            print(f"  {OK} {path}")
        except Exception as e:
            print(f" {FAIL} Extraction failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "simplify":
        try:
            result = simplify_model(
                model,
                fold_constants=not args.no_fold,
                fuse_bn=args.fuse_bn,
            )
            path = export(result, args.output)
            before = len(model.graph.node)
            after = len(result.graph.node)
            print(f"  Simplified: {before} -> {after} nodes")
            print(f"  {OK} {path}")
        except Exception as e:
            print(f" {FAIL} Simplify failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "report":
        try:
            html = generate_html_report(model, title=args.title)
            output_path = Path(args.output)
            output_path.write_text(html, encoding="utf-8")
            size = output_path.stat().st_size
            print(f"  {OK} HTML report saved to {output_path} ({size:,} bytes)")
        except Exception as e:
            print(f" {FAIL} Report generation failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

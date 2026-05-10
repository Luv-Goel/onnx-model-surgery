"""Interactive HTML diff report for comparing two ONNX models.

Produces a standalone HTML file (no external dependencies) that visualises:

- Summary cards showing node/input/output/parameter deltas
- Added / removed / changed node tables with colour-coded rows
- Operator distribution bar chart (CSS-based, no JS needed)
- Input / output shape diffs

Usage::

    from onnx_surgery.tools.diff_report import generate_diff_html

    model_a = load_model("v1.onnx")
    model_b = load_model("v2.onnx")
    html = generate_diff_html(model_a, model_b, title="My Model v1 vs v2")
    Path("diff.html").write_text(html, encoding="utf-8")
"""

from __future__ import annotations

from onnx import ModelProto
from .diff import diff


def generate_diff_html(
    model_a: ModelProto,
    model_b: ModelProto,
    title: str = "ONNX Model Diff Report",
) -> str:
    """Generate a standalone HTML diff report comparing two ONNX models.

    Args:
        model_a: First (original / baseline) model.
        model_b: Second (modified / new) model.
        title: Page title.

    Returns:
        Complete self-contained HTML string.
    """
    d = diff(model_a, model_b)
    s = d["summary_diff"]

    def delta_class(val):
        return "delta-pos" if val > 0 else ("delta-neg" if val < 0 else "delta-zero")

    # Summary card items
    def _summary_row(label: str, a: int, b: int) -> str:
        delta = b - a
        cls = delta_class(delta)
        sign = "+" if delta > 0 else ""
        return (
            f"<tr>"
            f"  <td>{label}</td>"
            f'  <td class="num">{a}</td>'
            f'  <td class="num">{b}</td>'
            f'  <td class="num {cls}">{sign}{delta}</td>'
            f"</tr>"
        )

    # --- Added nodes ---
    added_rows = ""
    for n in d["nodes_added"]:
        added_rows += (
            f'<tr class="row-added">'
            f"  <td>{n['name']}</td>"
            f'  <td><span class="tag">{n["op_type"]}</span></td>'
            f'  <td class="trunc">{", ".join(n["inputs"][:3])}</td>'
            f'  <td class="trunc">{", ".join(n["outputs"][:2])}</td>'
            f"</tr>"
        )

    # --- Removed nodes ---
    removed_rows = ""
    for n in d["nodes_removed"]:
        removed_rows += (
            f'<tr class="row-removed">'
            f"  <td>{n['name']}</td>"
            f'  <td><span class="tag">{n["op_type"]}</span></td>'
            f'  <td class="trunc">{", ".join(n["inputs"][:3])}</td>'
            f'  <td class="trunc">{", ".join(n["outputs"][:2])}</td>'
            f"</tr>"
        )

    # --- Changed nodes ---
    changed_rows = ""
    for n in d["nodes_changed"]:
        changes_str = "; ".join(
            f"{attr}: {old} → {new}" for attr, (old, new) in n["changes"].items()
        )
        changed_rows += (
            f'<tr class="row-changed">'
            f"  <td>{n['name']}</td>"
            f'  <td colspan="3" class="changes">{changes_str}</td>'
            f"</tr>"
        )

    # --- Op distribution diff ---
    op_rows = ""
    if d["op_dist_diff"]:
        max_count = max(max(ca, cb) for ca, cb in d["op_dist_diff"].values()) or 1
        for op, (ca, cb) in sorted(d["op_dist_diff"].items(), key=lambda x: -max(x[1])):
            delta = cb - ca
            cls = delta_class(delta)
            sign = "+" if delta > 0 else ""
            bar_a = int(ca / max_count * 100)
            bar_b = int(cb / max_count * 100)
            op_rows += (
                f"<tr>"
                f"  <td>{op}</td>"
                f"  <td class='num'>{ca}</td>"
                f"  <td class='num'>{cb}</td>"
                f"  <td class='num {cls}'>{sign}{delta}</td>"
                f"  <td>"
                f"    <div class='bar-wrap'>"
                f"      <div class='bar bar-a' style='width:{bar_a}%' title='Model A: {ca}'></div>"
                f"      <div class='bar bar-b' style='width:{bar_b}%' title='Model B: {cb}'></div>"
                f"    </div>"
                f"  </td>"
                f"</tr>"
            )

    # --- Input / Output changes ---
    io_rows = ""
    for name, (va, vb) in sorted(d["input_diff"].items()):
        io_rows += (
            f'<tr><td>{name}</td><td><span class="badge-in">input</span></td>'
            f'<td class="trunc">{va}</td><td class="trunc">{vb}</td></tr>'
        )
    for name, (va, vb) in sorted(d["output_diff"].items()):
        io_rows += (
            f'<tr><td>{name}</td><td><span class="badge-out">output</span></td>'
            f'<td class="trunc">{va}</td><td class="trunc">{vb}</td></tr>'
        )

    no_changes = not any(
        [
            d["nodes_added"],
            d["nodes_removed"],
            d["nodes_changed"],
            d["op_dist_diff"],
            d["input_diff"],
            d["output_diff"],
        ]
    )

    identical_block = (
        '<div class="identical-msg">✓ Models are structurally identical — no differences found.</div>'
        if no_changes
        else ""
    )

    has_ops = bool(d["op_dist_diff"])
    has_io = bool(d["input_diff"] or d["output_diff"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f111a; color: #e1e4eb; line-height: 1.5; padding: 20px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: .9rem; margin-bottom: 24px; }}
  h2 {{ font-size: 1.15rem; margin: 28px 0 12px; padding-bottom: 6px;
        border-bottom: 1px solid #2a2d3a; color: #b0b8d1; }}
  table {{ width: 100%; border-collapse: collapse; background: #161822; border-radius: 8px;
          overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  th {{ background: #1e2132; color: #8892b0; padding: 10px 14px; text-align: left;
        font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #1e2132; font-size: .85rem; }}
  .num {{ font-family: 'SF Mono', 'Fira Code', monospace; text-align: right; }}
  .trunc {{ max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 4px;
          font-size: .75rem; font-weight: 600; background: #2a3060; color: #7c8ce8; }}
  .badge-in {{ display: inline-block; padding: 1px 8px; border-radius: 4px;
               font-size: .7rem; background: #1a3a2a; color: #4caf78; }}
  .badge-out {{ display: inline-block; padding: 1px 8px; border-radius: 4px;
                font-size: .7rem; background: #3a1a1a; color: #e85555; }}

  /* Delta colours */
  .delta-pos {{ color: #4caf78; }}
  .delta-neg {{ color: #e85555; }}
  .delta-zero {{ color: #666; }}

  /* Row colours */
  .row-added {{ background: rgba(76, 175, 120, .08); }}
  .row-added:hover {{ background: rgba(76, 175, 120, .15); }}
  .row-removed {{ background: rgba(232, 85, 85, .08); }}
  .row-removed:hover {{ background: rgba(232, 85, 85, .15); }}
  .row-changed {{ background: rgba(255, 193, 7, .06); }}
  .row-changed:hover {{ background: rgba(255, 193, 7, .12); }}
  .changes {{ font-size: .8rem; color: #c8b878; }}

  /* Summary cards */
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                  gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #161822; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  .card .label {{ font-size: .7rem; text-transform: uppercase; color: #666f88; letter-spacing: .05em; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; }}
  .card .sub {{ font-size: .8rem; color: #888; }}

  /* Bar chart */
  .bar-wrap {{ display: flex; gap: 2px; height: 16px; border-radius: 3px; overflow: hidden; background: #1e2132; }}
  .bar {{ height: 100%; min-width: 2px; transition: width .3s; }}
  .bar-a {{ background: #4caf78; opacity: .6; }}
  .bar-b {{ background: #7c8ce8; opacity: .8; }}

  /* Status */
  .identical-msg {{ background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 8px;
                    padding: 20px; text-align: center; font-size: 1.1rem; color: #6aaa6a; }}
  .footer {{ margin-top: 30px; padding: 16px; text-align: center; color: #555; font-size: .75rem; }}
</style>
</head>
<body>
<div class="container">

<h1>{title}</h1>
<p class="subtitle">Generated by ONNX Model Surgery &mdash; Diff Report</p>

<div class="summary-grid">
  <div class="card">
    <div class="label">Nodes</div>
    <div class="value">{s["nodes"][1]}</div>
    <div class="sub">{s["nodes"][0]} → {s["nodes"][1]} ({s["nodes"][1] - s["nodes"][0]:+d})</div>
  </div>
  <div class="card">
    <div class="label">Inputs</div>
    <div class="value">{s["inputs"][1]}</div>
    <div class="sub">{s["inputs"][0]} → {s["inputs"][1]} ({s["inputs"][1] - s["inputs"][0]:+d})</div>
  </div>
  <div class="card">
    <div class="label">Outputs</div>
    <div class="value">{s["outputs"][1]}</div>
    <div class="sub">{s["outputs"][0]} → {s["outputs"][1]} ({s["outputs"][1] - s["outputs"][0]:+d})</div>
  </div>
  <div class="card">
    <div class="label">Parameters</div>
    <div class="value">{s["parameters"][1]}</div>
    <div class="sub">{s["parameters"][0]} → {s["parameters"][1]} ({s["parameters"][1] - s["parameters"][0]:+d})</div>
  </div>
</div>

{identical_block}

<!-- Added / Removed / Changed -->
{"<h2>➕ Added Nodes (" + str(len(d["nodes_added"])) + ")</h2>" if d["nodes_added"] else ""}
{"<table><tr><th>Name</th><th>Op Type</th><th>Inputs</th><th>Outputs</th></tr>" + added_rows + "</table>" if d["nodes_added"] else ""}

{"<h2>➖ Removed Nodes (" + str(len(d["nodes_removed"])) + ")</h2>" if d["nodes_removed"] else ""}
{"<table><tr><th>Name</th><th>Op Type</th><th>Inputs</th><th>Outputs</th></tr>" + removed_rows + "</table>" if d["nodes_removed"] else ""}

{"<h2>✏️ Changed Nodes (" + str(len(d["nodes_changed"])) + ")</h2>" if d["nodes_changed"] else ""}
{"<table><tr><th>Name</th><th colspan='3'>Changes</th></tr>" + changed_rows + "</table>" if d["nodes_changed"] else ""}

<!-- Op distribution -->
{"<h2>📊 Operator Distribution Changes</h2>" if has_ops else ""}
{"<table><tr><th>Op Type</th><th>Model A</th><th>Model B</th><th>Δ</th><th>Distribution</th></tr>" + op_rows + "</table>" if has_ops else ""}

<!-- Input / Output diffs -->
{"<h2>🔌 Input / Output Changes</h2>" if has_io else ""}
{"<table><tr><th>Name</th><th>Direction</th><th>Before</th><th>After</th></tr>" + io_rows + "</table>" if has_io else ""}

<div class="footer">
  Generated by <strong>ONNX Model Surgery</strong> &mdash; {title}
</div>

</div>
</body>
</html>"""


__all__ = ["generate_diff_html"]

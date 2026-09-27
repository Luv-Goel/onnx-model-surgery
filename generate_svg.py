from rich.console import Console

console = Console(record=True, width=80)
console.print("[bold green]$[/bold green] oms flops model.onnx")
console.print("========================================================")
console.print("  FLOPs & Parameter Estimation")
console.print("========================================================")
console.print("  Total FLOPs:            8.20 GFLOPs")
console.print("  Total MACs:             4.10 GMACs")
console.print("  Total Parameters:       25.56 M")
console.print("  Parameter Memory:       97.5 MB (float32)")
console.print()
console.print("  Per-Operator FLOPs:")
console.print("    Conv                 8.00 GFLOPs   ##############################")
console.print("    MatMul               0.20 GFLOPs   #")
console.print("    BatchNormalization   0.00 GFLOPs   ")

console.save_svg("assets/demo.svg", title="ONNX Model Surgery CLI")

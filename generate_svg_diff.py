from rich.console import Console

console = Console(record=True, width=80)
console.print("[bold green]$[/bold green] oms diff model_v1.onnx model_v2.onnx")
console.print("[bold cyan]Model Diff Summary[/bold cyan]")
console.print("========================================================")
console.print("  [bold red]- Removed nodes:[/bold red] 12 (Dropout, Identity)")
console.print("  [bold green]+ Added nodes:[/bold green]   2 (Conv, Relu)")
console.print("  [bold yellow]* Changed nodes:[/bold yellow] 3 (BatchNormalization)")
console.print("========================================================")
console.print("  Total parameters changed: -120 KB")

console.save_svg("assets/diff_demo.svg", title="ONNX Model Surgery - Diff")

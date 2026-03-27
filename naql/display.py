"""Rich terminal display for naql."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from naql.inspector import ModelInfo

console = Console()

BRANDING = "[bold magenta]naql[/bold magenta] [dim]- Arabic Model Format Converter[/dim]"

# ── Supported formats ─────────────────────────────────────────────

FORMATS = {
    "hf": "Hugging Face (PyTorch / Safetensors)",
    "gguf": "GGUF (llama.cpp / Ollama)",
    "mlx": "MLX (Apple Silicon)",
    "onnx": "ONNX Runtime",
    "safetensors": "Safetensors (raw weights)",
    "gptq": "GPTQ (GPU quantized, AutoGPTQ)",
    "awq": "AWQ (Activation-aware Weight Quantization)",
    "jang": "JANG (JAX Angular, experimental)",
}

# Conversion paths: (source, target) -> (tool, install_hint)
CONVERSION_PATHS: dict[tuple[str, str], tuple[str, str]] = {
    ("hf", "gguf"): ("llama.cpp convert", "brew install llama.cpp  OR  git clone https://github.com/ggerganov/llama.cpp"),
    ("hf", "mlx"): ("mlx-lm", "pip install mlx-lm"),
    ("hf", "onnx"): ("optimum", "pip install optimum[onnxruntime]"),
    ("hf", "safetensors"): ("transformers", "pip install transformers"),
    ("gguf", "hf"): ("llama.cpp convert-back", "brew install llama.cpp  OR  git clone https://github.com/ggerganov/llama.cpp"),
    ("mlx", "hf"): ("mlx-lm", "pip install mlx-lm"),
    ("mlx", "gguf"): ("mlx-lm -> hf -> llama.cpp", "pip install mlx-lm  AND  brew install llama.cpp"),
    ("safetensors", "hf"): ("transformers", "pip install transformers"),
    ("safetensors", "gguf"): ("llama.cpp convert", "brew install llama.cpp"),
    ("onnx", "hf"): ("optimum", "pip install optimum[onnxruntime]"),
    ("hf", "gptq"): ("auto-gptq", "pip install auto-gptq"),
    ("hf", "awq"): ("autoawq", "pip install autoawq"),
    ("gptq", "gguf"): ("llama.cpp convert", "git clone https://github.com/ggerganov/llama.cpp"),
    ("awq", "gguf"): ("llama.cpp convert", "git clone https://github.com/ggerganov/llama.cpp"),
}


# ── Model info display ────────────────────────────────────────────


def _format_size(size_mb: float) -> str:
    """Format a size in MB to a human-readable string."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.1f} MB"


def _format_params(num: int | None) -> str:
    """Format parameter count to a human-readable string."""
    if num is None:
        return "unknown"
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.0f}M"
    if num >= 1_000:
        return f"{num / 1_000:.0f}K"
    return str(num)


def display_info(info: ModelInfo) -> None:
    """Show model inspection results as a Rich table."""
    console.print()
    console.print(BRANDING)
    console.print()

    table = Table(
        show_header=False,
        border_style="dim",
        padding=(0, 1),
        min_width=50,
    )
    table.add_column("Field", style="bold cyan", min_width=18)
    table.add_column("Value", style="white")

    fields = [
        ("Path", info.path),
        ("Format", info.format),
        ("Size", _format_size(info.size_mb)),
        ("Parameters", _format_params(info.num_parameters)),
        ("Architecture", info.architecture or "unknown"),
        ("Vocab Size", str(info.vocab_size) if info.vocab_size else "unknown"),
        ("Quantization", info.quantization or "none"),
        ("Context Length", str(info.context_length) if info.context_length else "unknown"),
        ("Tokenizer", info.tokenizer_type or ("present" if info.has_tokenizer else "none")),
    ]

    for field, value in fields:
        if value and value not in ("unknown", "none"):
            table.add_row(field, str(value))
        else:
            table.add_row(field, f"[dim]{value}[/dim]")

    console.print(table)
    console.print()


# ── Arabic tokenizer check ────────────────────────────────────────


def display_arabic_check(result: dict) -> None:
    """Show Arabic tokenizer check results."""
    console.print()
    console.print(BRANDING)
    console.print()

    has_arabic = result.get("has_arabic", False)
    count = result.get("arabic_token_count", 0)
    total = result.get("total_vocab", 0)
    ratio = result.get("arabic_ratio", 0.0)
    tokenizer_type = result.get("tokenizer_type", "unknown")
    bigram_count = result.get("arabic_bigram_count", 0)
    common_words = result.get("common_words_list", [])
    efficiency = result.get("arabic_efficiency_score", 0)

    if has_arabic:
        console.print("  [bold green]Arabic support detected[/bold green]")
    else:
        console.print("  [bold red]No Arabic support detected[/bold red]")

    console.print()

    table = Table(
        show_header=False,
        border_style="dim",
        padding=(0, 1),
        min_width=40,
    )
    table.add_column("Metric", style="bold cyan", min_width=18)
    table.add_column("Value", style="white")

    table.add_row("Tokenizer Type", tokenizer_type)
    table.add_row("Total Vocab", str(total))
    table.add_row("Arabic Tokens", str(count))
    table.add_row("Arabic Ratio", f"{ratio:.1%}")
    table.add_row("Arabic Bigrams", str(bigram_count))

    # Efficiency score with color coding.
    if efficiency >= 70:
        eff_str = f"[bold green]{efficiency}/100[/bold green]"
    elif efficiency >= 40:
        eff_str = f"[bold yellow]{efficiency}/100[/bold yellow]"
    else:
        eff_str = f"[bold red]{efficiency}/100[/bold red]"
    table.add_row("Efficiency Score", eff_str)

    # Common Arabic words found.
    if common_words:
        words_str = ", ".join(common_words)
        table.add_row("Common Words", f"[dim]{words_str}[/dim]")
    else:
        table.add_row("Common Words", "[dim]none found[/dim]")

    table.add_row(
        "Verdict",
        "[green]Good[/green]" if has_arabic else "[red]No Arabic support[/red]",
    )

    console.print(table)
    console.print()


# ── Conversion plan ───────────────────────────────────────────────


def display_plan(plan: dict) -> None:
    """Show conversion plan: source -> target, command, tool, install hint."""
    console.print()
    console.print(BRANDING)
    console.print()

    source = plan.get("source_format", "?")
    target = plan.get("target_format", "?")

    console.print(f"  [bold]{source}[/bold] [dim]->[/dim] [bold]{target}[/bold]")
    console.print()

    table = Table(
        show_header=False,
        border_style="dim",
        padding=(0, 1),
        min_width=50,
    )
    table.add_column("Field", style="bold cyan", min_width=18)
    table.add_column("Value", style="white")

    table.add_row("Tool", plan.get("tool", "unknown"))
    table.add_row("Command", f"[bold white]{plan.get('command', 'N/A')}[/bold white]")

    if plan.get("install_hint"):
        table.add_row("Install", f"[dim]{plan['install_hint']}[/dim]")

    console.print(table)
    console.print()


# ── Supported conversions matrix ──────────────────────────────────


def display_conversions() -> None:
    """Show all supported conversion paths as a matrix table."""
    console.print()
    console.print(BRANDING)
    console.print()
    console.print("[bold]Supported conversion paths:[/bold]")
    console.print()

    format_keys = sorted(FORMATS.keys())

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )

    table.add_column("From \\ To", style="bold white", min_width=8)
    for fmt in format_keys:
        table.add_column(fmt.upper(), justify="center", min_width=6)

    for src in format_keys:
        row: list[str] = []
        for tgt in format_keys:
            if src == tgt:
                row.append("[dim]-[/dim]")
            elif (src, tgt) in CONVERSION_PATHS:
                row.append("[green]yes[/green]")
            else:
                row.append("[dim red]no[/dim red]")
        table.add_row(src.upper(), *row)

    console.print(table)
    console.print()

    # Legend
    console.print("[bold]Format descriptions:[/bold]")
    console.print()
    for key, desc in sorted(FORMATS.items()):
        console.print(f"  [bold cyan]{key:14s}[/bold cyan] {desc}")
    console.print()


# ── Validation results ────────────────────────────────────────────


def display_validate(issues: list[dict]) -> None:
    """Show validation results after conversion."""
    console.print()
    console.print(BRANDING)
    console.print()

    if not issues:
        console.print("  [bold green]Validation passed[/bold green] - no issues found.")
        console.print()
        return

    console.print(f"  [bold red]Found {len(issues)} issue(s)[/bold red]")
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Severity", style="bold", min_width=8)
    table.add_column("Check", style="white")
    table.add_column("Details", style="dim white", max_width=60)

    for issue in issues:
        severity = issue.get("severity", "warning")
        color = "red" if severity == "error" else "yellow"
        icon = "ERR" if severity == "error" else "WARN"
        table.add_row(
            f"[{color}]{icon}[/{color}]",
            issue.get("check", "unknown"),
            issue.get("details", ""),
        )

    console.print(table)
    console.print()


# ── Diff display ──────────────────────────────────────────────────


def display_diff(
    diffs: list[dict],
    source_info: ModelInfo,
    target_info: ModelInfo,
) -> None:
    """Show side-by-side model comparison as a Rich table."""
    console.print()
    console.print(BRANDING)
    console.print()

    console.print(f"  [bold cyan]Source:[/bold cyan] {source_info.path}  [dim]({source_info.format})[/dim]")
    console.print(f"  [bold cyan]Target:[/bold cyan] {target_info.path}  [dim]({target_info.format})[/dim]")
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
        min_width=60,
    )
    table.add_column("Field", style="bold white", min_width=16)
    table.add_column("Source", style="white", min_width=14)
    table.add_column("Target", style="white", min_width=14)
    table.add_column("Status", justify="center", min_width=8)

    for diff in diffs:
        status = diff.get("status", "mismatch")
        if status == "match":
            icon = "[green]OK[/green]"
        elif status == "warning":
            icon = "[yellow]WARN[/yellow]"
        else:
            icon = "[red]DIFF[/red]"

        src_val = diff.get("source", "")
        tgt_val = diff.get("target", "")

        # Color the values based on status.
        if status == "match":
            src_style = f"[green]{src_val}[/green]"
            tgt_style = f"[green]{tgt_val}[/green]"
        elif status == "warning":
            src_style = f"[yellow]{src_val}[/yellow]"
            tgt_style = f"[yellow]{tgt_val}[/yellow]"
        else:
            src_style = f"[red]{src_val}[/red]"
            tgt_style = f"[red]{tgt_val}[/red]"

        table.add_row(diff.get("field", ""), src_style, tgt_style, icon)

    console.print(table)

    # Summary line.
    matches = sum(1 for d in diffs if d.get("status") == "match")
    total = len(diffs)
    if matches == total:
        console.print(f"\n  [bold green]All {total} fields match.[/bold green]")
    else:
        mismatches = sum(1 for d in diffs if d.get("status") == "mismatch")
        warnings = sum(1 for d in diffs if d.get("status") == "warning")
        parts = []
        if matches:
            parts.append(f"[green]{matches} match[/green]")
        if warnings:
            parts.append(f"[yellow]{warnings} warning(s)[/yellow]")
        if mismatches:
            parts.append(f"[red]{mismatches} mismatch(es)[/red]")
        console.print(f"\n  {', '.join(parts)}")

    console.print()


# ── JSON output ───────────────────────────────────────────────────


def display_json(data: dict | list) -> None:
    """Print data as formatted JSON."""
    console.print(json.dumps(data, ensure_ascii=False, indent=2))


# ── Explain ───────────────────────────────────────────────────────


def display_explain() -> None:
    """Explain what naql does, the formats it supports, and why Arabic tokenizer preservation matters."""
    console.print()
    console.print(BRANDING)
    console.print()

    console.print("[bold]What is naql?[/bold]")
    console.print()
    console.print(
        "  naql (Arabic for 'transfer') helps you convert AI models between\n"
        "  formats. It inspects models, checks Arabic language support, and\n"
        "  generates the correct conversion commands.\n"
    )

    console.print("[bold]Supported formats:[/bold]")
    console.print()
    for key, desc in sorted(FORMATS.items()):
        console.print(f"  [bold cyan]{key:14s}[/bold cyan] {desc}")
    console.print()

    console.print("[bold]Why Arabic tokenizer preservation matters:[/bold]")
    console.print()
    console.print(
        "  When converting models between formats, the tokenizer must be\n"
        "  preserved exactly. Arabic text uses complex shaping, right-to-left\n"
        "  layout, and context-dependent letter forms. A broken tokenizer\n"
        "  will produce garbled output or silently degrade Arabic performance.\n"
    )
    console.print(
        "  naql checks for Arabic token coverage before and after conversion\n"
        "  to ensure nothing is lost in translation.\n"
    )

    console.print("[bold]Workflow:[/bold]")
    console.print()
    console.print("  1. [bold cyan]naql inspect model/[/bold cyan]          See what you have")
    console.print("  2. [bold cyan]naql arabic model/[/bold cyan]           Check Arabic support")
    console.print("  3. [bold cyan]naql convert model/ --to gguf[/bold cyan] Get the conversion command")
    console.print("  4. [bold cyan]naql validate src/ out/[/bold cyan]      Verify the conversion")
    console.print("  5. [bold cyan]naql diff model1/ model2/[/bold cyan]    Compare two models")
    console.print()

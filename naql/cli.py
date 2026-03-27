"""CLI entry point for naql."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from naql.display import (
    console,
    display_arabic_check,
    display_conversions,
    display_diff,
    display_explain,
    display_info,
    display_json,
    display_plan,
    display_validate,
)


# ── Subcommand handlers ──────────────────────────────────────────


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Run the inspect subcommand."""
    from naql.inspector import inspect_model

    info = inspect_model(args.path)

    if args.json:
        display_json(info.to_dict())
    else:
        display_info(info)

    return 0


def _cmd_arabic(args: argparse.Namespace) -> int:
    """Run the arabic subcommand - check Arabic tokenizer support."""
    from naql.inspector import check_arabic_tokenizer

    result = check_arabic_tokenizer(args.path)

    if args.json:
        display_json(result)
    else:
        display_arabic_check(result)

    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Run the convert subcommand - generate conversion command."""
    from naql.inspector import detect_format
    from naql.display import CONVERSION_PATHS

    source_format = detect_format(args.source)

    # Normalise HuggingFace variants to "hf".
    fmt = source_format if source_format != "huggingface" else "hf"
    target = args.to

    key = (fmt, target)
    if key not in CONVERSION_PATHS:
        console.print(
            f"\n[bold red]Error:[/bold red] no known conversion path "
            f"from [bold]{fmt}[/bold] to [bold]{target}[/bold].\n"
            f"Run [bold cyan]naql formats[/bold cyan] to see supported paths.\n"
        )
        return 1

    tool, install_hint = CONVERSION_PATHS[key]

    # Build conversion command.
    command = _build_command(
        source=args.source,
        source_format=fmt,
        target_format=target,
        output=args.output,
        quant=args.quant,
    )

    plan = {
        "source_format": fmt,
        "target_format": target,
        "tool": tool,
        "command": command,
        "install_hint": install_hint,
    }

    display_plan(plan)

    if args.run:
        return _run_command(command)

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run the validate subcommand - validate conversion results."""
    from naql.inspector import inspect_model, check_arabic_tokenizer

    issues: list[dict] = []

    # Check source exists and is inspectable.
    source_info = inspect_model(args.source)
    target_info = inspect_model(args.target)

    if target_info.format == "unknown":
        issues.append({
            "severity": "error",
            "check": "format_detection",
            "details": f"Could not detect format of target: {args.target}",
        })

    # Compare vocab sizes.
    if source_info.vocab_size and target_info.vocab_size:
        if source_info.vocab_size != target_info.vocab_size:
            issues.append({
                "severity": "warning",
                "check": "vocab_size",
                "details": (
                    f"Vocab size mismatch: source={source_info.vocab_size}, "
                    f"target={target_info.vocab_size}"
                ),
            })

    # Compare architecture.
    if source_info.architecture and target_info.architecture:
        if source_info.architecture != target_info.architecture:
            issues.append({
                "severity": "warning",
                "check": "architecture",
                "details": (
                    f"Architecture mismatch: source={source_info.architecture}, "
                    f"target={target_info.architecture}"
                ),
            })

    # Check Arabic tokenizer preservation.
    source_arabic = check_arabic_tokenizer(args.source)
    target_arabic = check_arabic_tokenizer(args.target)

    if source_arabic.get("has_arabic") and not target_arabic.get("has_arabic"):
        issues.append({
            "severity": "error",
            "check": "arabic_tokenizer",
            "details": "Source has Arabic tokens but target does not - tokenizer may be lost!",
        })
    elif source_arabic.get("arabic_token_count", 0) > 0 and target_arabic.get("arabic_token_count", 0) > 0:
        src_count = source_arabic["arabic_token_count"]
        tgt_count = target_arabic["arabic_token_count"]
        if abs(src_count - tgt_count) > 10:
            issues.append({
                "severity": "warning",
                "check": "arabic_token_count",
                "details": (
                    f"Arabic token count changed: source={src_count}, "
                    f"target={tgt_count}"
                ),
            })

    if args.json:
        display_json(issues)
    else:
        display_validate(issues)

    return 1 if issues else 0


def _cmd_formats(_args: argparse.Namespace) -> int:
    """Run the formats subcommand - show supported conversion paths."""
    display_conversions()
    return 0


def _cmd_explain(_args: argparse.Namespace) -> int:
    """Run the explain subcommand - explain the tool."""
    display_explain()
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Run the diff subcommand - compare two models side by side."""
    from naql.inspector import inspect_model, check_arabic_tokenizer
    from naql.converter import diff_models

    source_info = inspect_model(args.source)
    target_info = inspect_model(args.target)

    source_arabic = check_arabic_tokenizer(args.source)
    target_arabic = check_arabic_tokenizer(args.target)

    diffs = diff_models(source_info, target_info, source_arabic, target_arabic)

    if args.json:
        display_json(diffs)
    else:
        display_diff(diffs, source_info, target_info)

    return 0


# ── Command building ─────────────────────────────────────────────


def _build_command(
    source: str,
    source_format: str,
    target_format: str,
    output: str | None,
    quant: str | None,
) -> str:
    """Build the shell command string for a conversion."""
    output = output or f"{source}-{target_format}"

    if target_format == "gguf":
        cmd = f"python -m llama_cpp.convert {source} --outfile {output}.gguf"
        if quant:
            cmd += f" --outtype {quant}"
        return cmd

    if target_format == "mlx":
        cmd = f"python -m mlx_lm.convert --hf-path {source} --mlx-path {output}"
        if quant:
            cmd += f" -q --q-bits {quant.replace('bit', '')}"
        return cmd

    if target_format == "onnx":
        cmd = f"optimum-cli export onnx --model {source} {output}"
        if quant:
            cmd += f" --quantize {quant}"
        return cmd

    if target_format == "safetensors":
        cmd = (
            f"python -c \"from transformers import AutoModel; "
            f"m = AutoModel.from_pretrained('{source}'); "
            f"m.save_pretrained('{output}')\""
        )
        return cmd

    if target_format == "hf":
        if source_format == "gguf":
            return f"python -m llama_cpp.convert_back {source} --outdir {output}"
        if source_format == "mlx":
            return f"python -m mlx_lm.convert --mlx-path {source} --hf-path {output}"
        return f"# Manual conversion from {source_format} to hf may be needed"

    return f"# No automatic command available for {source_format} -> {target_format}"


def _run_command(command: str) -> int:
    """Execute a shell command and return the exit code."""
    import subprocess

    console.print(f"\n[bold]Running:[/bold] {command}\n")

    try:
        result = subprocess.run(command, shell=True, check=False)
        if result.returncode == 0:
            console.print("\n[bold green]Conversion completed successfully.[/bold green]\n")
        else:
            console.print(f"\n[bold red]Command exited with code {result.returncode}.[/bold red]\n")
        return result.returncode
    except FileNotFoundError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
        return 1


# ── Argument parser construction ─────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="naql",
        description="Arabic Model Format Converter - inspect, convert, and validate AI models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── inspect ──
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a model file or directory"
    )
    inspect_parser.add_argument(
        "path",
        help="Path to model file or directory",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )

    # ── arabic ──
    arabic_parser = subparsers.add_parser(
        "arabic", help="Check Arabic tokenizer support"
    )
    arabic_parser.add_argument(
        "path",
        help="Path to model file or directory",
    )
    arabic_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )

    # ── convert ──
    convert_parser = subparsers.add_parser(
        "convert", help="Generate a conversion command"
    )
    convert_parser.add_argument(
        "source",
        help="Path to source model",
    )
    convert_parser.add_argument(
        "--to",
        required=True,
        choices=["gguf", "mlx", "onnx", "safetensors", "hf", "gptq", "awq"],
        help="Target format",
    )
    convert_parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: auto-generated)",
    )
    convert_parser.add_argument(
        "--quant",
        default=None,
        help="Quantization type (e.g., q4_k_m, 4bit, 8bit)",
    )
    convert_parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Actually execute the conversion command (requires tools installed)",
    )

    # ── validate ──
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a conversion was successful"
    )
    validate_parser.add_argument(
        "source",
        help="Path to original model",
    )
    validate_parser.add_argument(
        "target",
        help="Path to converted model",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )

    # ── diff ──
    diff_parser = subparsers.add_parser(
        "diff", help="Compare two models side by side"
    )
    diff_parser.add_argument(
        "source",
        help="Path to first model (source)",
    )
    diff_parser.add_argument(
        "target",
        help="Path to second model (target)",
    )
    diff_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )

    # ── formats ──
    subparsers.add_parser(
        "formats", help="Show all supported conversion paths"
    )

    # ── explain ──
    subparsers.add_parser(
        "explain", help="Explain what naql does and why"
    )

    return parser


def _get_version() -> str:
    """Return the package version string."""
    from naql import __version__

    return __version__


# ── Entry point ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> NoReturn:
    """Main entry point for the naql CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to inspect when no subcommand is given.
    if args.command is None:
        remaining = argv or sys.argv[1:]
        if remaining:
            args = parser.parse_args(["inspect", *remaining])
        else:
            parser.print_help()
            sys.exit(0)

    dispatch = {
        "inspect": _cmd_inspect,
        "arabic": _cmd_arabic,
        "convert": _cmd_convert,
        "validate": _cmd_validate,
        "diff": _cmd_diff,
        "formats": _cmd_formats,
        "explain": _cmd_explain,
    }

    try:
        handler = dispatch[args.command]
        code = handler(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        code = 130
    except BrokenPipeError:
        # Silently handle piping to head/less.
        code = 0

    sys.exit(code)

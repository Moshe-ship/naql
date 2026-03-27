"""Model format conversion helpers and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .inspector import ModelInfo, check_arabic_tokenizer, detect_format, inspect_model


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionPlan:
    """A planned conversion with the command to execute."""

    source_format: str
    target_format: str
    source_path: str
    command: str
    tool: str
    install_hint: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversion command registry
# ---------------------------------------------------------------------------

# Maps (source_format, target_format) to a dict describing the conversion.
# Command templates use {src}, {dst}, and {quant} as placeholders.
CONVERSION_COMMANDS: dict[tuple[str, str], dict] = {
    ("gguf", "mlx"): {
        "command": "mlx_lm.convert --hf-path {src} --mlx-path {dst} -q",
        "tool": "mlx-lm",
        "install": "pip install mlx-lm",
        "notes": [
            "GGUF -> MLX typically requires an intermediate HuggingFace step.",
            "Consider converting GGUF -> HF first, then HF -> MLX.",
        ],
    },
    ("hf", "gguf"): {
        "command": (
            "python llama.cpp/convert_hf_to_gguf.py {src}"
            " --outfile {dst} --outtype {quant}"
        ),
        "tool": "llama.cpp",
        "install": "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make",
        "notes": [
            "Common quantization types: q4_k_m, q5_k_m, q8_0, f16.",
            "The convert script requires Python dependencies: pip install -r llama.cpp/requirements.txt",
        ],
    },
    ("hf", "mlx"): {
        "command": "mlx_lm.convert --hf-path {src} --mlx-path {dst}",
        "tool": "mlx-lm",
        "install": "pip install mlx-lm",
        "notes": [
            "Add -q to quantize during conversion (default 4-bit).",
            "Add --q-bits 8 for 8-bit quantization.",
        ],
    },
    ("hf", "onnx"): {
        "command": "optimum-cli export onnx --model {src} {dst}",
        "tool": "optimum",
        "install": "pip install optimum[exporters] onnxruntime",
        "notes": [
            "ONNX export may not support all model architectures.",
            "Check optimum docs for model compatibility.",
        ],
    },
    ("mlx", "hf"): {
        "command": "mlx_lm.convert --mlx-path {src} --hf-path {dst} --de-quantize",
        "tool": "mlx-lm",
        "install": "pip install mlx-lm",
        "notes": [
            "De-quantization restores float16 weights — file size will increase.",
            "The resulting HF model can then be converted to other formats.",
        ],
    },
    ("gguf", "hf"): {
        "command": "# No direct one-step conversion available",
        "tool": "llama.cpp",
        "install": "git clone https://github.com/ggerganov/llama.cpp",
        "notes": [
            "GGUF -> HF requires dequantization via llama.cpp.",
            "Use: python llama.cpp/convert_hf_to_gguf.py --reverse {src} --outdir {dst}",
            "This is lossy if the GGUF was quantized from a higher precision.",
        ],
    },
    ("huggingface", "gguf"): {
        "command": (
            "python llama.cpp/convert_hf_to_gguf.py {src}"
            " --outfile {dst} --outtype {quant}"
        ),
        "tool": "llama.cpp",
        "install": "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make",
        "notes": [
            "Common quantization types: q4_k_m, q5_k_m, q8_0, f16.",
            "The convert script requires Python dependencies: pip install -r llama.cpp/requirements.txt",
        ],
    },
    ("huggingface", "mlx"): {
        "command": "mlx_lm.convert --hf-path {src} --mlx-path {dst}",
        "tool": "mlx-lm",
        "install": "pip install mlx-lm",
        "notes": [
            "Add -q to quantize during conversion (default 4-bit).",
            "Add --q-bits 8 for 8-bit quantization.",
        ],
    },
    ("huggingface", "onnx"): {
        "command": "optimum-cli export onnx --model {src} {dst}",
        "tool": "optimum",
        "install": "pip install optimum[exporters] onnxruntime",
        "notes": [
            "ONNX export may not support all model architectures.",
            "Check optimum docs for model compatibility.",
        ],
    },
    # GPTQ quantization paths.
    ("hf", "gptq"): {
        "command": (
            "python -m auto_gptq.quantize {src}"
            " --output_dir {dst} --bits 4 --group_size 128"
        ),
        "tool": "auto-gptq",
        "install": "pip install auto-gptq",
        "notes": [
            "GPTQ requires calibration data for quantization.",
            "Add --dataset c4 or provide a custom calibration dataset.",
            "Common configs: 4-bit group_size 128 (balanced), 4-bit group_size 64 (more accurate).",
            "Requires a CUDA GPU for quantization (inference can run on CPU).",
        ],
    },
    ("huggingface", "gptq"): {
        "command": (
            "python -m auto_gptq.quantize {src}"
            " --output_dir {dst} --bits 4 --group_size 128"
        ),
        "tool": "auto-gptq",
        "install": "pip install auto-gptq",
        "notes": [
            "GPTQ requires calibration data for quantization.",
            "Add --dataset c4 or provide a custom calibration dataset.",
            "Common configs: 4-bit group_size 128 (balanced), 4-bit group_size 64 (more accurate).",
            "Requires a CUDA GPU for quantization (inference can run on CPU).",
        ],
    },
    # AWQ quantization paths.
    ("hf", "awq"): {
        "command": (
            "python -m awq.entry --model_path {src}"
            " --w_bit 4 --q_group_size 128"
            " --dump_awq {dst}"
        ),
        "tool": "autoawq",
        "install": "pip install autoawq",
        "notes": [
            "AWQ (Activation-aware Weight Quantization) preserves salient weights.",
            "Typically produces better quality than naive round-to-nearest quantization.",
            "Requires a CUDA GPU for quantization.",
            "After quantization, convert to GGUF with: llama.cpp/convert_hf_to_gguf.py --awq",
        ],
    },
    ("huggingface", "awq"): {
        "command": (
            "python -m awq.entry --model_path {src}"
            " --w_bit 4 --q_group_size 128"
            " --dump_awq {dst}"
        ),
        "tool": "autoawq",
        "install": "pip install autoawq",
        "notes": [
            "AWQ (Activation-aware Weight Quantization) preserves salient weights.",
            "Typically produces better quality than naive round-to-nearest quantization.",
            "Requires a CUDA GPU for quantization.",
            "After quantization, convert to GGUF with: llama.cpp/convert_hf_to_gguf.py --awq",
        ],
    },
    # GPTQ/AWQ -> GGUF paths (common post-quantization step).
    ("gptq", "gguf"): {
        "command": (
            "python llama.cpp/convert_hf_to_gguf.py {src}"
            " --outfile {dst} --outtype {quant}"
        ),
        "tool": "llama.cpp",
        "install": "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make",
        "notes": [
            "GPTQ models can be converted to GGUF for use with llama.cpp/Ollama.",
            "The converter will dequantize GPTQ weights and re-quantize to GGUF format.",
        ],
    },
    ("awq", "gguf"): {
        "command": (
            "python llama.cpp/convert_hf_to_gguf.py {src}"
            " --outfile {dst} --outtype {quant}"
        ),
        "tool": "llama.cpp",
        "install": "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make",
        "notes": [
            "AWQ models can be converted to GGUF for use with llama.cpp/Ollama.",
            "The converter will dequantize AWQ weights and re-quantize to GGUF format.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Conversion planning
# ---------------------------------------------------------------------------


def plan_conversion(
    source: str | Path,
    target_format: str,
    output_path: str | Path,
    quantization: str | None = None,
) -> ConversionPlan:
    """Generate a conversion plan for *source* to *target_format*.

    Does **not** execute the conversion — returns a ``ConversionPlan``
    with the shell command the user should run, along with tool install
    instructions and notes.
    """
    source_path = Path(source)
    source_format = detect_format(source_path)

    # Normalize "huggingface" to "hf" for lookup, try both keys.
    lookup_keys = [
        (source_format, target_format),
        (_normalize_format(source_format), _normalize_format(target_format)),
    ]

    entry: dict | None = None
    for key in lookup_keys:
        if key in CONVERSION_COMMANDS:
            entry = CONVERSION_COMMANDS[key]
            break

    if entry is None:
        return ConversionPlan(
            source_format=source_format,
            target_format=target_format,
            source_path=str(source_path),
            command=f"# No known conversion path: {source_format} -> {target_format}",
            tool="unknown",
            install_hint="",
            notes=[
                f"naql does not know how to convert {source_format} -> {target_format}.",
                "You may need to convert to an intermediate format first (e.g. HuggingFace).",
            ],
        )

    quant = quantization or "q4_k_m"
    command = entry["command"].format(
        src=str(source_path),
        dst=str(output_path),
        quant=quant,
    )

    notes = list(entry["notes"])

    # Add Arabic tokenizer warning if relevant.
    arabic_info = check_arabic_tokenizer(source_path)
    if arabic_info["has_arabic"]:
        notes.append(
            f"Source has {arabic_info['arabic_token_count']} Arabic tokens "
            f"({arabic_info['arabic_ratio']:.1%} of vocab). "
            "Verify Arabic tokenizer is preserved after conversion."
        )

    return ConversionPlan(
        source_format=source_format,
        target_format=target_format,
        source_path=str(source_path),
        command=command,
        tool=entry["tool"],
        install_hint=entry["install"],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Post-conversion validation
# ---------------------------------------------------------------------------


def validate_conversion(
    source_info: ModelInfo,
    target_path: str | Path,
) -> list[str]:
    """Validate a conversion result by comparing source and target.

    Checks that the target exists, compares vocab sizes, and verifies
    that Arabic tokenizer support is preserved.  Returns a list of
    warning/error messages — an empty list means everything looks good.
    """
    issues: list[str] = []
    target = Path(target_path)

    # --- Existence check ---
    if not target.exists():
        issues.append(f"Target does not exist: {target}")
        return issues

    # --- Inspect the target ---
    target_info = inspect_model(target)

    if target_info.format == "unknown":
        issues.append(f"Could not detect format of target: {target}")

    # --- Vocab size comparison ---
    if source_info.vocab_size and target_info.vocab_size:
        if source_info.vocab_size != target_info.vocab_size:
            issues.append(
                f"Vocab size mismatch: source={source_info.vocab_size}, "
                f"target={target_info.vocab_size}"
            )

    # --- Architecture check ---
    if source_info.architecture and target_info.architecture:
        src_arch = source_info.architecture.lower()
        tgt_arch = target_info.architecture.lower()
        if src_arch != tgt_arch:
            issues.append(
                f"Architecture mismatch: source={source_info.architecture}, "
                f"target={target_info.architecture}"
            )

    # --- Arabic tokenizer preservation ---
    source_arabic = check_arabic_tokenizer(source_info.path)
    target_arabic = check_arabic_tokenizer(str(target))

    if source_arabic["has_arabic"] and not target_arabic["has_arabic"]:
        issues.append(
            "Arabic tokenizer LOST during conversion! "
            f"Source had {source_arabic['arabic_token_count']} Arabic tokens, "
            "target has none."
        )
    elif source_arabic["has_arabic"] and target_arabic["has_arabic"]:
        src_count = source_arabic["arabic_token_count"]
        tgt_count = target_arabic["arabic_token_count"]
        if abs(src_count - tgt_count) > src_count * 0.05:
            issues.append(
                f"Arabic token count changed significantly: "
                f"source={src_count}, target={tgt_count}"
            )

    # --- Size sanity check ---
    if target_info.size_mb < source_info.size_mb * 0.01:
        issues.append(
            f"Target is suspiciously small: {target_info.size_mb:.1f} MB "
            f"vs source {source_info.size_mb:.1f} MB"
        )

    return issues


# ---------------------------------------------------------------------------
# Model diff
# ---------------------------------------------------------------------------


def diff_models(
    source_info: ModelInfo,
    target_info: ModelInfo,
    source_arabic: dict,
    target_arabic: dict,
) -> list[dict]:
    """Compare two models and return a list of field diffs.

    Each diff is a dict with keys: *field*, *source*, *target*, *status*.
    Status is one of ``"match"``, ``"mismatch"``, ``"warning"``.
    """
    diffs: list[dict] = []

    def _add(field: str, src_val: object, tgt_val: object, *, warn_only: bool = False) -> None:
        src_str = str(src_val) if src_val is not None else "unknown"
        tgt_str = str(tgt_val) if tgt_val is not None else "unknown"
        if src_str == tgt_str:
            status = "match"
        elif warn_only:
            status = "warning"
        else:
            status = "mismatch"
        diffs.append({"field": field, "source": src_str, "target": tgt_str, "status": status})

    _add("Format", source_info.format, target_info.format)
    _add("Architecture", source_info.architecture, target_info.architecture)
    _add("Vocab Size", source_info.vocab_size, target_info.vocab_size)
    _add("Parameters", source_info.num_parameters, target_info.num_parameters, warn_only=True)
    _add("Quantization", source_info.quantization, target_info.quantization, warn_only=True)
    _add("Context Length", source_info.context_length, target_info.context_length)
    _add("Tokenizer", source_info.tokenizer_type, target_info.tokenizer_type, warn_only=True)
    _add(
        "Size",
        f"{source_info.size_mb:.1f} MB",
        f"{target_info.size_mb:.1f} MB",
        warn_only=True,
    )

    # Arabic-specific diffs.
    src_ar = source_arabic.get("arabic_token_count", 0)
    tgt_ar = target_arabic.get("arabic_token_count", 0)
    delta = tgt_ar - src_ar
    delta_str = f"{delta:+d}" if delta != 0 else "0"
    ar_status = "match" if delta == 0 else ("warning" if abs(delta) <= 10 else "mismatch")
    diffs.append({
        "field": "Arabic Tokens",
        "source": str(src_ar),
        "target": f"{tgt_ar} ({delta_str})",
        "status": ar_status,
    })

    src_eff = source_arabic.get("arabic_efficiency_score", 0)
    tgt_eff = target_arabic.get("arabic_efficiency_score", 0)
    _add("Arabic Efficiency", f"{src_eff}/100", f"{tgt_eff}/100", warn_only=True)

    return diffs


# ---------------------------------------------------------------------------
# Supported conversions listing
# ---------------------------------------------------------------------------


def list_supported_conversions() -> list[tuple[str, str]]:
    """Return all supported (source_format, target_format) pairs."""
    return sorted(CONVERSION_COMMANDS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_format(fmt: str) -> str:
    """Normalize format names for lookup consistency.

    Maps ``"huggingface"`` to ``"hf"`` and passes everything else through.
    """
    aliases = {
        "huggingface": "hf",
    }
    return aliases.get(fmt, fmt)

<div align="center">

# naql (&#x0646;&#x0642;&#x0644;) — Arabic Model Format Converter

**Inspect, convert, and validate ML models. Preserve your Arabic tokenizer.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)

</div>

---

## Why

Converting models between GGUF, MLX, ONNX, and SafeTensors often breaks Arabic tokenizers. Token mappings get dropped, vocab indices shift, and your model loses the ability to read and write Arabic.

naql inspects models, plans conversions, and validates that Arabic tokenizer preservation survives the round trip. Born from converting Qwen models for the [Aamil](https://github.com/Moshe-ship/Aamil) iOS app, where a single broken token mapping meant gibberish output on-device.

No heavy dependencies. naql doesn't import torch, mlx, or onnxruntime — it generates the right commands for you, then validates the results.

## Install

```bash
pip install naql
```

## Quick Start

```bash
# Inspect a model file — format, size, quantization, vocab
naql inspect model.gguf

# Check Arabic tokenizer coverage
naql arabic model/

# Convert between formats
naql convert model/ --to mlx

# Validate tokenizer preservation after conversion
naql validate source/ target/
```

## Commands

| Command | Description |
|---------|-------------|
| `inspect` | Inspect a model file or directory. Shows format, size, quantization level, layer count, and vocab size. |
| `arabic` | Scan the tokenizer vocab for Arabic tokens. Reports character coverage, token count, and script distribution. |
| `convert` | Convert a model between formats. Generates the conversion command, runs it, and validates the output. |
| `validate` | Compare source and target models after conversion. Checks vocab alignment, token mapping, and Arabic preservation. |
| `formats` | List all supported formats with detection rules, file extensions, and conversion paths. |
| `explain` | Show how naql works — format detection, conversion pipeline, and Arabic validation steps. |

## Supported Formats

| Format | Extensions | Read | Convert From | Convert To |
|--------|-----------|:----:|:------------:|:----------:|
| **GGUF** | `.gguf` | Yes | Yes | Yes |
| **SafeTensors** | `.safetensors` | Yes | Yes | Yes |
| **ONNX** | `.onnx` | Yes | Yes | Yes |
| **MLX** | `weights.npz`, `*.safetensors` + `config.json` | Yes | Yes | Yes |
| **PyTorch** | `.pt`, `.bin` | Yes | Yes | Yes |
| **HuggingFace** | `model/` directory | Yes | Yes | Yes |

## Arabic Tokenizer Check

naql scans the tokenizer vocabulary and reports Arabic coverage:

```bash
$ naql arabic ./qwen-2b-mlx/

Arabic Tokenizer Report
━━━━━━━━━━━━━━━━━━━━━━━
Total vocab tokens:  151,936
Arabic tokens:       4,217 (2.8%)
Arabic characters:   ✓ All 28 base letters covered
Tashkeel (diacritics): ✓ Present
Arabic digits:       ✓ Present
Common bigrams:      ✓ 94% coverage
Verdict:             GOOD — Arabic tokenizer intact
```

## Conversion Matrix

Which formats can convert to which:

| From ↓ / To → | GGUF | SafeTensors | ONNX | MLX | PyTorch | HuggingFace |
|----------------|:----:|:-----------:|:----:|:---:|:-------:|:-----------:|
| **GGUF** | — | Yes | — | Yes | — | Yes |
| **SafeTensors** | Yes | — | Yes | Yes | Yes | Yes |
| **ONNX** | — | Yes | — | — | Yes | — |
| **MLX** | Yes | Yes | — | — | — | Yes |
| **PyTorch** | Yes | Yes | Yes | Yes | — | Yes |
| **HuggingFace** | Yes | Yes | Yes | Yes | Yes | — |

## Lightweight

naql does not depend on torch, mlx, onnxruntime, or any heavy ML framework. It:

1. **Inspects** model files by reading headers and metadata directly
2. **Generates** the right conversion command for your system
3. **Validates** the output by comparing tokenizer vocab before and after

You install the conversion tools you need separately. naql orchestrates and validates.

---

<p dir="rtl">مقدمة من <a href="https://x.com/i/communities/2032184341682643429">مجتمع الذكاء الاصطناعي السعودي</a> للعرب أولا وللعالم أجمع</p>

Brought to you by the [Saudi AI Community](https://x.com/i/communities/2032184341682643429) — for Arabs first, and the world at large.

## License

MIT — [Musa the Carpenter](https://github.com/Moshe-ship)

---

**The Series:** [artok](https://github.com/Moshe-ship/artok) · [bidi-guard](https://github.com/Moshe-ship/bidi-guard) · [arabench](https://github.com/Moshe-ship/arabench) · [majal](https://github.com/Moshe-ship/majal) · [khalas](https://github.com/Moshe-ship/khalas) · [safha](https://github.com/Moshe-ship/safha) · [raqeeb](https://github.com/Moshe-ship/raqeeb) · [sarih](https://github.com/Moshe-ship/sarih) · [qalam](https://github.com/Moshe-ship/qalam) · [naql](https://github.com/Moshe-ship/naql)

"""ML model file inspector."""
from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Inspection result for a single model file or directory."""

    path: str
    format: str
    size_mb: float
    num_parameters: int | None = None
    architecture: str | None = None
    vocab_size: int | None = None
    has_tokenizer: bool = False
    tokenizer_type: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of the model info.

        Excludes the raw *metadata* dict (which may contain large nested
        structures) and formats *size_mb* to two decimal places.
        """
        d = asdict(self)
        d.pop("metadata", None)
        d["size_mb"] = round(self.size_mb, 2)
        return d


# ---------------------------------------------------------------------------
# Arabic tokenizer analysis
# ---------------------------------------------------------------------------

# Arabic Unicode block: U+0600 .. U+06FF
_ARABIC_RANGE = range(0x0600, 0x06FF + 1)


def check_arabic_tokenizer(path: str | Path) -> dict:
    """Check whether a tokenizer at *path* includes Arabic tokens.

    Looks for ``tokenizer.json`` or ``tokenizer.model`` inside the given
    directory (or treats *path* itself as a tokenizer file).  Returns a
    dict with keys: *has_arabic*, *arabic_token_count*, *total_vocab*,
    *arabic_ratio*, *tokenizer_type*.
    """
    p = Path(path)

    result: dict = {
        "has_arabic": False,
        "arabic_token_count": 0,
        "total_vocab": 0,
        "arabic_ratio": 0.0,
        "tokenizer_type": "unknown",
    }

    tokenizer_path = _find_tokenizer_file(p)
    if tokenizer_path is None:
        return result

    if tokenizer_path.name == "tokenizer.json":
        return _check_arabic_json_tokenizer(tokenizer_path)

    if tokenizer_path.name == "tokenizer.model":
        result["tokenizer_type"] = "sentencepiece"
        # SentencePiece .model files are protobuf — we cannot parse them
        # without the sentencepiece library, so we report what we can.
        return result

    return result


def _find_tokenizer_file(path: Path) -> Path | None:
    """Locate a tokenizer file at or inside *path*."""
    if path.is_file():
        if path.name in ("tokenizer.json", "tokenizer.model"):
            return path
        return None

    for name in ("tokenizer.json", "tokenizer.model"):
        candidate = path / name
        if candidate.is_file():
            return candidate

    return None


def _check_arabic_json_tokenizer(path: Path) -> dict:
    """Parse a HuggingFace ``tokenizer.json`` and count Arabic tokens."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "has_arabic": False,
            "arabic_token_count": 0,
            "total_vocab": 0,
            "arabic_ratio": 0.0,
            "tokenizer_type": "unknown",
        }

    model_section = data.get("model", {})
    vocab: dict = model_section.get("vocab", {})
    total = len(vocab)

    tokenizer_type = model_section.get("type", "unknown")

    arabic_count = 0
    for token in vocab:
        if any(ord(ch) in _ARABIC_RANGE for ch in token):
            arabic_count += 1

    ratio = arabic_count / total if total else 0.0

    return {
        "has_arabic": arabic_count > 0,
        "arabic_token_count": arabic_count,
        "total_vocab": total,
        "arabic_ratio": ratio,
        "tokenizer_type": tokenizer_type,
    }


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"


def detect_format(path: str | Path) -> str:
    """Auto-detect the model format from a file or directory.

    Returns one of: ``"gguf"``, ``"safetensors"``, ``"onnx"``,
    ``"mlx"``, ``"pytorch"``, ``"huggingface"``, or ``"unknown"``.
    """
    p = Path(path)

    # --- Directory-based formats ---
    if p.is_dir():
        return _detect_directory_format(p)

    # --- File-based formats ---
    if not p.is_file():
        return "unknown"

    return _detect_file_format(p)


def _detect_directory_format(path: Path) -> str:
    """Detect format for a directory containing model files."""
    has_config = (path / "config.json").is_file()

    # MLX: directory with weights/ folder containing .npz or .safetensors
    weights_dir = path / "weights"
    if weights_dir.is_dir():
        weight_files = list(weights_dir.glob("*.npz")) + list(
            weights_dir.glob("*.safetensors")
        )
        if weight_files:
            return "mlx"

    # HuggingFace: config.json + model weights + optional tokenizer
    if has_config:
        has_model = (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
            or any(path.glob("model-*.safetensors"))
            or any(path.glob("pytorch_model-*.bin"))
        )
        if has_model:
            return "huggingface"

    # MLX variant: config.json + *.safetensors directly in dir (no weights/)
    if has_config and any(path.glob("*.safetensors")):
        return "mlx"

    return "unknown"


def _detect_file_format(path: Path) -> str:
    """Detect format for a single model file."""
    suffix = path.suffix.lower()

    # Extension-based detection for unambiguous formats.
    if suffix == ".onnx":
        return "onnx"
    if suffix in (".pt", ".pth", ".bin"):
        return "pytorch"
    if suffix == ".safetensors":
        return "safetensors"

    # Magic-bytes detection for GGUF.
    if suffix in (".gguf", ""):
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic == _GGUF_MAGIC:
                return "gguf"
        except OSError:
            pass

    # Safetensors can also be detected by JSON header.
    if suffix == "":
        if _looks_like_safetensors(path):
            return "safetensors"

    return "unknown"


def _looks_like_safetensors(path: Path) -> bool:
    """Return *True* if *path* starts with a safetensors JSON header."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            header_len = struct.unpack("<Q", raw)[0]
            # Sanity check — header should be under 100 MB.
            return 0 < header_len < 100_000_000
    except OSError:
        return False


# ---------------------------------------------------------------------------
# GGUF inspection
# ---------------------------------------------------------------------------


def inspect_gguf(path: str | Path) -> ModelInfo:
    """Parse a GGUF file header and extract metadata.

    Reads the version, tensor count, and metadata key-value pairs from
    the binary header.  Architecture, vocab size, and context length are
    extracted from the metadata when available.
    """
    p = Path(path)
    size_mb = p.stat().st_size / (1024 * 1024)

    metadata: dict = {}
    architecture: str | None = None
    vocab_size: int | None = None
    context_length: int | None = None

    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return ModelInfo(
                    path=str(p), format="gguf", size_mb=size_mb,
                    metadata={"error": "invalid GGUF magic bytes"},
                )

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            metadata["gguf_version"] = version
            metadata["tensor_count"] = tensor_count
            metadata["kv_count"] = kv_count

            # Read metadata KV pairs.
            kv_pairs = _read_gguf_metadata(f, kv_count)
            metadata["kv_pairs"] = kv_pairs

            # Extract well-known keys.
            architecture = kv_pairs.get("general.architecture")
            vocab_size_raw = kv_pairs.get("tokenizer.ggml.tokens")
            if isinstance(vocab_size_raw, list):
                vocab_size = len(vocab_size_raw)
            elif isinstance(vocab_size_raw, int):
                vocab_size = vocab_size_raw

            ctx_key = f"{architecture}.context_length" if architecture else None
            if ctx_key and ctx_key in kv_pairs:
                context_length = kv_pairs[ctx_key]

    except (OSError, struct.error) as exc:
        metadata["error"] = str(exc)

    # Guess quantization from filename (e.g. Q4_K_M, Q8_0, 4bit).
    quantization = _guess_quantization(p.name)

    # Check for a tokenizer in the same directory.
    has_tokenizer = _find_tokenizer_file(p.parent) is not None

    return ModelInfo(
        path=str(p),
        format="gguf",
        size_mb=size_mb,
        num_parameters=None,
        architecture=architecture,
        vocab_size=vocab_size,
        has_tokenizer=has_tokenizer,
        tokenizer_type="gguf_internal" if vocab_size else None,
        quantization=quantization,
        context_length=context_length,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# GGUF metadata reader
# ---------------------------------------------------------------------------

# GGUF metadata value type codes.
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


def _read_gguf_string(f) -> str:
    """Read a GGUF-encoded string (uint64 length + UTF-8 bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, value_type: int):
    """Read a single GGUF metadata value of *value_type*."""
    if value_type == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if value_type == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if value_type == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if value_type == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if value_type == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if value_type == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if value_type == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == _GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    if value_type == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if value_type == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if value_type == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if value_type == _GGUF_TYPE_ARRAY:
        element_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        # For large arrays (e.g. token lists) only store the count to
        # avoid reading gigabytes of vocab data into memory.
        if count > 10_000:
            # Skip past the array data — estimate sizes for common types.
            _skip_gguf_array(f, element_type, count)
            return count
        return [_read_gguf_value(f, element_type) for _ in range(count)]

    # Unknown type — cannot continue parsing.
    raise ValueError(f"unknown GGUF value type: {value_type}")


def _skip_gguf_array(f, element_type: int, count: int) -> None:
    """Skip over a large GGUF array without reading every element."""
    fixed_sizes = {
        _GGUF_TYPE_UINT8: 1, _GGUF_TYPE_INT8: 1,
        _GGUF_TYPE_UINT16: 2, _GGUF_TYPE_INT16: 2,
        _GGUF_TYPE_UINT32: 4, _GGUF_TYPE_INT32: 4,
        _GGUF_TYPE_FLOAT32: 4, _GGUF_TYPE_BOOL: 1,
        _GGUF_TYPE_UINT64: 8, _GGUF_TYPE_INT64: 8,
        _GGUF_TYPE_FLOAT64: 8,
    }

    if element_type in fixed_sizes:
        f.seek(fixed_sizes[element_type] * count, os.SEEK_CUR)
    elif element_type == _GGUF_TYPE_STRING:
        for _ in range(count):
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, os.SEEK_CUR)
    else:
        # Nested arrays or unknown — read element by element as fallback.
        for _ in range(count):
            _read_gguf_value(f, element_type)


def _read_gguf_metadata(f, kv_count: int) -> dict:
    """Read *kv_count* metadata key-value pairs from a GGUF file."""
    pairs: dict = {}

    for _ in range(kv_count):
        try:
            key = _read_gguf_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = _read_gguf_value(f, value_type)
            pairs[key] = value
        except (struct.error, ValueError, OSError):
            break

    return pairs


# ---------------------------------------------------------------------------
# SafeTensors inspection
# ---------------------------------------------------------------------------


def inspect_safetensors(path: str | Path) -> ModelInfo:
    """Parse a safetensors file header to extract tensor information.

    The header is a JSON object at the start of the file preceded by an
    8-byte little-endian length.  Tensor names and shapes are extracted
    to estimate the total parameter count.
    """
    p = Path(path)
    size_mb = p.stat().st_size / (1024 * 1024)

    metadata: dict = {}
    num_parameters: int | None = None

    try:
        with open(p, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                return ModelInfo(
                    path=str(p), format="safetensors", size_mb=size_mb,
                    metadata={"error": "header too large"},
                )
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes)

        # __metadata__ is a special key in safetensors headers.
        st_meta = header.pop("__metadata__", {})
        metadata["safetensors_metadata"] = st_meta

        # Count parameters from tensor shapes.
        total_params = 0
        tensor_names: list[str] = []
        for name, info in header.items():
            tensor_names.append(name)
            shape = info.get("shape", [])
            if shape:
                param_count = 1
                for dim in shape:
                    param_count *= dim
                total_params += param_count

        num_parameters = total_params if total_params > 0 else None
        metadata["tensor_count"] = len(tensor_names)
        metadata["tensor_names_sample"] = tensor_names[:20]

    except (OSError, struct.error, json.JSONDecodeError) as exc:
        metadata["error"] = str(exc)

    # Check for tokenizer in the same directory.
    has_tokenizer = _find_tokenizer_file(p.parent) is not None

    return ModelInfo(
        path=str(p),
        format="safetensors",
        size_mb=size_mb,
        num_parameters=num_parameters,
        has_tokenizer=has_tokenizer,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# MLX inspection
# ---------------------------------------------------------------------------


def inspect_mlx(path: str | Path) -> ModelInfo:
    """Inspect an MLX model directory by reading its ``config.json``.

    Expects a directory containing ``config.json`` and a ``weights/``
    subdirectory (or ``.safetensors`` files at the top level).
    """
    p = Path(path)
    size_mb = _directory_size_mb(p)
    metadata: dict = {}

    config_path = p / "config.json"
    if not config_path.is_file():
        return ModelInfo(
            path=str(p), format="mlx", size_mb=size_mb,
            metadata={"error": "missing config.json"},
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ModelInfo(
            path=str(p), format="mlx", size_mb=size_mb,
            metadata={"error": str(exc)},
        )

    metadata["config"] = config

    architecture = config.get("model_type") or config.get("architectures", [None])[0]
    vocab_size = config.get("vocab_size")
    context_length = (
        config.get("max_position_embeddings")
        or config.get("max_sequence_length")
    )

    # Detect quantization from config or weights filename patterns.
    quantization = config.get("quantization") or _guess_quantization(p.name)

    # Check for tokenizer.
    has_tokenizer = _find_tokenizer_file(p) is not None
    tokenizer_type: str | None = None
    if (p / "tokenizer.json").is_file():
        tokenizer_type = "json"
    elif (p / "tokenizer.model").is_file():
        tokenizer_type = "sentencepiece"

    return ModelInfo(
        path=str(p),
        format="mlx",
        size_mb=size_mb,
        architecture=architecture,
        vocab_size=vocab_size,
        has_tokenizer=has_tokenizer,
        tokenizer_type=tokenizer_type,
        quantization=quantization,
        context_length=context_length,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# HuggingFace inspection
# ---------------------------------------------------------------------------


def inspect_huggingface(path: str | Path) -> ModelInfo:
    """Inspect a HuggingFace model directory.

    Reads ``config.json`` for architecture details and checks for
    tokenizer files alongside the model weights.
    """
    p = Path(path)
    size_mb = _directory_size_mb(p)
    metadata: dict = {}

    config_path = p / "config.json"
    if not config_path.is_file():
        return ModelInfo(
            path=str(p), format="huggingface", size_mb=size_mb,
            metadata={"error": "missing config.json"},
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ModelInfo(
            path=str(p), format="huggingface", size_mb=size_mb,
            metadata={"error": str(exc)},
        )

    metadata["config"] = config

    architecture = config.get("model_type")
    if not architecture:
        archs = config.get("architectures", [])
        architecture = archs[0] if archs else None

    vocab_size = config.get("vocab_size")
    context_length = (
        config.get("max_position_embeddings")
        or config.get("max_sequence_length")
    )

    quantization = config.get("quantization_config", {}).get("quant_method")

    # Tokenizer detection.
    has_tokenizer = _find_tokenizer_file(p) is not None
    tokenizer_type: str | None = None
    if (p / "tokenizer.json").is_file():
        tokenizer_type = "json"
    elif (p / "tokenizer.model").is_file():
        tokenizer_type = "sentencepiece"

    # Count parameters from safetensors if present.
    num_parameters: int | None = None
    st_file = p / "model.safetensors"
    if st_file.is_file():
        st_info = inspect_safetensors(st_file)
        num_parameters = st_info.num_parameters

    return ModelInfo(
        path=str(p),
        format="huggingface",
        size_mb=size_mb,
        num_parameters=num_parameters,
        architecture=architecture,
        vocab_size=vocab_size,
        has_tokenizer=has_tokenizer,
        tokenizer_type=tokenizer_type,
        quantization=quantization,
        context_length=context_length,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def inspect_model(path: str | Path) -> ModelInfo:
    """Auto-detect the model format at *path* and return full inspection.

    This is the main entry point — callers should prefer this over the
    format-specific functions unless they already know the format.
    """
    p = Path(path)
    fmt = detect_format(p)

    dispatch = {
        "gguf": inspect_gguf,
        "safetensors": inspect_safetensors,
        "mlx": inspect_mlx,
        "huggingface": inspect_huggingface,
    }

    handler = dispatch.get(fmt)
    if handler is not None:
        return handler(p)

    # For formats we detect but don't deeply inspect, return basic info.
    if p.is_file():
        size_mb = p.stat().st_size / (1024 * 1024)
    elif p.is_dir():
        size_mb = _directory_size_mb(p)
    else:
        size_mb = 0.0

    return ModelInfo(path=str(p), format=fmt, size_mb=size_mb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guess_quantization(name: str) -> str | None:
    """Try to guess quantization type from a filename.

    Common patterns: ``Q4_K_M``, ``Q8_0``, ``4bit``, ``8bit``, ``f16``.
    """
    import re

    # GGUF quantization tags (Q4_K_M, Q5_0, IQ2_XS, etc.)
    m = re.search(r"[_\-\.]((?:I?Q\d+_\w+)|(?:F(?:16|32)))[_\-\.]", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Simple bit-width tags (4bit, 8bit).
    m = re.search(r"(\d+)[\-_]?bit", name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}bit"

    return None


def _directory_size_mb(path: Path) -> float:
    """Return the total size of all files under *path* in megabytes."""
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        pass
    return total / (1024 * 1024)

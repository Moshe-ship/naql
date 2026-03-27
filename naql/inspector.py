"""ML model file inspector."""
from __future__ import annotations

import json
import os
import re
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

# Common Arabic words expected as single tokens in a good Arabic tokenizer.
_COMMON_ARABIC_WORDS = [
    "\u0627\u0644\u0630\u0643\u0627\u0621",      # الذكاء
    "\u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a",  # الاصطناعي
    "\u0645\u0631\u062d\u0628\u0627",      # مرحبا
    "\u0627\u0644\u0639\u0627\u0644\u0645",  # العالم
    "\u0628\u0633\u0645",          # بسم
    "\u0627\u0644\u0644\u0647",      # الله
]


def _has_arabic_char(text: str) -> bool:
    """Return *True* if *text* contains at least one Arabic character."""
    return any(ord(ch) in _ARABIC_RANGE for ch in text)


def _count_arabic_bigrams(tokens: list[str]) -> int:
    """Count tokens that are Arabic bigrams or trigrams (2-3 Arabic chars)."""
    count = 0
    for token in tokens:
        arabic_chars = [ch for ch in token if ord(ch) in _ARABIC_RANGE]
        if 2 <= len(arabic_chars) <= 3 and len(arabic_chars) == len(token.strip()):
            count += 1
    return count


def _check_common_arabic_words(tokens: list[str]) -> tuple[int, list[str]]:
    """Check which common Arabic words appear as single tokens.

    Returns (count_found, list_of_found_words).
    """
    token_set = set(tokens)
    found: list[str] = []
    for word in _COMMON_ARABIC_WORDS:
        if word in token_set:
            found.append(word)
    return len(found), found


def _calculate_arabic_efficiency(
    arabic_count: int,
    total_vocab: int,
    bigram_count: int,
    common_words_found: int,
) -> int:
    """Calculate an Arabic efficiency score from 0 to 100.

    Weighted: 40% single-char ratio, 30% bigram presence, 30% common words.
    """
    if total_vocab == 0:
        return 0

    # Single-char coverage: ratio of Arabic tokens to total, capped at 10%
    # being considered "excellent".
    char_ratio = min(arabic_count / total_vocab / 0.10, 1.0)

    # Bigram coverage: expect at least 500 bigrams for a good tokenizer.
    bigram_ratio = min(bigram_count / 500, 1.0)

    # Common words: 6 target words.
    common_ratio = common_words_found / len(_COMMON_ARABIC_WORDS)

    score = int(char_ratio * 40 + bigram_ratio * 30 + common_ratio * 30)
    return min(score, 100)


def check_arabic_tokenizer(path: str | Path) -> dict:
    """Check whether a tokenizer at *path* includes Arabic tokens.

    Looks for ``tokenizer.json`` or ``tokenizer.model`` inside the given
    directory (or treats *path* itself as a tokenizer file).  Also checks
    GGUF embedded tokenizers.  Returns a dict with keys: *has_arabic*,
    *arabic_token_count*, *total_vocab*, *arabic_ratio*, *tokenizer_type*,
    *arabic_bigram_count*, *common_words_found*, *common_words_list*,
    *arabic_efficiency_score*.
    """
    p = Path(path)

    result: dict = {
        "has_arabic": False,
        "arabic_token_count": 0,
        "total_vocab": 0,
        "arabic_ratio": 0.0,
        "tokenizer_type": "unknown",
        "arabic_bigram_count": 0,
        "common_words_found": 0,
        "common_words_list": [],
        "arabic_efficiency_score": 0,
    }

    # Check if path is a GGUF file with embedded tokenizer.
    if p.is_file() and p.suffix.lower() == ".gguf":
        return _check_arabic_gguf_tokenizer(p)

    tokenizer_path = _find_tokenizer_file(p)
    if tokenizer_path is None:
        # If the directory contains a GGUF file, try its embedded tokenizer.
        if p.is_dir():
            gguf_files = list(p.glob("*.gguf"))
            if gguf_files:
                return _check_arabic_gguf_tokenizer(gguf_files[0])
        return result

    if tokenizer_path.name == "tokenizer.json":
        return _check_arabic_json_tokenizer(tokenizer_path)

    if tokenizer_path.name == "tokenizer.model":
        return _check_arabic_sentencepiece_tokenizer(tokenizer_path)

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
            "arabic_bigram_count": 0,
            "common_words_found": 0,
            "common_words_list": [],
            "arabic_efficiency_score": 0,
        }

    model_section = data.get("model", {})
    vocab: dict = model_section.get("vocab", {})
    total = len(vocab)

    tokenizer_type = model_section.get("type", "unknown")

    token_list = list(vocab.keys())
    arabic_count = sum(1 for t in token_list if _has_arabic_char(t))
    bigram_count = _count_arabic_bigrams(token_list)
    common_count, common_list = _check_common_arabic_words(token_list)

    ratio = arabic_count / total if total else 0.0
    efficiency = _calculate_arabic_efficiency(
        arabic_count, total, bigram_count, common_count,
    )

    return {
        "has_arabic": arabic_count > 0,
        "arabic_token_count": arabic_count,
        "total_vocab": total,
        "arabic_ratio": ratio,
        "tokenizer_type": tokenizer_type,
        "arabic_bigram_count": bigram_count,
        "common_words_found": common_count,
        "common_words_list": common_list,
        "arabic_efficiency_score": efficiency,
    }


def _check_arabic_sentencepiece_tokenizer(path: Path) -> dict:
    """Scan a SentencePiece ``.model`` file for Arabic UTF-8 byte sequences.

    SentencePiece ``.model`` files are serialised protobuf.  We do a raw
    binary scan rather than requiring the ``sentencepiece`` library.
    Arabic UTF-8 bytes start with 0xD8-0xDB (first byte of 2-byte Arabic).
    """
    result: dict = {
        "has_arabic": False,
        "arabic_token_count": 0,
        "total_vocab": 0,
        "arabic_ratio": 0.0,
        "tokenizer_type": "sentencepiece",
        "arabic_bigram_count": 0,
        "common_words_found": 0,
        "common_words_list": [],
        "arabic_efficiency_score": 0,
    }

    try:
        raw = path.read_bytes()
    except OSError:
        return result

    # Extract printable UTF-8 string-like segments from the protobuf.
    # SentencePiece tokens are stored as field type 2 (length-delimited)
    # in the protobuf wire format.  We use a heuristic: find sequences
    # of valid UTF-8 that contain Arabic characters.
    #
    # Arabic UTF-8 encoding: U+0600..U+06FF maps to bytes D8 80..DB BF.
    arabic_token_count = 0
    total_tokens_estimate = 0
    arabic_tokens: list[str] = []

    # Scan for length-delimited strings in protobuf — look for UTF-8
    # sequences between 1 and 50 bytes that decode cleanly.
    i = 0
    while i < len(raw) - 1:
        # Heuristic: look for Arabic first bytes (0xD8..0xDB)
        if 0xD8 <= raw[i] <= 0xDB and i + 1 < len(raw) and 0x80 <= raw[i + 1] <= 0xBF:
            # Found potential Arabic UTF-8 — back-scan for token start.
            # Try to decode a window around this position.
            start = max(0, i - 30)
            end = min(len(raw), i + 50)
            try:
                snippet = raw[start:end].decode("utf-8", errors="ignore")
                # Extract Arabic substrings.
                for match in re.finditer(r'[\u0600-\u06FF]+', snippet):
                    token_text = match.group()
                    arabic_tokens.append(token_text)
            except Exception:
                pass
            i += 2
        else:
            i += 1

    # Deduplicate — each unique Arabic piece counts once.
    unique_arabic = set(arabic_tokens)
    arabic_token_count = len(unique_arabic)

    # Estimate total vocab from file size heuristic (rough).
    # A typical SentencePiece model has ~30 bytes per token on average.
    total_tokens_estimate = max(len(raw) // 30, arabic_token_count)

    # Count bigrams in unique tokens.
    bigram_count = _count_arabic_bigrams(list(unique_arabic))
    common_count, common_list = _check_common_arabic_words(list(unique_arabic))

    ratio = arabic_token_count / total_tokens_estimate if total_tokens_estimate else 0.0
    efficiency = _calculate_arabic_efficiency(
        arabic_token_count, total_tokens_estimate, bigram_count, common_count,
    )

    result.update({
        "has_arabic": arabic_token_count > 0,
        "arabic_token_count": arabic_token_count,
        "total_vocab": total_tokens_estimate,
        "arabic_ratio": ratio,
        "arabic_bigram_count": bigram_count,
        "common_words_found": common_count,
        "common_words_list": common_list,
        "arabic_efficiency_score": efficiency,
    })
    return result


def _check_arabic_gguf_tokenizer(path: Path) -> dict:
    """Extract vocab from a GGUF file's embedded tokenizer and check Arabic.

    Reads the ``tokenizer.ggml.tokens`` array from the GGUF metadata.
    """
    result: dict = {
        "has_arabic": False,
        "arabic_token_count": 0,
        "total_vocab": 0,
        "arabic_ratio": 0.0,
        "tokenizer_type": "gguf_internal",
        "arabic_bigram_count": 0,
        "common_words_found": 0,
        "common_words_list": [],
        "arabic_efficiency_score": 0,
    }

    try:
        info = inspect_gguf(path)
    except Exception:
        return result

    kv_pairs = info.metadata.get("kv_pairs", {})
    tokens = kv_pairs.get("tokenizer.ggml.tokens")

    if not isinstance(tokens, list):
        return result

    total = len(tokens)
    arabic_count = sum(1 for t in tokens if isinstance(t, str) and _has_arabic_char(t))
    str_tokens = [t for t in tokens if isinstance(t, str)]
    bigram_count = _count_arabic_bigrams(str_tokens)
    common_count, common_list = _check_common_arabic_words(str_tokens)

    ratio = arabic_count / total if total else 0.0
    efficiency = _calculate_arabic_efficiency(
        arabic_count, total, bigram_count, common_count,
    )

    result.update({
        "has_arabic": arabic_count > 0,
        "arabic_token_count": arabic_count,
        "total_vocab": total,
        "arabic_ratio": ratio,
        "arabic_bigram_count": bigram_count,
        "common_words_found": common_count,
        "common_words_list": common_list,
        "arabic_efficiency_score": efficiency,
    })
    return result


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

    # JANG: check before MLX since JANG is a subset of MLX format.
    if _is_jang_directory(path):
        return "jang"

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


def _is_jang_directory(path: Path) -> bool:
    """Return *True* if *path* looks like a JANG model directory.

    JANG is an MLX variant that uses mixed-bit quantization.  Detected by:
    - Directory name contains "JANG" (case-insensitive), OR
    - ``config.json`` has a ``"quantization"`` key with mixed bit values
      (e.g. different bits_per_weight for attention vs MLP layers).
    """
    if "jang" in path.name.lower():
        config_path = path / "config.json"
        if config_path.is_file():
            return True

    config_path = path / "config.json"
    if not config_path.is_file():
        return False

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return False

    # JANG uses mixed-bit quantization — look for group-specific configs
    # or a "group_size" alongside multiple "bits" values in nested dicts.
    group_bits: set[int] = set()
    for key, value in quantization.items():
        if isinstance(value, dict) and "bits" in value:
            group_bits.add(value["bits"])
        elif key == "bits":
            group_bits.add(value)

    # Mixed bits means at least 2 different bit widths.
    return len(group_bits) >= 2


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
    the binary header.  Architecture, vocab size, context length, and
    architecture-specific metadata (attention heads, layers, embedding
    dimensions) are extracted.  Individual tensor info (names, shapes,
    quantization types) is stored in ``metadata["tensors"]``.
    """
    p = Path(path)
    size_mb = p.stat().st_size / (1024 * 1024)

    metadata: dict = {}
    architecture: str | None = None
    vocab_size: int | None = None
    context_length: int | None = None
    num_parameters: int | None = None

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

            # --- Architecture-specific metadata ---
            if architecture:
                arch_meta: dict = {}
                head_count_key = f"{architecture}.attention.head_count"
                if head_count_key in kv_pairs:
                    arch_meta["attention_heads"] = kv_pairs[head_count_key]
                head_count_kv_key = f"{architecture}.attention.head_count_kv"
                if head_count_kv_key in kv_pairs:
                    arch_meta["attention_heads_kv"] = kv_pairs[head_count_kv_key]
                block_count_key = f"{architecture}.block_count"
                if block_count_key in kv_pairs:
                    arch_meta["layers"] = kv_pairs[block_count_key]
                embed_key = f"{architecture}.embedding_length"
                if embed_key in kv_pairs:
                    arch_meta["embedding_length"] = kv_pairs[embed_key]
                feed_forward_key = f"{architecture}.feed_forward_length"
                if feed_forward_key in kv_pairs:
                    arch_meta["feed_forward_length"] = kv_pairs[feed_forward_key]
                if arch_meta:
                    metadata["architecture_details"] = arch_meta

            # --- Count Arabic tokens in GGUF embedded vocab ---
            if isinstance(vocab_size_raw, list):
                arabic_count = sum(
                    1 for t in vocab_size_raw
                    if isinstance(t, str) and _has_arabic_char(t)
                )
                metadata["arabic_token_count"] = arabic_count

            # --- Read tensor info from the header ---
            tensors = _read_gguf_tensor_info(f, tensor_count, version)
            if tensors:
                metadata["tensors"] = tensors
                # Estimate parameters from tensor shapes.
                total_params = 0
                for ti in tensors:
                    shape = ti.get("shape", [])
                    if shape:
                        params = 1
                        for dim in shape:
                            params *= dim
                        total_params += params
                if total_params > 0:
                    num_parameters = total_params

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
        num_parameters=num_parameters,
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
        # Exception: string arrays up to 300k are read (tokenizer vocab).
        if count > 10_000:
            if element_type == _GGUF_TYPE_STRING and count <= 300_000:
                return [_read_gguf_value(f, element_type) for _ in range(count)]
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


# GGUF tensor quantization type names (from ggml-common.h).
_GGUF_TENSOR_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M",
}


def _read_gguf_tensor_info(f, tensor_count: int, version: int) -> list[dict]:
    """Read tensor info entries from the GGUF header.

    Each tensor info contains: name, n_dimensions, dimensions, type, offset.
    Returns a list of dicts with keys: *name*, *shape*, *type*, *offset*.
    """
    tensors: list[dict] = []

    for _ in range(tensor_count):
        try:
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            tensor_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            type_name = _GGUF_TENSOR_TYPE_NAMES.get(tensor_type, f"type_{tensor_type}")

            tensors.append({
                "name": name,
                "shape": dims,
                "type": type_name,
                "offset": offset,
            })
        except (struct.error, ValueError, OSError):
            break

    return tensors


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
# JANG inspection
# ---------------------------------------------------------------------------


def inspect_jang(path: str | Path) -> ModelInfo:
    """Inspect a JANG model directory (MLX mixed-bit quantization variant).

    JANG directories contain ``config.json`` with a ``"quantization"``
    section that specifies different bit widths for attention vs MLP
    layers.  Extracts average bits, attention bits, and MLP bits.
    """
    p = Path(path)
    size_mb = _directory_size_mb(p)
    metadata: dict = {}

    config_path = p / "config.json"
    if not config_path.is_file():
        return ModelInfo(
            path=str(p), format="jang", size_mb=size_mb,
            metadata={"error": "missing config.json"},
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ModelInfo(
            path=str(p), format="jang", size_mb=size_mb,
            metadata={"error": str(exc)},
        )

    metadata["config"] = config

    architecture = config.get("model_type") or config.get("architectures", [None])[0]
    vocab_size = config.get("vocab_size")
    context_length = (
        config.get("max_position_embeddings")
        or config.get("max_sequence_length")
    )

    # Parse JANG mixed-bit quantization details.
    quant_config = config.get("quantization", {})
    jang_meta: dict = {}
    all_bits: list[int] = []
    attention_bits: int | None = None
    mlp_bits: int | None = None

    if isinstance(quant_config, dict):
        for key, value in quant_config.items():
            if isinstance(value, dict) and "bits" in value:
                bits = value["bits"]
                all_bits.append(bits)
                key_lower = key.lower()
                if "attn" in key_lower or "attention" in key_lower or "self" in key_lower:
                    attention_bits = bits
                elif "mlp" in key_lower or "ffn" in key_lower or "feed" in key_lower:
                    mlp_bits = bits
            elif key == "bits":
                all_bits.append(value)

    if all_bits:
        avg_bits = sum(all_bits) / len(all_bits)
        jang_meta["average_bits"] = round(avg_bits, 2)
        jang_meta["all_bits"] = all_bits
    if attention_bits is not None:
        jang_meta["attention_bits"] = attention_bits
    if mlp_bits is not None:
        jang_meta["mlp_bits"] = mlp_bits

    metadata["jang_quantization"] = jang_meta

    # Build quantization description string.
    quant_parts: list[str] = []
    if attention_bits is not None:
        quant_parts.append(f"attn={attention_bits}bit")
    if mlp_bits is not None:
        quant_parts.append(f"mlp={mlp_bits}bit")
    if all_bits and not quant_parts:
        quant_parts.append(f"mixed-{'/'.join(str(b) for b in sorted(set(all_bits)))}bit")
    quantization = "jang:" + ",".join(quant_parts) if quant_parts else "jang"

    # Check for tokenizer.
    has_tokenizer = _find_tokenizer_file(p) is not None
    tokenizer_type: str | None = None
    if (p / "tokenizer.json").is_file():
        tokenizer_type = "json"
    elif (p / "tokenizer.model").is_file():
        tokenizer_type = "sentencepiece"

    return ModelInfo(
        path=str(p),
        format="jang",
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
        "jang": inspect_jang,
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

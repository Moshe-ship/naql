"""naql - Arabic Model Format Converter."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ModelInfo",
    "inspect_model",
    "inspect_jang",
    "detect_format",
    "check_arabic_tokenizer",
    "diff_models",
]

from naql.converter import diff_models
from naql.inspector import (
    ModelInfo,
    check_arabic_tokenizer,
    detect_format,
    inspect_jang,
    inspect_model,
)

"""naql - Arabic Model Format Converter."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ModelInfo",
    "inspect_model",
    "detect_format",
    "check_arabic_tokenizer",
]

from naql.inspector import ModelInfo, check_arabic_tokenizer, detect_format, inspect_model

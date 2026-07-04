"""
One-call industrial ASR post-processing API.
"""

from __future__ import annotations

from pathlib import Path

from industrial_normalizer import normalize_text
from term_corrector import correct_terms


def postprocess_text(
    text: str,
    config_path: str | Path | None = None,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 0.92,
) -> dict:
    normalized = normalize_text(text)
    corrected = correct_terms(
        normalized["text"],
        config_path=config_path,
        enable_fuzzy=enable_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
    )
    return {
        "raw_text": text or "",
        "norm_text": normalized["text"],
        "final_text": corrected["text"],
        "fixed_terms": corrected["fixed_terms"],
        "correction_log": normalized["correction_log"] + corrected["correction_log"],
    }

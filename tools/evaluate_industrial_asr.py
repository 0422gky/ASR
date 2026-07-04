"""
Evaluate industrial ASR post-processing from a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ASR_ROOT = Path(__file__).resolve().parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from industrial_postprocess import postprocess_text
from term_corrector import load_terms


OUTPUT_FIELDS = [
    "id",
    "audio_path",
    "ref_text",
    "asr_text",
    "norm_text",
    "final_text",
    "cer_before",
    "cer_after",
    "term_acc_before",
    "term_acc_after",
    "fixed_terms",
    "error_type",
]


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def term_accuracy(ref: str, hyp: str, terms) -> float:
    expected = [term.canonical for term in terms if term.canonical in (ref or "")]
    if not expected:
        return 1.0
    hits = sum(1 for term in expected if term in (hyp or ""))
    return hits / len(expected)


def classify(cer_before: float, cer_after: float, term_before: float, term_after: float) -> str:
    labels = []
    if cer_after < cer_before:
        labels.append("improved")
    elif cer_after > cer_before:
        labels.append("regressed")
    else:
        labels.append("unchanged")
    if term_after > term_before:
        labels.append("term_improved")
    elif term_after < term_before:
        labels.append("term_regressed")
    return "+".join(labels)


def evaluate(input_path: Path, output_path: Path, config_path: str | None, enable_fuzzy: bool) -> None:
    terms = load_terms(config_path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        missing = {"id", "audio_path", "ref_text", "asr_text"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")

        writer = csv.DictWriter(dst, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in reader:
            result = postprocess_text(
                row.get("asr_text", ""),
                config_path=config_path,
                enable_fuzzy=enable_fuzzy,
            )
            before = cer(row.get("ref_text", ""), row.get("asr_text", ""))
            after = cer(row.get("ref_text", ""), result["final_text"])
            term_before = term_accuracy(row.get("ref_text", ""), row.get("asr_text", ""), terms)
            term_after = term_accuracy(row.get("ref_text", ""), result["final_text"], terms)
            writer.writerow({
                "id": row.get("id", ""),
                "audio_path": row.get("audio_path", ""),
                "ref_text": row.get("ref_text", ""),
                "asr_text": row.get("asr_text", ""),
                "norm_text": result["norm_text"],
                "final_text": result["final_text"],
                "cer_before": f"{before:.6f}",
                "cer_after": f"{after:.6f}",
                "term_acc_before": f"{term_before:.6f}",
                "term_acc_after": f"{term_after:.6f}",
                "fixed_terms": json.dumps(result["fixed_terms"], ensure_ascii=False),
                "error_type": classify(before, after, term_before, term_after),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate industrial ASR post-processing")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--terms-config", default=None, help="industrial_terms.yaml path")
    parser.add_argument("--enable-fuzzy", action="store_true", help="Enable conservative fuzzy correction")
    args = parser.parse_args()

    evaluate(Path(args.input), Path(args.output), args.terms_config, args.enable_fuzzy)


if __name__ == "__main__":
    main()

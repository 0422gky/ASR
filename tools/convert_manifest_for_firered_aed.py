"""
Validate and optionally convert generic fine-tuning manifests for FireRedASR2-AED.

Input JSONL rows are expected to contain at least:
  utt_id, audio, text

The output JSONL keeps only rows with existing audio and non-empty text. No audio
files are copied.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"utt_id", "audio", "text"}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".tsv":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 3:
                    rows.append({
                        "utt_id": row[0],
                        "audio": row[1],
                        "text": row[2],
                        "source": row[3] if len(row) > 3 else "unknown",
                        "speaker": row[4] if len(row) > 4 else "unknown",
                        "env": row[5] if len(row) > 5 else "unknown",
                    })
        return rows
    raise ValueError(f"Unsupported manifest format: {path}")


def resolve_audio(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_rows(rows: list[dict[str, Any]], manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = []
    skipped = []
    missing_audio = 0
    missing_text = 0
    missing_fields = 0

    for idx, row in enumerate(rows):
        if not REQUIRED_FIELDS.issubset(row):
            missing_fields += 1
            skipped.append({"index": idx, "reason": "missing_required_fields", "row": row})
            continue
        audio_path = resolve_audio(str(row["audio"]), manifest_path)
        text = str(row.get("text", "")).strip()
        if not audio_path.exists():
            missing_audio += 1
            skipped.append({"utt_id": row.get("utt_id"), "reason": "missing_audio", "audio": str(row.get("audio"))})
            continue
        if not text:
            missing_text += 1
            skipped.append({"utt_id": row.get("utt_id"), "reason": "empty_text", "audio": str(row.get("audio"))})
            continue
        fixed = dict(row)
        fixed["audio"] = str(audio_path)
        fixed["text"] = text
        valid.append(fixed)

    stats = {
        "input_rows": len(rows),
        "valid_rows": len(valid),
        "missing_audio": missing_audio,
        "missing_text": missing_text,
        "missing_required_fields": missing_fields,
        "source_counts": dict(Counter(row.get("source", "unknown") for row in valid)),
        "skipped": skipped[:200],
        "skipped_total": len(skipped),
    }
    return valid, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/convert FireRedASR2-AED fine-tune manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_manifest(input_path)
    valid, stats = validate_rows(rows, input_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.dry_run:
        return
    if args.output:
        write_jsonl(Path(args.output), valid)
    if args.stats_output:
        Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

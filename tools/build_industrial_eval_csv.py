"""
Build evaluate_industrial_asr.py input CSV from transcription outputs.

The batch processor writes post-processed .txt files and correction logs. This
tool reconstructs the pre-correction ASR text from the correction log where
possible, joins it with the numbered reference script, and emits:

id,audio_path,ref_text,asr_text
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".aac", ".wma")
CSV_FIELDS = ["id", "audio_path", "ref_text", "asr_text"]


def load_references(reference_path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for raw_line in reference_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)[\.、]\s*(.+)$", line)
        if match:
            refs[match.group(1)] = match.group(2).strip()
    return refs


def infer_reference_id(stem: str) -> str | None:
    match = re.search(r"-(\d+)$", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)号", stem)
    if match:
        return match.group(1)
    return None


def index_audio_files(audio_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def load_correction_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else data.get("correction_log", [])


def reconstruct_asr_text(final_text: str, correction_log: list[dict]) -> str:
    text = final_text
    for entry in reversed(correction_log):
        source = entry.get("source")
        replacement = entry.get("replacement")
        if not source or not replacement:
            continue
        text = text.replace(str(replacement), str(source), 1)
    return text


def build_rows(
    transcriptions_dir: Path,
    reference_path: Path,
    audio_root: Path,
    use_reconstructed_asr: bool = True,
) -> list[dict[str, str]]:
    refs = load_references(reference_path)
    audio_index = index_audio_files(audio_root)
    rows: list[dict[str, str]] = []

    for txt_path in sorted(transcriptions_dir.glob("*.txt")):
        ref_id = infer_reference_id(txt_path.stem)
        if not ref_id or ref_id not in refs:
            continue

        final_text = txt_path.read_text(encoding="utf-8").strip()
        correction_log = load_correction_log(txt_path.with_suffix(".correction_log.json"))
        asr_text = (
            reconstruct_asr_text(final_text, correction_log)
            if use_reconstructed_asr
            else final_text
        )
        audio_path = audio_index.get(txt_path.stem)

        rows.append({
            "id": txt_path.stem,
            "audio_path": str(audio_path) if audio_path else "",
            "ref_text": refs[ref_id],
            "asr_text": asr_text,
        })

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluate_industrial_asr.py input CSV from batch outputs"
    )
    parser.add_argument(
        "--transcriptions-dir",
        default="train_audio_folder/transcriptions",
        help="Directory containing transcription .txt and .correction_log.json files",
    )
    parser.add_argument(
        "--reference",
        default="train_audio_folder/测试语音文本原稿.txt",
        help="Numbered reference text file",
    )
    parser.add_argument(
        "--audio-root",
        default="train_audio_folder",
        help="Root directory used to find matching audio files",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--use-final-text",
        action="store_true",
        help="Use .txt content directly as asr_text instead of reversing correction logs",
    )
    args = parser.parse_args()

    rows = build_rows(
        Path(args.transcriptions_dir),
        Path(args.reference),
        Path(args.audio_root),
        use_reconstructed_asr=not args.use_final_text,
    )
    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

"""
Evaluate FireRedASR2-AED on a JSONL manifest with industrial metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ASR_ROOT = Path(__file__).resolve().parents[1]
FIRERED_ROOT = ASR_ROOT / "FireRedASR2S"
FIRERED_PACKAGE_ROOT = FIRERED_ROOT / "fireredasr2s"
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))
if str(FIRERED_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRERED_PACKAGE_ROOT))

try:
    from fireredasr2 import FireRedAsr2, FireRedAsr2Config
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install FireRedASR2S requirements first:\n"
        f"  pip install -r FireRedASR2S/requirements.txt"
    ) from exc
from industrial_postprocess import postprocess_text
from tools.evaluate_industrial_asr import cer, term_accuracy
from term_corrector import load_terms


FOCUS_TERMS = [
    "AcOPOStrak", "AcOPOSmulti", "AcOPOS P3", "ACOPOSD1", "ParID",
    "mcTCs坐标系", "A工位", "B工位", "C工位", "20秒", "40度", "-1067186135",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ASR_ROOT / path


def load_rows(path: Path, max_items: int = 0) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if max_items and len(rows) >= max_items:
            break
    return rows


def contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def number_accuracy(ref: str, hyp: str) -> float:
    ref_nums = re.findall(r"-?\d+(?:\.\d+)?", ref or "")
    if not ref_nums:
        return 1.0
    return sum(1 for num in ref_nums if num in (hyp or "")) / len(ref_nums)


def focus_accuracy(ref: str, hyp: str, terms: list[str]) -> float:
    expected = [term for term in terms if term in (ref or "")]
    if not expected:
        return 1.0
    return sum(1 for term in expected if term in (hyp or "")) / len(expected)


def run_eval(model_dir: Path, manifest: Path, output_csv: Path, postprocess: bool, max_items: int) -> dict:
    rows = load_rows(manifest, max_items=max_items)
    config = FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=1, nbest=1)
    model = FireRedAsr2.from_pretrained("aed", str(model_dir), config)
    terms = load_terms()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cers = []
    term_accs = []
    num_accs = []
    err_accs = []
    station_accs = []
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "audio", "ref_text", "hyp_text", "final_text", "cer", "term_acc", "number_acc", "error_code_acc", "station_acc"])
        writer.writeheader()
        for row in rows:
            result = model.transcribe([row["utt_id"]], [str(resolve_path(row["audio"]))])[0]
            hyp = result.get("text", "")
            final = postprocess_text(hyp)["final_text"] if postprocess else hyp
            ref = row["text"]
            row_cer = cer(ref, final)
            row_term = max(term_accuracy(ref, final, terms), focus_accuracy(ref, final, FOCUS_TERMS))
            row_num = number_accuracy(ref, final)
            row_err = focus_accuracy(ref, final, ["-1067186135"])
            row_station = focus_accuracy(ref, final, ["A工位", "B工位", "C工位"])
            cers.append(row_cer)
            term_accs.append(row_term)
            num_accs.append(row_num)
            err_accs.append(row_err)
            station_accs.append(row_station)
            writer.writerow({
                "utt_id": row["utt_id"],
                "audio": row["audio"],
                "ref_text": ref,
                "hyp_text": hyp,
                "final_text": final,
                "cer": f"{row_cer:.6f}",
                "term_acc": f"{row_term:.6f}",
                "number_acc": f"{row_num:.6f}",
                "error_code_acc": f"{row_err:.6f}",
                "station_acc": f"{row_station:.6f}",
            })

    summary = {
        "items": len(rows),
        "cer": sum(cers) / max(1, len(cers)),
        "term_accuracy": sum(term_accs) / max(1, len(term_accs)),
        "number_accuracy": sum(num_accs) / max(1, len(num_accs)),
        "error_code_accuracy": sum(err_accs) / max(1, len(err_accs)),
        "station_accuracy": sum(station_accs) / max(1, len(station_accs)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FireRedASR2-AED manifest")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()
    run_eval(resolve_path(args.model_dir), resolve_path(args.manifest), resolve_path(args.output_csv), args.postprocess, args.max_items)


if __name__ == "__main__":
    main()

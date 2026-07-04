"""
Build mixed fine-tuning manifests for FireRedASR2-AED.

The current FireRedASR2S package exposes inference code, so this script writes
generic JSONL manifests plus simple TSV files that are easy to adapt to a
future training entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ASR_ROOT = Path(__file__).resolve().parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from industrial_postprocess import postprocess_text


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"}
NON_SPEECH_LABELS = {"[ENS]", "[NPS]", "[*]", "[LAUGHTER]", "[SONANT]"}
JSONL_FILES = ("train_mix.jsonl", "dev_mix.jsonl")
TSV_FILES = ("train_mix.tsv", "dev_mix.tsv")


@dataclass
class Sample:
    utt_id: str
    audio: str
    text: str
    source: str
    speaker: str = "unknown"
    env: str = "unknown"
    duration: float | None = None
    start: float | None = None
    end: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class SourceStats:
    raw_samples: int = 0
    train_before_repeat: int = 0
    dev: int = 0
    train_after_repeat: int = 0
    total_duration: float | None = 0.0
    missing_transcript: int = 0
    missing_audio: int = 0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def read_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def add_skip(stats: SourceStats, reason: str, item: str, **extra: Any) -> None:
    stats.skipped.append({"reason": reason, "item": item, **extra})


def parse_aishell(root: Path) -> tuple[list[Sample], SourceStats]:
    stats = SourceStats()
    transcript_dir = root / "transcript"
    transcripts: dict[str, str] = {}
    for transcript_path in sorted(transcript_dir.glob("*.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = clean_text(parts[1])

    wavs: dict[str, Path] = {}
    for base in (root / "train", root / "wav"):
        if base.exists():
            for wav_path in sorted(base.rglob("*.wav")):
                wavs[wav_path.stem] = wav_path

    samples: list[Sample] = []
    for utt_id, wav_path in sorted(wavs.items()):
        text = transcripts.get(utt_id, "")
        if not text:
            stats.missing_transcript += 1
            add_skip(stats, "missing_transcript", utt_id, audio=str(wav_path))
            continue
        if not wav_path.exists():
            stats.missing_audio += 1
            add_skip(stats, "missing_audio", utt_id, audio=str(wav_path))
            continue
        duration = read_duration(wav_path)
        samples.append(Sample(
            utt_id=f"aishell_{utt_id}",
            audio=rel_or_abs(wav_path),
            text=text,
            source="aishell",
            speaker=utt_id[6:11] if len(utt_id) >= 11 else "unknown",
            env="clean",
            duration=duration,
        ))

    stats.raw_samples = len(samples)
    stats.total_duration = sum(s.duration or 0.0 for s in samples)
    return samples, stats


def parse_magichub_utterance_info(root: Path) -> dict[str, dict[str, str]]:
    info_path = root / "UTTERANCEINFO.txt"
    info: dict[str, dict[str, str]] = {}
    if not info_path.exists():
        return info
    reader = csv.DictReader(info_path.read_text(encoding="utf-8").splitlines(), delimiter="\t")
    for row in reader:
        utt = row.get("UTTRANS_ID", "")
        if utt:
            info[Path(utt).stem] = row
    return info


def parse_magichub_txt(txt_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^\[([0-9.]+),([0-9.]+)\]\t([^\t]+)\t([^\t]+)\t(.+)$")
    for idx, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines()):
        match = pattern.match(line.strip())
        if not match:
            continue
        start = float(match.group(1))
        end = float(match.group(2))
        speaker = match.group(3).strip()
        gender = match.group(4).strip()
        text = clean_text(match.group(5))
        if speaker == "0" or text in NON_SPEECH_LABELS:
            continue
        rows.append({
            "index": idx,
            "start": start,
            "end": end,
            "speaker": speaker or "unknown",
            "gender": gender,
            "text": text,
        })
    return rows


def parse_magichub(root: Path) -> tuple[list[Sample], SourceStats]:
    stats = SourceStats()
    wav_dir = root / "WAV"
    txt_dir = root / "TXT"
    wavs = {path.stem: path for path in sorted(wav_dir.glob("*.wav"))}
    txts = {path.stem: path for path in sorted(txt_dir.glob("*.txt"))}
    info = parse_magichub_utterance_info(root)

    for stem in sorted(set(wavs) - set(txts)):
        stats.missing_transcript += 1
        add_skip(stats, "missing_transcript", stem, audio=str(wavs[stem]))
    for stem in sorted(set(txts) - set(wavs)):
        stats.missing_audio += 1
        add_skip(stats, "missing_audio", stem, transcript=str(txts[stem]))

    samples: list[Sample] = []
    for stem in sorted(set(wavs) & set(txts)):
        wav_path = wavs[stem]
        meta = info.get(stem, {})
        for segment in parse_magichub_txt(txts[stem]):
            text = segment["text"]
            if not text:
                add_skip(stats, "empty_text", stem, transcript=str(txts[stem]))
                continue
            start = segment["start"]
            end = segment["end"]
            samples.append(Sample(
                utt_id=f"magichub_{stem}_{segment['index']:04d}",
                audio=rel_or_abs(wav_path),
                text=text,
                source="magichub",
                speaker=segment["speaker"] or meta.get("SPEAKER_ID", "unknown"),
                env=meta.get("TOPIC", "development") or "development",
                duration=max(0.0, end - start),
                start=start,
                end=end,
            ))

    stats.raw_samples = len(samples)
    stats.total_duration = sum(s.duration or 0.0 for s in samples)
    return samples, stats


def load_numbered_references(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    if not path.exists():
        return refs
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+)[\.、]\s*(.+)$", line.strip())
        if match:
            refs[match.group(1)] = clean_text(match.group(2))
    return refs


def infer_number_from_stem(stem: str) -> str | None:
    match = re.search(r"-(\d+)$", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)号", stem)
    if match:
        return match.group(1)
    return None


def infer_speaker_env(stem: str, parent: str) -> tuple[str, str]:
    parts = stem.split("-")
    speaker = parts[0] if parts else parent or "unknown"
    env = f"env{parts[1]}" if len(parts) >= 3 and parts[1].isdigit() else "unknown"
    return speaker or "unknown", env


def normalize_industrial_text(text: str, utt_id: str, correction_log: list[dict[str, Any]]) -> str:
    current = text
    current = re.sub(r"(?<=\d)[oO](?=\s*(秒|度|分钟|毫米|厘米|米))", "0", current)
    if current != text:
        correction_log.append({
            "utt_id": utt_id,
            "rule": "suspicious_digit_o",
            "source": text,
            "replacement": current,
        })

    result = postprocess_text(current)
    for item in result["correction_log"]:
        correction_log.append({"utt_id": utt_id, **item})
    return result["final_text"]


def parse_industrial_csv(csv_path: Path, correction_log: list[dict[str, Any]], stats: SourceStats) -> list[Sample]:
    samples: list[Sample] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not {"audio", "text"}.issubset(fieldnames):
            stats.warnings.append(
                f"ignore csv without required audio/text columns: {csv_path}"
            )
            return []
        for row in reader:
            utt_id = row.get("utt_id") or row.get("id") or Path(row.get("audio", "")).stem
            audio = row.get("audio", "")
            if not audio:
                stats.missing_audio += 1
                add_skip(stats, "missing_audio", utt_id, audio="")
                continue
            audio_path = (csv_path.parent / audio).resolve() if audio and not Path(audio).is_absolute() else Path(audio)
            text = clean_text(row.get("text", ""))
            if not audio_path.exists():
                stats.missing_audio += 1
                add_skip(stats, "missing_audio", utt_id, audio=str(audio_path))
                continue
            if not text:
                stats.missing_transcript += 1
                add_skip(stats, "empty_text", utt_id, audio=str(audio_path))
                continue
            text = normalize_industrial_text(text, utt_id, correction_log)
            samples.append(Sample(
                utt_id=f"industrial_{utt_id}",
                audio=rel_or_abs(audio_path),
                text=text,
                source="industrial",
                speaker=row.get("speaker") or "unknown",
                env=row.get("env") or "unknown",
                duration=read_duration(audio_path),
            ))
    return samples


def parse_industrial(root: Path, correction_log: list[dict[str, Any]]) -> tuple[list[Sample], SourceStats]:
    stats = SourceStats()
    csv_candidates = sorted(root.glob("*.csv"))
    for csv_path in csv_candidates:
        samples = parse_industrial_csv(csv_path, correction_log, stats)
        if samples:
            stats.raw_samples = len(samples)
            stats.total_duration = sum(s.duration or 0.0 for s in samples)
            return samples, stats

    refs = load_numbered_references(root / "测试语音文本原稿.txt")
    same_stem_txts = {path.stem: path for path in root.rglob("*.txt") if path.parent.name != "transcriptions"}
    audio_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)

    samples: list[Sample] = []
    for audio_path in audio_files:
        text = ""
        if audio_path.stem in same_stem_txts:
            text = clean_text(same_stem_txts[audio_path.stem].read_text(encoding="utf-8"))
        else:
            ref_id = infer_number_from_stem(audio_path.stem)
            if ref_id:
                text = refs.get(ref_id, "")

        if not text:
            stats.missing_transcript += 1
            add_skip(stats, "missing_transcript", audio_path.stem, audio=str(audio_path))
            continue

        speaker, env = infer_speaker_env(audio_path.stem, audio_path.parent.name)
        utt_id = f"industrial_{audio_path.stem}"
        text = normalize_industrial_text(text, utt_id, correction_log)
        samples.append(Sample(
            utt_id=utt_id,
            audio=rel_or_abs(audio_path),
            text=text,
            source="industrial",
            speaker=speaker,
            env=env,
            duration=read_duration(audio_path),
        ))

    stats.raw_samples = len(samples)
    stats.total_duration = sum(s.duration or 0.0 for s in samples)
    return samples, stats


def split_by_audio(samples: list[Sample], dev_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.audio].append(sample)

    audio_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(audio_keys)
    dev_count = int(round(len(audio_keys) * dev_ratio))
    if dev_ratio > 0 and len(audio_keys) > 1:
        dev_count = max(1, dev_count)
    dev_audio = set(audio_keys[:dev_count])

    train: list[Sample] = []
    dev: list[Sample] = []
    for audio in audio_keys:
        if audio in dev_audio:
            dev.extend(groups[audio])
        else:
            train.extend(groups[audio])
    return sorted(train, key=lambda s: s.utt_id), sorted(dev, key=lambda s: s.utt_id)


def repeat_train_samples(samples: list[Sample], repeat: int) -> list[Sample]:
    repeated: list[Sample] = []
    for sample in samples:
        for rep_idx in range(1, max(1, repeat) + 1):
            item = deepcopy(sample)
            item.utt_id = f"{sample.utt_id}_rep{rep_idx:02d}"
            repeated.append(item)
    return repeated


def find_audio_leakage(train: list[Sample], dev: list[Sample]) -> list[str]:
    train_audio = {sample.audio for sample in train}
    dev_audio = {sample.audio for sample in dev}
    return sorted(train_audio & dev_audio)


def write_jsonl(path: Path, samples: list[Sample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_manifest(), ensure_ascii=False) + "\n")


def write_tsv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for sample in samples:
            writer.writerow([sample.utt_id, sample.audio, sample.text, sample.source, sample.speaker, sample.env])


def source_summary(stats: dict[str, SourceStats]) -> dict[str, Any]:
    return {source: asdict(value) for source, value in stats.items()}


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    correction_log: list[dict[str, Any]] = []
    all_stats: dict[str, SourceStats] = {}

    aishell, all_stats["aishell"] = parse_aishell(Path(args.aishell_root))
    magichub, all_stats["magichub"] = parse_magichub(Path(args.magichub_root))
    industrial, all_stats["industrial"] = parse_industrial(Path(args.industrial_root), correction_log)

    sources = {
        "aishell": (aishell, args.aishell_repeat),
        "magichub": (magichub, args.magichub_repeat),
        "industrial": (industrial, args.industrial_repeat),
    }

    train_mix: list[Sample] = []
    dev_mix: list[Sample] = []
    train_before_repeat: list[Sample] = []

    for source, (samples, repeat) in sources.items():
        if source == "industrial" and args.use_all_industrial_train:
            train, dev = sorted(samples, key=lambda s: s.utt_id), []
        else:
            train, dev = split_by_audio(samples, args.dev_ratio, args.seed)
        repeated = repeat_train_samples(train, repeat)

        all_stats[source].train_before_repeat = len(train)
        all_stats[source].dev = len(dev)
        all_stats[source].train_after_repeat = len(repeated)
        train_before_repeat.extend(train)
        train_mix.extend(repeated)
        dev_mix.extend(dev)

    leakage = find_audio_leakage(train_before_repeat, dev_mix)
    if leakage:
        for stats in all_stats.values():
            stats.warnings.append(f"train/dev audio leakage: {len(leakage)} audio paths")

    output = {
        "sources": source_summary(all_stats),
        "totals": {
            "train_before_repeat": len(train_before_repeat),
            "train_after_repeat": len(train_mix),
            "dev": len(dev_mix),
            "corrections": len(correction_log),
            "train_dev_audio_leakage": leakage,
        },
    }

    if args.dry_run:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train_mix.jsonl", train_mix)
    write_jsonl(out_dir / "dev_mix.jsonl", dev_mix)
    write_tsv(out_dir / "train_mix.tsv", train_mix)
    write_tsv(out_dir / "dev_mix.tsv", dev_mix)
    (out_dir / "data_stats.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "correction_log.json").write_text(json.dumps(correction_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(output["totals"], ensure_ascii=False, indent=2))
    print(f"Wrote manifests to {out_dir}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed FireRedASR2-AED fine-tuning manifests")
    parser.add_argument("--aishell-root", default="data/AISHELL")
    parser.add_argument("--magichub-root", default="data/MagicHub/development")
    parser.add_argument("--industrial-root", default="train_audio_folder")
    parser.add_argument("--out-dir", default="manifests")
    parser.add_argument("--industrial-repeat", type=int, default=10)
    parser.add_argument("--magichub-repeat", type=int, default=3)
    parser.add_argument("--aishell-repeat", type=int, default=1)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-all-industrial-train", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_manifests(args)


if __name__ == "__main__":
    main()

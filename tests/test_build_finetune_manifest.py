import argparse
import csv
import json
import sys
import wave
from pathlib import Path

ASR_ROOT = Path(__file__).resolve().parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from tools.build_finetune_manifest import (
    build_manifests,
    parse_aishell,
    parse_industrial,
    parse_magichub,
    split_by_audio,
)


def write_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 160)


def test_parse_aishell_matches_local_wavs(tmp_path):
    root = tmp_path / "AISHELL"
    write_wav(root / "train" / "S0002" / "BAC009S0002W0122.wav")
    (root / "transcript").mkdir(parents=True)
    (root / "transcript" / "aishell.txt").write_text(
        "BAC009S0002W0122 你好 世界\nBAC009S0002W9999 缺失 音频\n",
        encoding="utf-8",
    )

    samples, stats = parse_aishell(root)

    assert len(samples) == 1
    assert samples[0].utt_id == "aishell_BAC009S0002W0122"
    assert samples[0].text == "你好 世界"
    assert stats.missing_transcript == 0


def test_parse_magichub_segments_and_filters_non_speech(tmp_path):
    root = tmp_path / "MagicHub" / "development"
    write_wav(root / "WAV" / "G0001_S0001_0_SPK003.wav")
    txt_dir = root / "TXT"
    txt_dir.mkdir(parents=True)
    (txt_dir / "G0001_S0001_0_SPK003.txt").write_text(
        "[0.000,1.000]\t0\tnone\t[ENS]\n"
        "[1.000,2.500]\tSPK003\t女\t北京爱数智慧语言云采集。\n",
        encoding="utf-8",
    )

    samples, _ = parse_magichub(root)

    assert len(samples) == 1
    assert samples[0].utt_id == "magichub_G0001_S0001_0_SPK003_0001"
    assert samples[0].speaker == "SPK003"
    assert samples[0].start == 1.0
    assert samples[0].end == 2.5


def test_parse_industrial_numbered_reference_and_correction(tmp_path):
    root = tmp_path / "train_audio_folder"
    write_wav(root / "人声样本1" / "YQL-1-5.wav")
    (root / "测试语音文本原稿.txt").write_text(
        "5. 帮我移动小车从A工位到B工位，在B工位停顿2o秒后移动到c工位。\n",
        encoding="utf-8",
    )
    correction_log = []

    samples, stats = parse_industrial(root, correction_log)

    assert stats.raw_samples == 1
    assert samples[0].text == "帮我移动小车从A工位到B工位，在B工位停顿20秒后移动到C工位。"
    assert samples[0].speaker == "YQL"
    assert samples[0].env == "env1"
    assert any(item["rule"] == "suspicious_digit_o" for item in correction_log)


def test_split_by_audio_prevents_segment_leakage():
    from tools.build_finetune_manifest import Sample

    samples = [
        Sample("a1", "same.wav", "一", "magichub"),
        Sample("a2", "same.wav", "二", "magichub"),
        Sample("b1", "other.wav", "三", "magichub"),
    ]
    train, dev = split_by_audio(samples, dev_ratio=0.5, seed=1)

    assert {s.audio for s in train}.isdisjoint({s.audio for s in dev})


def test_build_manifests_outputs_files_and_repeats(tmp_path):
    aishell = tmp_path / "data" / "AISHELL"
    write_wav(aishell / "train" / "S0002" / "BAC009S0002W0122.wav")
    (aishell / "transcript").mkdir(parents=True)
    (aishell / "transcript" / "aishell.txt").write_text("BAC009S0002W0122 你好\n", encoding="utf-8")

    magichub = tmp_path / "data" / "MagicHub" / "development"
    write_wav(magichub / "WAV" / "G0001_S0001_0_SPK003.wav")
    (magichub / "TXT").mkdir(parents=True)
    (magichub / "TXT" / "G0001_S0001_0_SPK003.txt").write_text(
        "[1.000,2.000]\tSPK003\t女\t你好。\n",
        encoding="utf-8",
    )

    industrial = tmp_path / "train_audio_folder"
    write_wav(industrial / "人声样本1" / "YQL-1-1.wav")
    (industrial / "测试语音文本原稿.txt").write_text("1. 如何检查ACOPOStrak轨道？\n", encoding="utf-8")

    out_dir = tmp_path / "manifests"
    args = argparse.Namespace(
        aishell_root=str(aishell),
        magichub_root=str(magichub),
        industrial_root=str(industrial),
        out_dir=str(out_dir),
        industrial_repeat=2,
        magichub_repeat=1,
        aishell_repeat=1,
        dev_ratio=0,
        seed=42,
        use_all_industrial_train=False,
        dry_run=False,
    )

    build_manifests(args)

    train_rows = [json.loads(line) for line in (out_dir / "train_mix.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == 4
    assert any(row["utt_id"].endswith("_rep02") for row in train_rows)
    assert (out_dir / "dev_mix.jsonl").exists()
    assert (out_dir / "train_mix.tsv").exists()
    assert (out_dir / "data_stats.json").exists()


def test_dry_run_writes_nothing(tmp_path):
    out_dir = tmp_path / "manifests"
    args = argparse.Namespace(
        aishell_root=str(tmp_path / "missing_aishell"),
        magichub_root=str(tmp_path / "missing_magic"),
        industrial_root=str(tmp_path / "missing_industrial"),
        out_dir=str(out_dir),
        industrial_repeat=10,
        magichub_repeat=3,
        aishell_repeat=1,
        dev_ratio=0.1,
        seed=42,
        use_all_industrial_train=False,
        dry_run=True,
    )

    build_manifests(args)

    assert not out_dir.exists()

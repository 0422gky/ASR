"""
Small-scale FireRedASR2-AED fine-tuning entrypoint.

This script is intentionally conservative:
- no LoRA is injected because this repo does not provide a LoRA training path
  for its custom AED modules;
- encoder is frozen by default;
- smoke mode limits training to a few steps.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ASR_ROOT = Path(__file__).resolve().parents[1]
FIRERED_ROOT = ASR_ROOT / "FireRedASR2S"
FIRERED_PACKAGE_ROOT = FIRERED_ROOT / "fireredasr2s"
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))
if str(FIRERED_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRERED_PACKAGE_ROOT))

try:
    from fireredasr2.models.fireredasr_aed import FireRedAsrAed
    from fireredasr2.data.asr_feat import ASRFeatExtractor
    from fireredasr2.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install FireRedASR2S requirements first:\n"
        f"  pip install -r FireRedASR2S/requirements.txt"
    ) from exc


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to read YAML config {path}: {exc}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ASR_ROOT / path


class JsonlManifestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.rows = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("audio") and row.get("text"):
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def load_audio(path: str) -> tuple[int, np.ndarray]:
    audio_path = Path(path)
    if not audio_path.is_absolute():
        audio_path = ASR_ROOT / audio_path
    try:
        import soundfile as sf

        wav, sample_rate = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return int(sample_rate), wav.astype(np.float32)
    except Exception:
        import torchaudio

        wav_tensor, sample_rate = torchaudio.load(str(audio_path))
        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        return int(sample_rate), wav_tensor.squeeze(0).numpy().astype(np.float32)


def collate_batch(batch: list[dict[str, Any]], feat_extractor: ASRFeatExtractor, tokenizer: ChineseCharEnglishSpmTokenizer):
    uttids = [str(row["utt_id"]) for row in batch]
    audio_data = [load_audio(str(row["audio"])) for row in batch]
    feats, feat_lengths, durs, _, uttids = feat_extractor(audio_data, uttids)
    token_ids = []
    kept_rows = []
    kept_durs = []
    for row, dur in zip(batch, durs):
        _, ids = tokenizer.tokenize(str(row["text"]))
        if ids:
            token_ids.append(torch.tensor(ids, dtype=torch.long))
            kept_rows.append(row)
            kept_durs.append(dur)
    if not token_ids or feats is None:
        return None
    max_len = max(x.numel() for x in token_ids)
    pad_id = tokenizer.dict.get("<blank>", 0)
    targets = torch.full((len(token_ids), max_len), pad_id, dtype=torch.long)
    target_lengths = torch.tensor([x.numel() for x in token_ids], dtype=torch.long)
    for i, ids in enumerate(token_ids):
        targets[i, : ids.numel()] = ids
    return {
        "rows": kept_rows,
        "feats": feats[: len(token_ids)],
        "feat_lengths": feat_lengths[: len(token_ids)],
        "targets": targets,
        "target_lengths": target_lengths,
        "durations": kept_durs,
    }


def decoder_logits(model, enc_outputs, enc_mask, targets):
    decoder = model.decoder
    batch_size = targets.size(0)
    sos = torch.full((batch_size, 1), decoder.sos_id, dtype=torch.long, device=targets.device)
    eos = torch.full((batch_size, 1), decoder.eos_id, dtype=torch.long, device=targets.device)
    y_in = torch.cat([sos, targets], dim=1)
    y_out = torch.cat([targets, eos], dim=1)
    tgt_mask = decoder.ignored_target_position_is_0(y_in, decoder.pad_id)
    dec_output = decoder.dropout(
        decoder.tgt_word_emb(y_in) * decoder.scale + decoder.positional_encoding(y_in)
    )
    for layer in decoder.layer_stack:
        dec_output = layer(dec_output, enc_outputs, tgt_mask, enc_mask, cache=None)
    dec_output = decoder.layer_norm_out(dec_output)
    return decoder.tgt_word_prj(dec_output), y_out


def compute_loss(model, batch, ctc_weight: float):
    feats = batch["feats"]
    feat_lengths = batch["feat_lengths"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    enc_outputs, enc_lengths, enc_mask = model.encoder(feats, feat_lengths)
    logits, y_out = decoder_logits(model, enc_outputs, enc_mask, targets)
    att_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y_out.reshape(-1),
        ignore_index=model.decoder.pad_id,
    )
    if ctc_weight <= 0:
        return att_loss, {"att_loss": float(att_loss.detach().cpu()), "ctc_loss": 0.0}
    ctc_log_probs = model.ctc(enc_outputs).transpose(0, 1)
    ctc_loss = F.ctc_loss(
        ctc_log_probs,
        targets,
        enc_lengths.long(),
        target_lengths.long(),
        blank=0,
        zero_infinity=True,
    )
    loss = (1.0 - ctc_weight) * att_loss + ctc_weight * ctc_loss
    return loss, {
        "att_loss": float(att_loss.detach().cpu()),
        "ctc_loss": float(ctc_loss.detach().cpu()),
    }


def configure_trainable(model, freeze_encoder: bool, train_decoder: bool, train_ctc: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if not freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True
    if train_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = True
    if train_ctc:
        for param in model.ctc.parameters():
            param.requires_grad = True


def set_training_mode(model, freeze_encoder: bool) -> None:
    model.train()
    if freeze_encoder:
        model.encoder.eval()


@torch.no_grad()
def evaluate_loss(model, loader, device, ctc_weight: float, max_batches: int) -> dict[str, float]:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if batch is None:
            continue
        batch = move_batch(batch, device)
        loss, _ = compute_loss(model, batch, ctc_weight)
        losses.append(float(loss.cpu()))
        if max_batches and idx + 1 >= max_batches:
            break
    return {"loss": sum(losses) / max(1, len(losses)), "batches": len(losses)}


def move_batch(batch, device):
    for key in ("feats", "feat_lengths", "targets", "target_lengths"):
        batch[key] = batch[key].to(device)
    return batch


def save_checkpoint(path: Path, model, optimizer, step: int, epoch: int, source_model_path: Path) -> None:
    package = torch.load(source_model_path, map_location="cpu", weights_only=False)
    package["model_state_dict"] = model.state_dict()
    package["optimizer_state_dict"] = optimizer.state_dict()
    package["finetune_step"] = step
    package["finetune_epoch"] = epoch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)


def load_aed_model(model_path: Path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FireRedASR2-AED")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(int(cfg.get("seed", 42)))
    model_dir = resolve_path(cfg["model_dir"])
    source_model_path = model_dir / "model.pth.tar"
    tokenizer = ChineseCharEnglishSpmTokenizer(str(model_dir / "dict.txt"), str(model_dir / "train_bpe1000.model"))
    feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
    model = load_aed_model(source_model_path)

    use_gpu = bool(cfg.get("use_gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    configure_trainable(
        model,
        freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        train_decoder=bool(cfg.get("train_decoder", True)),
        train_ctc=bool(cfg.get("train_ctc", True)),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable}")

    train_dataset = JsonlManifestDataset(resolve_path(cfg["train_manifest"]))
    dev_dataset = JsonlManifestDataset(resolve_path(cfg["dev_manifest"]))
    collate = lambda batch: collate_batch(batch, feat_extractor, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolve_path(args.config), out_dir / "config.yaml")

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(cfg.get("smoke_max_steps", 5)) if args.smoke_test else int(cfg.get("max_steps", 0))
    ctc_weight = float(cfg.get("ctc_weight", 0.3))
    grad_clip = float(cfg.get("grad_clip", 5.0))
    step = 0
    freeze_encoder = bool(cfg.get("freeze_encoder", True))
    set_training_mode(model, freeze_encoder)

    for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
        for batch in train_loader:
            if batch is None:
                continue
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_loss(model, batch, ctc_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            step += 1
            print(f"step={step} epoch={epoch} loss={float(loss.detach().cpu()):.4f} att={parts['att_loss']:.4f} ctc={parts['ctc_loss']:.4f}")
            if step % int(cfg.get("save_every_steps", 100)) == 0:
                save_checkpoint(out_dir / f"checkpoint_step{step}.pth.tar", model, optimizer, step, epoch, source_model_path)
            if max_steps and step >= max_steps:
                break
        eval_result = evaluate_loss(model, dev_loader, device, ctc_weight, int(cfg.get("eval_max_batches", 10)))
        print(f"dev_loss={eval_result['loss']:.4f} dev_batches={eval_result['batches']}")
        set_training_mode(model, freeze_encoder)
        if max_steps and step >= max_steps:
            break

    save_checkpoint(out_dir / "model.pth.tar", model, optimizer, step, epoch, source_model_path)
    for name in ("dict.txt", "train_bpe1000.model", "cmvn.ark"):
        src = model_dir / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
    print(f"saved_checkpoint={out_dir / 'model.pth.tar'}")


if __name__ == "__main__":
    main()

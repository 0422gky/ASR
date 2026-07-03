"""
共享工具模块 - FireRedASR2-AED 语音识别

提供:
  - 模型加载 (单例模式, 避免重复加载 GPU 显存)
  - 音频格式转换 (mp3/ogg/... → 16kHz 16-bit mono WAV)
  - 路径解析

使用方法:
  from asr_utils import get_model, transcribe_file
"""

import os
import sys
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("asr_utils")

# ─── 路径设置 ───────────────────────────────────────────────
_CURRENT_DIR = Path(__file__).resolve().parent
_FIRERED_ROOT = _CURRENT_DIR / "FireRedASR2S"

# 将 FireRedASR2S 加入 Python path, 以便 import fireredasr2s
if str(_FIRERED_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIRERED_ROOT))

# AED 模型默认路径
DEFAULT_MODEL_DIR = _FIRERED_ROOT / "pretrained_models" / "FireRedASR2-AED"

# 全局模型缓存 (单例)
_model_cache = None


# ─── 音频转换 ───────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用."""
    return shutil.which("ffmpeg") is not None


def convert_to_wav(audio_path: str, output_dir: str = None) -> str:
    """
    将任意音频格式转换为 16kHz 16-bit 单声道 PCM WAV.

    FireRedASR2-AED 只接受此格式. 转换策略:
      1) 已是 .wav → 直接返回 (假定格式正确)
      2) ffmpeg 可用 → 使用 ffmpeg 转换 (推荐)
      3) 否则 → 使用 torchaudio 转换

    Args:
        audio_path: 输入音频文件路径 (.mp3 / .wav / .ogg / ...)
        output_dir:  转换输出目录 (默认: 系统临时目录)

    Returns:
        转换后的 .wav 文件路径
    """
    if audio_path.lower().endswith('.wav'):
        return audio_path

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="firered_wav_")

    basename = Path(audio_path).stem
    wav_path = os.path.join(output_dir, f"{basename}.wav")

    # 已转换过则跳过
    if os.path.exists(wav_path):
        return wav_path

    if check_ffmpeg():
        _convert_with_ffmpeg(audio_path, wav_path)
    else:
        _convert_with_torchaudio(audio_path, wav_path)

    return wav_path


def _convert_with_ffmpeg(input_path: str, output_path: str):
    """ffmpeg 转换: 16kHz / mono / 16-bit PCM WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",          # 采样率 16kHz
        "-ac", "1",              # 单声道
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-f", "wav",             # WAV 容器
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"ffmpeg 转换完成: {Path(input_path).name} → {Path(output_path).name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 转换失败: {input_path}\n"
            f"stderr: {e.stderr}\n"
            f"请确认 ffmpeg 已安装: apt install ffmpeg 或 brew install ffmpeg"
        )


def _convert_with_torchaudio(input_path: str, output_path: str):
    """torchaudio 转换: 16kHz / mono / 16-bit WAV."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(input_path)

    # 重采样到 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # 转单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    torchaudio.save(output_path, waveform, 16000, bits_per_sample=16)
    logger.info(f"torchaudio 转换完成: {Path(input_path).name} → {Path(output_path).name}")


def get_audio_duration(wav_path: str) -> float:
    """获取音频时长 (秒), 用于检测是否超过 AED 60s 限制."""
    import soundfile as sf
    try:
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return 0.0


# ─── 模型加载 ───────────────────────────────────────────────

def get_model(model_dir: str = None, use_gpu: bool = True, use_half: bool = False):
    """
    加载 FireRedASR2-AED 模型 (单例模式).

    Args:
        model_dir: 模型目录路径 (默认: pretrained_models/FireRedASR2-AED)
        use_gpu:   是否使用 GPU
        use_half:  是否使用 FP16 推理 (RTX 4090 推荐关闭, 精度更稳)

    Returns:
        FireRedAsr2 实例
    """
    global _model_cache

    # 如果已加载且配置一致, 直接返回缓存
    if _model_cache is not None:
        return _model_cache

    # 延迟导入 — 确保路径已设置
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    if model_dir is None:
        model_dir = str(DEFAULT_MODEL_DIR)
    else:
        model_dir = str(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"模型目录不存在: {model_dir}\n"
            f"请参考 FireRedASR2S/README.md 下载模型:\n"
            f"  modelscope download --model xukaituo/FireRedASR2-AED "
            f"--local_dir {model_dir}"
        )

    logger.info(f"加载 FireRedASR2-AED 模型: {model_dir}")
    logger.info(f"  GPU: {use_gpu}, FP16: {use_half}")

    asr_config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=use_half,
        beam_size=3,
        nbest=1,
        decode_max_len=0,         # 0 = 不限
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=True,    # 返回时间戳
    )

    model = FireRedAsr2.from_pretrained("aed", model_dir, asr_config)

    _model_cache = model
    logger.info("模型加载完成 ✓")
    return model


# ─── 推理接口 ───────────────────────────────────────────────

def transcribe_file(model, audio_path: str) -> dict | None:
    """
    识别单个音频文件.

    自动处理格式转换. 若音频 > 60s 会打印警告 (AED 模型限制).

    Args:
        model:      FireRedAsr2 模型实例
        audio_path: 音频文件路径 (.mp3 / .wav / ...)

    Returns:
        dict with keys: uttid, text, confidence, dur_s, rtf, timestamp
        失败时返回 None
    """
    wav_path = convert_to_wav(audio_path)
    uttid = Path(audio_path).stem

    # 检查时长
    dur = get_audio_duration(wav_path)
    if dur > 200.0:
        logger.error(
            f"✗ 音频时长 {dur:.1f}s 超过 200s, AED 位置编码会报错, 跳过."
        )
        return None
    elif dur > 60.0:
        logger.warning(
            f"⚠ 音频时长 {dur:.1f}s 超过 60s, AED 模型可能出现幻觉."
            f"建议使用 FireRedASR2S 完整管线 (含 VAD 分段) 或截断音频."
        )

    results = model.transcribe([uttid], [wav_path])
    return results[0] if results else None

"""
批量语音识别 — 使用 FireRedASR2-AED 模型

遍历指定文件夹内所有音频文件, 逐一识别并保存 .txt 转录结果.

用法:
  python batch_process.py                          # 处理默认文件夹 audio_folder
  python batch_process.py <文件夹路径>               # 处理指定文件夹
  python batch_process.py <文件夹路径> --recursive   # 递归处理子文件夹

说明:
  - 支持 .mp3 / .wav / .ogg 格式
  - 自动转换为 16kHz WAV 后送入 AED 模型
  - 转录结果保存在 <文件夹>/transcriptions/ 下
  - AED 模型支持中文 (普通话 + 方言) 和英文, 自动识别
  - 音频超过 60s 会有警告 (AED 限制), 超过 200s 会跳过

依赖: FireRedASR2S 项目 (已在本目录下), ffmpeg (音频转换)
"""

import os
import sys
import logging
import json
from pathlib import Path

# —— 内部模块 ——
from asr_utils import get_model, transcribe_file, convert_to_wav, get_audio_duration
from industrial_postprocess import postprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_process")

# 支持的音频格式
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".aac", ".wma"}


def find_audio_files(folder_path: str, recursive: bool = False) -> list[str]:
    """
    查找文件夹中所有音频文件.

    Args:
        folder_path: 文件夹路径
        recursive:   是否递归子文件夹

    Returns:
        按文件名排序的音频文件路径列表
    """
    audio_files = []
    folder = Path(folder_path)

    if recursive:
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(str(p) for p in folder.rglob(f"*{ext}"))
    else:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(str(entry))

    audio_files.sort()
    return audio_files


def batch_process(
    folder_path: str,
    model_dir: str = None,
    use_gpu: bool = True,
    use_half: bool = False,
    recursive: bool = False,
    postprocess: bool = False,
    terms_config: str = None,
    enable_fuzzy: bool = False,
):
    """
    批量处理文件夹中的所有音频文件, 保存转录结果.

    Args:
        folder_path: 音频文件夹路径
        model_dir:   AED 模型目录 (默认自动查找)
        use_gpu:     使用 GPU 推理
        use_half:    FP16 推理
        recursive:   递归处理子文件夹
    """
    # 1. 检查文件夹
    if not os.path.isdir(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")
        sys.exit(1)

    # 2. 加载模型 (只加载一次)
    try:
        model = get_model(model_dir=model_dir, use_gpu=use_gpu, use_half=use_half)
        logger.info("模型加载成功 ✓")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        sys.exit(1)

    # 3. 查找音频文件
    audio_files = find_audio_files(folder_path, recursive=recursive)

    if not audio_files:
        logger.warning(f"文件夹中没有支持的音频文件: {folder_path}")
        logger.info(f"支持的格式: {', '.join(AUDIO_EXTENSIONS)}")
        return

    logger.info(f"找到 {len(audio_files)} 个音频文件")

    # 4. 创建输出目录
    output_folder = os.path.join(folder_path, "transcriptions")
    os.makedirs(output_folder, exist_ok=True)

    # 5. 逐个处理
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, audio_file in enumerate(audio_files, 1):
        try:
            file_name = Path(audio_file).name
            print(f"\n[{idx}/{len(audio_files)}] {file_name}")
            print("-" * 50)

            # 预检查音频时长 (跳过过长文件)
            try:
                wav_path = convert_to_wav(audio_file)
                dur = get_audio_duration(wav_path)
                if dur > 200.0:
                    logger.warning(f"  ⚠ 音频 {dur:.1f}s 超过 200s 限制, 跳过")
                    skip_count += 1
                    continue
            except Exception:
                pass  # 转换阶段再处理错误

            # 识别
            result = transcribe_file(model, audio_file)

            if result is None:
                logger.error(f"  识别失败")
                fail_count += 1
                continue

            text = result.get("text", "")

            if not text.strip():
                logger.warning(f"  识别结果为空")
                fail_count += 1
                continue

            postprocess_result = None
            if postprocess:
                postprocess_result = postprocess_text(
                    text,
                    config_path=terms_config,
                    enable_fuzzy=enable_fuzzy,
                )
                text = postprocess_result["final_text"]

            # 显示结果
            # 截断显示 (避免过长)
            display_text = text[:120] + "..." if len(text) > 120 else text
            print(f"  {display_text}")
            print(f"  置信度: {result.get('confidence', 'N/A')}  "
                  f"时长: {result.get('dur_s', 'N/A')}s  "
                  f"RTF: {result.get('rtf', 'N/A')}")

            # 保存 txt
            output_file = os.path.join(output_folder, f"{Path(audio_file).stem}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  已保存: {output_file}")

            if postprocess_result is not None:
                log_file = os.path.join(output_folder, f"{Path(audio_file).stem}.correction_log.json")
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(postprocess_result["correction_log"], f, ensure_ascii=False, indent=2)
                print(f"  后处理日志: {log_file}")

            success_count += 1

        except Exception as e:
            logger.error(f"  处理出错: {e}")
            fail_count += 1

    # 6. 汇总
    print("\n" + "=" * 60)
    print(f"处理完成!  成功: {success_count}  跳过: {skip_count}  失败: {fail_count}")
    print(f"转录结果保存在: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FireRedASR2-AED 批量语音识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_process.py                          # 处理 audio_folder
  python batch_process.py ./人声样本/人声样本1       # 处理指定文件夹
  python batch_process.py ./audio_folder --cpu     # CPU 推理
  python batch_process.py ./audio_folder --half    # FP16 推理 (省显存)
  python batch_process.py ./audio_folder -r        # 递归处理子文件夹
        """,
    )
    parser.add_argument(
        "folder", nargs="?", default="test_audio_folder",
        help="音频文件夹路径 (默认: test_audio_folder)"
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="AED 模型目录 (默认: FireRedASR2S/pretrained_models/FireRedASR2-AED)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="使用 CPU 推理 (默认使用 GPU)"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="启用 FP16 推理 (节省显存)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归处理子文件夹中的所有音频文件"
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="启用工业术语后处理"
    )
    parser.add_argument(
        "--terms-config", default=None,
        help="工业术语 YAML 配置路径 (默认: configs/industrial_terms.yaml)"
    )
    parser.add_argument(
        "--enable-fuzzy", action="store_true",
        help="启用保守模糊术语纠错 (默认关闭)"
    )
    args = parser.parse_args()

    batch_process(
        folder_path=args.folder,
        model_dir=args.model_dir,
        use_gpu=not args.cpu,
        use_half=args.half,
        recursive=args.recursive,
        postprocess=args.postprocess,
        terms_config=args.terms_config,
        enable_fuzzy=args.enable_fuzzy,
    )

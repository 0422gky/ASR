"""
单文件语音识别 — 使用 FireRedASR2-AED 模型

用法:
  python test.py                     # 识别默认音频文件 example1.mp3
  python test.py <音频文件路径>        # 识别指定文件
  python test.py --model-dir <路径>  # 指定模型目录

依赖: FireRedASR2S 项目 (已在本目录下)
"""

import os
import sys
import logging
import json
from pathlib import Path

# —— 内部模块 ——
from asr_utils import get_model, transcribe_file
from industrial_postprocess import postprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test")


def main(audio_file: str = "example1.mp3", model_dir: str = None,
         use_gpu: bool = True, use_half: bool = False,
         postprocess: bool = False, terms_config: str = None,
         enable_fuzzy: bool = False):
    # 1. 检查音频文件
    if not os.path.exists(audio_file):
        logger.error(f"音频文件不存在: {audio_file}")
        sys.exit(1)

    logger.info(f"音频文件: {audio_file}")

    # 2. 加载 AED 模型
    logger.info("正在加载 FireRedASR2-AED 模型..."
                f" (GPU={'on' if use_gpu else 'off'}, FP16={'on' if use_half else 'off'})")
    model = get_model(
        model_dir=model_dir,
        use_gpu=use_gpu,
        use_half=use_half,
    )

    # 3. 识别
    logger.info("开始识别...")
    result = transcribe_file(model, audio_file)

    if result is None:
        logger.error("识别失败")
        sys.exit(1)

    text = result["text"]
    postprocess_result = None
    if postprocess:
        postprocess_result = postprocess_text(
            text,
            config_path=terms_config,
            enable_fuzzy=enable_fuzzy,
        )
        text = postprocess_result["final_text"]

        log_path = Path(audio_file).with_suffix(".correction_log.json")
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(postprocess_result["correction_log"], f, ensure_ascii=False, indent=2)
        logger.info(f"后处理日志已保存: {log_path}")

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("识别结果:" if not postprocess else "识别结果(已后处理):")
    print("=" * 60)
    print(text)
    if postprocess_result is not None and postprocess_result["norm_text"] != postprocess_result["final_text"]:
        print("-" * 60)
        print("规范化中间结果:")
        print(postprocess_result["norm_text"])
    print("=" * 60)
    print(f"置信度: {result.get('confidence', 'N/A')}")
    print(f"音频时长: {result.get('dur_s', 'N/A')}s")
    print(f"实时率 (RTF): {result.get('rtf', 'N/A')}")
    print("=" * 60)

    # 可选: 打印词级时间戳
    if "timestamp" in result and result["timestamp"]:
        print("\n词级时间戳:")
        for token, start, end in result["timestamp"]:
            print(f"  [{start:7.3f}s - {end:7.3f}s] {token}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FireRedASR2-AED 单文件语音识别"
    )
    parser.add_argument(
        "audio_file", nargs="?", default="example1.mp3",
        help="音频文件路径 (默认: example1.mp3)"
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
        help="启用 FP16 推理 (节省显存, 精度略降)"
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

    main(
        audio_file=args.audio_file,
        model_dir=args.model_dir,
        use_gpu=not args.cpu,
        use_half=args.half,
        postprocess=args.postprocess,
        terms_config=args.terms_config,
        enable_fuzzy=args.enable_fuzzy,
    )

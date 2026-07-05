# 文件结构

```text
program_and_data/
├── asr_utils.py                    ← 模型加载、音频转换、单文件识别工具
├── test.py                         ← 单文件 ASR 转录
├── batch_process.py                ← 批量 ASR 转录
├── industrial_normalizer.py        ← 工业 ASR 文本规范化
├── term_corrector.py               ← 工业术语纠错
├── industrial_postprocess.py       ← 后处理统一入口
├── configs/
│   └── industrial_terms.yaml       ← 工业术语词表
├── tools/
│   ├── build_industrial_eval_csv.py ← 从转录结果整理评估输入 CSV
│   ├── build_finetune_manifest.py  ← 构建微调训练/验证 manifest
│   └── evaluate_industrial_asr.py  ← 后处理效果评估脚本
├── tests/
│   ├── test_industrial_postprocess.py
│   └── test_build_finetune_manifest.py
└── FireRedASR2S/                   ← ASR 系统 + 预训练模型
├─data                              ← AISHELL 和 MagicHuB 数据，之后用来微调AED模型
│  ├─AISHELL
│  │  ├─train
│  │  │  ├─S0002
│  │  │  └─S0003
│  │  ├─transcript
│  │  └─wav
│  └─MagicHub
│      └─development
│          ├─TXT
│          └─WAV
```

# QuickStart

## 1. 安装依赖

按照 `FireRedASR2S/README.md` 安装依赖：

```bash
pip install -r FireRedASR2S/requirements.txt
```

## 2. 安装 ffmpeg

`ffmpeg` 用于音频格式转换：

```bash
sudo apt install ffmpeg
```

## 3. 准备 AED 模型

参考 https://www.modelscope.cn/models/xukaituo/FireRedASR2-AED/ 获取 ASR 模型权重。

仓库里没有权重，需要自行下载，或直接 clone FireRedASR2-AED 代码。默认模型路径：

```text
FireRedASR2S/pretrained_models/FireRedASR2-AED/
```

## 4. 运行 ASR

```bash
python test.py example1.mp3
python batch_process.py audio_folder
python batch_process.py 人声样本/人声样本1 -r
```

# 工业场景后处理

新增的工业后处理模块用于修正常见工业术语、型号大小写、数字单位和工位写法。默认不改变原始 ASR 流程，只有传入 `--postprocess` 时才启用。

## 模块说明

- `industrial_normalizer.py`
  - 中文数字转阿拉伯数字，例如 `二十秒 -> 20秒`、`四十度 -> 40度`
  - 单位空格规范化，例如 `20 秒 -> 20秒`
  - 工位大小写规范化，例如 `a工位 -> A工位`
  - 英文型号大小写规范化，例如 `acopos d1 -> ACOPOSD1`
  - 保留错误码，例如 `-1067186135` 不拆分、不改写

- `term_corrector.py`
  - 读取 `configs/industrial_terms.yaml`
  - 将 alias 替换为 canonical，例如 `par ID -> ParID`
  - 支持可选模糊匹配，默认关闭，避免过度纠错

- `industrial_postprocess.py`
  - 提供统一入口 `postprocess_text(text, config_path=None, enable_fuzzy=False)`
  - 返回 `raw_text`、`norm_text`、`final_text`、`fixed_terms`、`correction_log`

- `configs/industrial_terms.yaml`
  - 内置工业词表，包含 `AcOPOStrak`、`AcOPOSmulti`、`AcOPOS P3`、`ACOPOSD1`、`ParID`、`mcTCs坐标系`、`A工位`、`B工位`、`C工位` 等术语

## 后处理运行示例

单文件识别并启用后处理：

```bash
python test.py example1.mp3 --postprocess
```

批量识别并启用后处理：

```bash
python batch_process.py test_audio_folder --postprocess
```

递归批量识别，并启用保守模糊纠错：

```bash
python batch_process.py train_audio_folder -r --postprocess --enable-fuzzy --terms-config configs/industrial_terms.yaml
```

使用自定义术语词表：

```bash
python test.py example1.mp3 --postprocess --terms-config configs/industrial_terms.yaml
```

启用后处理时，会额外输出 correction log，方便记录哪些规则或术语被修正。

# 后处理效果评估

批处理会生成 `.txt` 和 `.correction_log.json`，评估前需要先整理成 `evaluate_industrial_asr.py` 所需的 CSV。

## 1. 从转录结果生成评估输入 CSV

```bash
python tools/build_industrial_eval_csv.py \
  --transcriptions-dir train_audio_folder/transcriptions \
  --reference train_audio_folder/测试语音文本原稿.txt \
  --audio-root train_audio_folder \
  --output train_audio_folder/eval_input.csv
```

生成的 CSV 字段为：

```text
id,audio_path,ref_text,asr_text
```

默认会根据 `.correction_log.json` 反推后处理前的 `asr_text`，这样可以继续用评估脚本比较后处理前后的 CER 和术语准确率。

如果只想把当前 `.txt` 内容直接作为 `asr_text`，可以加：

```bash
python tools/build_industrial_eval_csv.py \
  --transcriptions-dir train_audio_folder/transcriptions \
  --reference train_audio_folder/测试语音文本原稿.txt \
  --audio-root train_audio_folder \
  --output train_audio_folder/eval_input.csv \
  --use-final-text
```

## 2. 运行评估脚本

```bash
python tools/evaluate_industrial_asr.py \
  --input train_audio_folder/eval_input.csv \
  --output train_audio_folder/eval_result.csv
```

评估输入 CSV 必须包含字段：

```text
id,audio_path,ref_text,asr_text
```

输出 CSV 会包含：

- `norm_text`: 文本规范化后的结果
- `final_text`: 术语纠错后的最终结果
- `cer_before`: 后处理前 CER
- `cer_after`: 后处理后 CER
- `term_acc_before`: 后处理前术语准确率
- `term_acc_after`: 后处理后术语准确率
- `fixed_terms`: 被修正的术语记录
- `error_type`: 改善类型，例如 `improved`、`unchanged`、`regressed`

启用模糊纠错评估：

```bash
python tools/evaluate_industrial_asr.py --input input.csv --output output.csv --enable-fuzzy
```

# 单元测试

运行后处理测试：

```bash
python -m pytest tests/test_industrial_postprocess.py
```

当前覆盖示例：

- `二十秒 -> 20秒`
- `四十度 -> 40度`
- `par ID -> ParID`
- `MC TCS坐标系 -> mcTCs坐标系`
- `AC O P O S trak -> AcOPOStrak`
- `a工位 -> A工位`

# 数据集构建
工业 60 条 × repeat 10
AISHELL-1 小子集 1000 条
MagicHub 普通话 500 条

## 微调 manifest 构建

`FireRedASR2S` 当前仓库主要提供推理代码，未提供明确训练 manifest 格式。因此这里先生成通用 `jsonl`，同时输出简单 `tsv`，方便后续接入 FireRedASR2-AED 微调脚本。

默认混合：

- 工业数据：来自 `train_audio_folder`，按 `测试语音文本原稿.txt` 的编号映射到音频文件名末尾编号，例如 `YQL-1-5` 对应第 5 条标注。
- AISHELL：来自 `data/AISHELL`，只保留本地已有 wav 且有 transcript 的样本。
- MagicHub：来自 `data/MagicHub/development`，按 wav/txt 同名匹配，并从时间戳 TXT 中拆出语音片段。

生成清单：

```bash
python tools/build_finetune_manifest.py \
  --aishell-root data/AISHELL \
  --magichub-root data/MagicHub/development \
  --industrial-root train_audio_folder \
  --out-dir manifests \
  --industrial-repeat 10 \
  --magichub-repeat 1 \
  --aishell-repeat 1 \
  --dev-ratio 0.1 \
  --seed 42
```

输出文件：

```text
manifests/train_mix.jsonl
manifests/dev_mix.jsonl
manifests/train_mix.tsv
manifests/dev_mix.tsv
manifests/data_stats.json
manifests/correction_log.json
```

`jsonl` 每行示例：

```json
{"utt_id":"industrial_YQL-1-5_rep03","audio":"train_audio_folder/人声样本1/YQL-1-5.mp3","text":"帮我移动小车从A工位到B工位，在B工位停顿20秒后移动到C工位。","source":"industrial","speaker":"YQL","env":"env1","duration":5.31}
```

只查看统计、不写文件：

```bash
python tools/build_finetune_manifest.py \
  --aishell-root data/AISHELL \
  --magichub-root data/MagicHub/development \
  --industrial-root train_audio_folder \
  --out-dir manifests \
  --dry-run
```

重要规则：

- 只在 manifest 中重复样本，不复制 wav/mp3。
- dev 集不重复采样。
- train/dev 按 audio path 分组切分，避免同一个音频泄漏到两个集合。
- 工业标签会走 `industrial_postprocess.py`，例如 `2o秒`、`c工位` 这类疑似错误会修正并写入 `correction_log.json`。

## FireRedASR2-AED 微调

当前 `FireRedASR2S` 仓库没有官方训练入口，因此新增了一个轻量 AED 微调脚本：

```text
tools/finetune_firered_aed.py
configs/finetune_industrial_aed.yaml
tools/evaluate_firered_aed_manifest.py
tools/convert_manifest_for_firered_aed.py
```

先检查 manifest：

```bash
python tools/convert_manifest_for_firered_aed.py --input manifests/train_mix.jsonl --dry-run
python tools/convert_manifest_for_firered_aed.py --input manifests/dev_mix.jsonl --dry-run
```

先跑 1-5 step smoke test，确认 dataloader、loss、checkpoint 正常：

```bash
python tools/finetune_firered_aed.py \
  --config configs/finetune_industrial_aed.yaml \
  --smoke-test
```

默认配置会冻结 encoder，只训练 decoder 和 CTC 头，避免小规模工业数据过拟合。当前没有强行加入 LoRA，因为仓库里的 AED 是自定义 PyTorch 模块，没有现成 LoRA 训练路径。

正式微调命令：

```bash
python tools/finetune_firered_aed.py --config configs/finetune_industrial_aed.yaml
```

默认输出目录：

```text
outputs/finetune_industrial_aed/
```

评估 baseline：

```bash
python tools/evaluate_firered_aed_manifest.py \
  --model-dir FireRedASR2S/pretrained_models/FireRedASR2-AED \
  --manifest manifests/dev_mix.jsonl \
  --output-csv outputs/baseline_dev_eval.csv \
  --postprocess
```

评估微调后模型：

```bash
python tools/evaluate_firered_aed_manifest.py \
  --model-dir outputs/finetune_industrial_aed \
  --manifest manifests/dev_mix.jsonl \
  --output-csv outputs/finetune_dev_eval.csv \
  --postprocess
```

评估指标包含 CER、工业术语准确率、数字准确率、错误码准确率、工位准确率。



## AISHELL-1
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd. 400 people from different accent areas in China are invited to participate in the recording, which is conducted in a quiet indoor environment using high fidelity microphone and downsampled to 16kHz. The manual transcription accuracy is above 95%, through professional speech annotation and strict quality inspection. The data is free for academic use. We hope to provide moderate amount of data for new researchers in the field of speech recognition.

You can cite the data using the following BibTeX entry:

@inproceedings{aishell_2017,
title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
booktitle={Oriental COCOSDA 2017},
pages={Submitted},
year={2017}
}



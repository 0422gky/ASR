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
│   └── evaluate_industrial_asr.py  ← 后处理效果评估脚本
├── tests/
│   └── test_industrial_postprocess.py
└── FireRedASR2S/                   ← ASR 系统 + 预训练模型
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
python batch_process.py train_audio_folder -r --postprocess --enable-fuzzy
```

使用自定义术语词表：

```bash
python test.py example1.mp3 --postprocess --terms-config configs/industrial_terms.yaml
```

启用后处理时，会额外输出 correction log，方便记录哪些规则或术语被修正。

# 后处理效果评估

评估脚本：

```bash
python tools/evaluate_industrial_asr.py --input input.csv --output output.csv
```

输入 CSV 必须包含字段：

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

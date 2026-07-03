# 文件结构
```text
program_and_data/
├── asr_utils.py    ← 新增  共享工具模块
├── test.py         ← 已修改  单文件 ASR 转录
├── pichuli.py      ← 已修改  批量 ASR 转录
└── FireRedASR2S/   ← 未修改  ASR 系统 + 预训练模型
```

# QuickStart
## 1. 按照 FireRedASR2S/README.md 安装依赖
pip install -r FireRedASR2S/requirements.txt

## 2. 安装 ffmpeg（用于音频转换）
sudo apt install ffmpeg

## 3. AED模型
FireRedASR2S/pretrained_models/FireRedASR2-AED/

## 4. 运行
python test.py example1.mp3
python pichuli.py audio_folder
python pichuli.py 人声样本/人声样本1 -r

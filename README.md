# 文件结构
```text
program_and_data/
├── asr_utils.py    
├── test.py         ← 单文件 ASR 转录
├── batch_process.py      ← 批量 ASR 转录
└── FireRedASR2S/   ← ASR 系统 + 预训练模型
```

# QuickStart
## 1. 按照 FireRedASR2S/README.md 安装依赖
pip install -r FireRedASR2S/requirements.txt

## 2. 安装 ffmpeg（用于音频转换）
sudo apt install ffmpeg

## 3. AED模型
参考 https://www.modelscope.cn/models/xukaituo/FireRedASR2-AED/ 获取ASR模型权重
仓库里没有权重，要自己下载，或者直接clone FireRedASR2-AED代码也行
FireRedASR2S/pretrained_models/FireRedASR2-AED/

## 4. 运行
python test.py example1.mp3
python batch_process.py audio_folder
python batch_process.py 人声样本/人声样本1 -r

# Robot Voice Command

本项目提供三种运行方式:

```bash
# 1. 不用麦克风，直接测试指令解析
python3 asr.py --text "小白抓取杯子" --no-ollama

# 2. 识别音频文件并解析
python3 asr.py --audio temp.wav --whisper-model small

# 3. 默认进入麦克风唤醒模式
python3 asr.py
```

`mic.py` 现在是兼容入口，等价于运行 `asr.py`:

```bash
python3 mic.py --text "停止" --no-ollama
```

查看并指定麦克风:

```bash
python3 asr.py --list-mics
python3 asr.py --mic-index 5
```

默认会隐藏 ALSA/PyAudio 的底层设备枚举日志。需要排查音频底层问题时再打开:

```bash
python3 asr.py --mic-index 5 --show-alsa-logs
```

## 依赖

```bash
python3 -m pip install -r requirements.txt
```

如果当前环境出现 NumPy 2.x 与 PyTorch/Whisper 的兼容警告，执行:

```bash
python3 -m pip install "numpy<2"
```

## Ollama

默认会尝试连接或后台启动 Ollama，并使用 `qwen2.5:7b` 做 JSON 意图解析。没有 Ollama 或模型不可用时，程序会退回本地规则解析。

需要手动准备模型时运行:

```bash
ollama pull qwen2.5:7b
```

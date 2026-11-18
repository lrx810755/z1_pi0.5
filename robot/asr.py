import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ChatOllama.*deprecated.*")

try:
    import speech_recognition as sr
except ImportError as exc:
    raise SystemExit("缺少依赖: SpeechRecognition。请运行: python3 -m pip install -r requirements.txt") from exc

try:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise SystemExit("缺少依赖: langchain/pydantic。请运行: python3 -m pip install -r requirements.txt") from exc


OLLAMA_URL = "http://127.0.0.1:11434"


class CommandIntent(BaseModel):
    action: str = Field(
        description="用户动作意图，简短英文动词，例如 pick、move、stop。没有明确动作时输出 none。"
    )
    object: str = Field(description="动作目标物体，翻译为英文。没有目标时输出 none。")


@dataclass
class RuntimeConfig:
    whisper_model: str = "small"
    llm_model: str = "qwen2.5:3b"
    wake_word: str = "小白"
    temp_wav: str = "temp.wav"
    mic_index: Optional[int] = None
    no_ollama: bool = False
    show_alsa_logs: bool = False


def clean_command(text: str) -> str:
    return re.sub(r"[^\w\s\u4e00-\u9fa5]", "", text).strip()


@contextlib.contextmanager
def suppress_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def list_microphones(show_alsa_logs: bool = False) -> list[str]:
    with suppress_stderr(not show_alsa_logs):
        return sr.Microphone.list_microphone_names()


def validate_microphone(mic_index: Optional[int], show_alsa_logs: bool = False) -> bool:
    names = list_microphones(show_alsa_logs=show_alsa_logs)
    if not names:
        print("[错误] 没有发现麦克风设备。请检查系统是否识别麦克风，或先用 --audio 测试音频文件。")
        return False
    if mic_index is not None and not 0 <= mic_index < len(names):
        print(f"[错误] --mic-index {mic_index} 不存在。可用编号范围: 0-{len(names) - 1}")
        print("运行 `python3 asr.py --list-mics` 查看设备列表。")
        return False
    return True


def is_ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1)
        return True
    except urllib.error.URLError:
        return False
    except OSError:
        return False


def ensure_ollama_running(required: bool = True) -> bool:
    if is_ollama_running():
        print("[系统] Ollama 已运行")
        return True

    if shutil.which("ollama") is None:
        message = "[系统] 未找到 ollama 命令，将使用本地规则解析。"
        if required:
            print(message)
        return False

    print("[系统] Ollama 未运行，正在尝试后台启动...")
    try:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["preexec_fn"] = os.setpgrp
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except Exception as exc:
        print(f"[系统] Ollama 自动启动失败: {exc}")
        return False

    for _ in range(10):
        time.sleep(0.5)
        if is_ollama_running():
            print("[系统] Ollama 后台启动成功")
            return True

    print("[系统] Ollama 未能在 5 秒内就绪，将使用本地规则解析。")
    return False


def heuristic_parse_intent(text: str) -> dict:
    lower = text.lower()
    action_map = [
        ("stop", ["停", "停止", "别动", "stop"]),
        ("pick", ["抓", "拿", "取", "捡", "夹", "pick", "grab"]),
        ("move", ["移动", "过去", "过来", "去", "到", "move"]),
        ("open", ["打开", "open"]),
        ("close", ["关闭", "合上", "close"]),
        ("put", ["放", "放下", "put", "place"]),
    ]
    object_map = [
        ("bottle", ["瓶", "水瓶", "bottle"]),
        ("cup", ["杯", "杯子", "cup"]),
        ("box", ["盒", "盒子", "box"]),
        ("apple", ["苹果", "apple"]),
        ("ball", ["球", "ball"]),
        ("table", ["桌", "桌子", "table"]),
    ]

    action = "none"
    for name, keywords in action_map:
        if any(keyword in lower or keyword in text for keyword in keywords):
            action = name
            break

    obj = "none"
    for name, keywords in object_map:
        if any(keyword in lower or keyword in text for keyword in keywords):
            obj = name
            break

    return {"action": action, "object": obj}


class VoiceTranslator:
    def __init__(self, config: RuntimeConfig, load_asr: bool = True):
        self.config = config
        self.wake_word = config.wake_word
        self.recognizer = sr.Recognizer()
        self.asr_model = None
        self.device = "未加载"

        if load_asr:
            self._load_asr_model()

        self.ollama_ready = False
        self.chain = None
        if not config.no_ollama:
            self.ollama_ready = ensure_ollama_running(required=False)
            if self.ollama_ready:
                self.chain = self._build_llm_chain(config.llm_model)

        mode = "Ollama + LangChain" if self.chain else "本地规则"
        print(f"[系统] 初始化完成，设备: {self.device}，解析模式: {mode}，唤醒词: {self.wake_word}")

    def _load_asr_model(self):
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise SystemExit("缺少依赖: torch/openai-whisper。请运行: python3 -m pip install -r requirements.txt") from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[系统] 正在加载 Whisper 模型: {self.config.whisper_model}")
        self.asr_model = whisper.load_model(self.config.whisper_model).to(self.device)

    def _build_llm_chain(self, llm_model: str):
        parser = JsonOutputParser(pydantic_object=CommandIntent)
        prompt = PromptTemplate(
            template=(
                "你是一个机械臂意图解析器。请只输出 JSON，提取用户的动作和目标。\n\n"
                "{format_instructions}\n\n"
                "规则:\n"
                "- 用户想停止时 action 必须是 stop。\n"
                "- 没有明确动作时 action 和 object 都输出 none。\n"
                "- action 使用简短英文动词，object 使用简短英文名词。\n\n"
                '用户原话: "{text}"'
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | ChatOllama(model=llm_model, temperature=0, format="json") | parser

    def listen(self, phrase_time_limit: int = 5):
        try:
            with suppress_stderr(not self.config.show_alsa_logs):
                microphone = sr.Microphone(device_index=self.config.mic_index)
                with microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    self.recognizer.energy_threshold += 150
                    return self.recognizer.listen(source, timeout=None, phrase_time_limit=phrase_time_limit)
        except Exception as exc:
            hint = "可先运行: python3 asr.py --list-mics"
            if self.config.mic_index is None:
                hint += "，再用 --mic-index 指定设备"
            print(f"[错误] 麦克风采集失败: {exc}。{hint}")
            return None

    def recognize_speech(self, audio_or_path) -> str:
        if self.asr_model is None:
            print("[系统] 首次使用语音识别，正在加载 Whisper 模型...")
            self._load_asr_model()

        if isinstance(audio_or_path, str):
            wav_path = audio_or_path
        else:
            wav_path = self.config.temp_wav
            with open(wav_path, "wb") as f:
                f.write(audio_or_path.get_wav_data())

        vocab_prompt = f"{self.wake_word}，抓取，移动，停止，过来，过去，杯子，瓶子，盒子。"
        result = self.asr_model.transcribe(wav_path, language="zh", initial_prompt=vocab_prompt)
        text = result["text"].strip()

        hallucinations = ["字幕", "感谢观看", "请不吝赐教", "未经允许", "的机器人指令"]
        if any(item in text for item in hallucinations):
            return ""
        return text

    def parse_intent(self, text: str) -> dict:
        if self.chain:
            try:
                return self.chain.invoke({"text": text})
            except Exception as exc:
                print(f"[提醒] Ollama 解析失败，改用本地规则: {exc}")
        return heuristic_parse_intent(text)

    def process_command(self, text: str) -> dict:
        text = clean_command(text)
        if not text:
            print("[提醒] 没有可解析的文本")
            return {"action": "none", "object": "none"}

        print(f"[输入] {text}")
        intent = self.parse_intent(text)
        if intent and intent.get("action") != "none":
            print("[解析结果]")
            print(json.dumps(intent, indent=2, ensure_ascii=False))
        else:
            print("[提醒] 指令不明确")
        return intent

    def run(self):
        is_awake = False
        while True:
            if not is_awake:
                print(f"\n[待机] 等待唤醒词 '{self.wake_word}'...")
                audio = self.listen(phrase_time_limit=4)
                if not audio:
                    continue

                text = self.recognize_speech(audio)
                if not text:
                    continue

                if self.wake_word in text:
                    print(f"[唤醒] 音频内容: {text}")
                    is_awake = True
                    command = clean_command(text.replace(self.wake_word, ""))
                    if len(command) >= 2:
                        self.process_command(command)
                        is_awake = False
            else:
                print("[聆听] 请下达具体指令...")
                audio = self.listen(phrase_time_limit=6)
                if not audio:
                    print("[超时] 未听到指令，重新进入待机")
                    is_awake = False
                    continue

                text = self.recognize_speech(audio)
                command = clean_command(text)
                if len(command) >= 2:
                    self.process_command(command)
                is_awake = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="语音唤醒 + Whisper 识别 + 机械臂指令解析")
    parser.add_argument("--text", help="直接解析一段文字，用于无麦克风快速测试")
    parser.add_argument("--audio", help="识别并解析一个 wav/mp3/m4a 等音频文件")
    parser.add_argument("--whisper-model", default="small", help="Whisper 模型名，默认 small；快速测试可用 tiny/base")
    parser.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama 模型名，默认 qwen2.5:7b")
    parser.add_argument("--wake-word", default="小白", help="唤醒词，默认 小白")
    parser.add_argument("--mic-index", type=int, help="指定麦克风设备编号")
    parser.add_argument("--list-mics", action="store_true", help="列出可用麦克风设备后退出")
    parser.add_argument("--show-alsa-logs", action="store_true", help="显示 ALSA/PyAudio 底层日志，默认隐藏")
    parser.add_argument("--no-ollama", action="store_true", help="不连接 Ollama，只使用本地规则解析")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_mics:
        names = list_microphones(show_alsa_logs=args.show_alsa_logs)
        if not names:
            print("[错误] 没有发现麦克风设备")
            return 1
        print("[麦克风列表]")
        for index, name in enumerate(names):
            print(f"{index}: {name}")
        return 0

    config = RuntimeConfig(
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        wake_word=args.wake_word,
        mic_index=args.mic_index,
        no_ollama=args.no_ollama,
        show_alsa_logs=args.show_alsa_logs,
    )

    if args.text is None and args.audio is None:
        if not validate_microphone(args.mic_index, args.show_alsa_logs):
            return 1

    load_asr = args.audio is not None or args.text is None
    app = VoiceTranslator(config, load_asr=load_asr)

    if args.text:
        app.process_command(args.text)
        return 0

    if args.audio:
        text = app.recognize_speech(args.audio)
        print(f"[识别文本] {text}")
        app.process_command(text)
        return 0

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

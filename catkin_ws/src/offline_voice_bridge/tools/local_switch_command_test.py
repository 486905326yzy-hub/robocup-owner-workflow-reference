#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""电气开关语音指令测试。

人用任意自然说法表达"开启/关闭电气开关"的意图（说法格式不固定），
链路为：

    麦克风录音 (arecord)
      -> faster-whisper 中文识别
      -> 本地 qwen3.5:2b 判断开关意图，只输出极短 JSON
      -> Python 确定性生成播报文本
      -> Piper 中文合成 wav（默认，快；可选 MeloTTS）
      -> aplay 播放

低延迟设计（参考 local_qwen_profile_test.py 和 local_melo_echo_zh.py）：

    - think: false + /no_think：彻底关掉千问思考模式，2b 不推理只输出
    - format: json + num_predict 20：强制短 JSON，杜绝长篇生成
    - temperature: 0：确定性输出，兼顾准确
    - keep_alive 10m：模型常驻显存，避免每次重新加载
    - 启动时预热 LLM，把冷启动耗时挪到开始测试之前
    - TTS 默认 Piper：轻量 ONNX 推理，单次合成约 0.2~0.5 秒，无需预热
    - ASR beam_size=1、关闭 VAD，固定短录音下的最快配置
    - 播报句子由 Python 固定生成，不让大模型写长回复
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import local_melo_tts_zh
import local_voice_test


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WHISPER_ZH_MODEL = os.path.join(ROOT_DIR, "models", "whisper", "faster-whisper-small")
DEFAULT_PIPER = os.path.join(ROOT_DIR, "tools", "piper", "piper")
DEFAULT_PIPER_MODEL = os.path.join(ROOT_DIR, "models", "piper", "zh_CN-huayan-medium.onnx")
DEFAULT_PIPER_CONFIG = os.path.join(ROOT_DIR, "models", "piper", "zh_CN-huayan-medium.onnx.json")
# 默认走 PulseAudio 的 default 设备；桌面有 PulseAudio 占用 plughw 时也能用。
# 机器人/无桌面环境可用 --mic-device/--speaker-device 改成 plughw:CARD=Generic_1,DEV=0
DEFAULT_MIC_DEVICE = "default"
DEFAULT_SPEAKER_DEVICE = "default"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LLM_MODEL = "qwen3.5:2b"
DEFAULT_RECORD_WAV = "/dev/shm/local_switch_record_zh.wav"
DEFAULT_REPLY_WAV = "/dev/shm/local_switch_reply_zh.wav"

REPLY_ON = "好的，已开启电气开关。"
REPLY_OFF = "好的，已关闭电气开关。"
REPLY_UNKNOWN = "抱歉，我没有听出开关指令。"

# LLM 输出归一化：把各种可能的回答都映射到 on/off/unknown
ACTION_MAP = {
    "on": "on", "开": "on", "开启": "on", "打开": "on", "open": "on",
    "1": "on", "true": "on", "接通": "on", "上电": "on",
    "off": "off", "关": "off", "关闭": "off", "关掉": "off", "close": "off",
    "0": "off", "false": "off", "断开": "off", "断电": "off",
}

LLM_PROMPT_TEMPLATE = (
    "/no_think\n"
    "判断说话内容是否要开启或关闭电气开关。只输出一行 JSON，不要任何解释。\n"
    'Schema: {"action":"on"} 表示开启；{"action":"off"} 表示关闭；'
    '{"action":"unknown"} 表示与开关无关或无法判断。\n'
    "示例：\n"
    '"把灯打开" -> {"action":"on"}\n'
    '"麻烦帮我把开关打开一下" -> {"action":"on"}\n'
    '"关掉电源" -> {"action":"off"}\n'
    '"不用了，请把插座关了吧" -> {"action":"off"}\n'
    '"今天天气怎么样" -> {"action":"unknown"}\n'
    "说话内容：%s"
)


def build_parser():
    parser = argparse.ArgumentParser(description="电气开关语音指令测试：语音输入 -> 2b 模型判断开关 -> 语音播报。")
    parser.add_argument("--input-text", help="跳过麦克风和 ASR，直接判断这段文本")
    parser.add_argument("--input-wav", help="跳过麦克风录音，识别已有 wav")
    parser.add_argument("--record-wav", default=DEFAULT_RECORD_WAV, help="麦克风录音保存路径")
    parser.add_argument("--output", default=DEFAULT_REPLY_WAV, help="播报 wav 输出路径")
    parser.add_argument("--loop", action="store_true", help="连续测试；ASR/LLM 只加载一次，Ctrl+C 退出")
    parser.add_argument("--count", type=int, default=1, help="测试轮数；--loop 时忽略，0 表示一直循环")
    parser.add_argument("--keep-wav", action="store_true", help="打印并保留录音 wav 路径")
    parser.add_argument("--no-play", action="store_true", help="只生成 wav，不播放")

    parser.add_argument("--mic-device", default=DEFAULT_MIC_DEVICE)
    parser.add_argument("--speaker-device", default=DEFAULT_SPEAKER_DEVICE)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--energy-threshold", type=int, default=300)

    parser.add_argument("--model-size", default=DEFAULT_WHISPER_ZH_MODEL)
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--beam-size", type=int, default=1, help="Whisper beam size；1 最快")
    parser.add_argument("--no-vad", action="store_true", help="关闭 faster-whisper VAD，固定短录音时更快")
    parser.add_argument("--vad-filter", dest="no_vad", action="store_false", help="开启 VAD；环境噪声大时可试")
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.set_defaults(no_vad=True)

    parser.add_argument("--llm-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=30.0)
    parser.add_argument("--llm-max-tokens", type=int, default=20, help="LLM 最大输出 token；JSON 短输出足够")
    parser.add_argument("--llm-keep-alive", default="10m", help="Ollama 模型保活时间")
    parser.add_argument("--no-llm-warmup", action="store_true", help="关闭 LLM 预热；首次判断可能因模型加载变慢")

    parser.add_argument("--tts-engine", choices=["piper", "melo"], default="piper",
                        help="播报 TTS 引擎；默认 piper 更快，melo 更自然")
    parser.add_argument("--piper", default=DEFAULT_PIPER)
    parser.add_argument("--piper-model", default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-config", default=DEFAULT_PIPER_CONFIG)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)

    parser.add_argument("--melo-language", default="ZH")
    parser.add_argument("--voice", default="ZH")
    parser.add_argument("--speed", type=float, default=1.0, help="MeloTTS 语速")
    parser.add_argument("--melo-device", default="cpu")
    parser.add_argument("--melo-config", default=local_melo_tts_zh.DEFAULT_CONFIG)
    parser.add_argument("--melo-ckpt", default=local_melo_tts_zh.DEFAULT_CKPT)
    parser.add_argument("--no-melo-warmup", action="store_true", help="关闭 MeloTTS 预热")
    parser.add_argument("--player", default="aplay")
    return parser


def load_asr_model(args):
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run tools/setup_offline_voice.sh first. %s" % exc)

    print("正在加载中文 ASR 模型: %s" % args.model_size)
    start = time.time()
    model = WhisperModel(args.model_size, device=args.asr_device, compute_type=args.compute_type)
    print("ASR 模型加载完成，用时 %.2f 秒" % (time.time() - start))
    return model


def load_melo_model(args):
    local_melo_tts_zh.add_melo_to_path()
    from melo.api import TTS

    if not os.path.exists(args.melo_config):
        raise RuntimeError("Melo config is missing: %s" % args.melo_config)
    if not os.path.exists(args.melo_ckpt):
        raise RuntimeError("Melo checkpoint is missing: %s" % args.melo_ckpt)

    print("正在加载 MeloTTS 模型: %s" % args.voice)
    start = time.time()
    model = TTS(language=args.melo_language, device=args.melo_device,
                config_path=args.melo_config, ckpt_path=args.melo_ckpt)
    speaker_map = dict(model.hps.data.spk2id)
    if args.voice not in speaker_map:
        raise RuntimeError("voice '%s' not found. Available: %s" % (args.voice, ", ".join(sorted(speaker_map))))
    speaker_id = speaker_map[args.voice]
    print("MeloTTS 模型加载完成，用时 %.2f 秒，speaker_id=%s" % (time.time() - start, speaker_id))
    if not args.no_melo_warmup:
        warmup_wav = "/dev/shm/local_switch_melo_warmup_zh.wav"
        print("正在预热 MeloTTS，后续播报会更快...")
        start = time.time()
        model.tts_to_file("预热。", speaker_id, warmup_wav, speed=args.speed, quiet=True)
        print("MeloTTS 预热完成，用时 %.2f 秒" % (time.time() - start))
    return model, speaker_id


def load_tts(args):
    """加载播报 TTS 引擎，返回 (engine_name, melo_ctx)。

    Piper 是独立二进制，每次合成时按需启动，无需预加载；
    MeloTTS 需要提前把 torch 模型加载进内存并预热。
    """
    if args.tts_engine == "melo":
        return "melo", load_melo_model(args)

    if not os.path.exists(args.piper) or not os.access(args.piper, os.X_OK):
        raise RuntimeError("Piper binary is not ready: %s" % args.piper)
    if not os.path.exists(args.piper_model):
        raise RuntimeError("Piper model is missing: %s" % args.piper_model)
    if not os.path.exists(args.piper_config):
        raise RuntimeError("Piper config is missing: %s" % args.piper_config)
    print("TTS engine: piper（%s，无需预加载）" % os.path.basename(args.piper_model))
    return "piper", None


def call_llm(args, transcript):
    """调用本地 2b 模型判断开关意图，返回 on/off/unknown 和原始输出。"""
    payload = {
        "model": args.llm_model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": args.llm_keep_alive,
        "messages": [{"role": "user", "content": LLM_PROMPT_TEMPLATE % transcript}],
        "options": {
            "temperature": 0,
            "top_p": 0.7,
            "num_predict": args.llm_max_tokens,
            "num_ctx": 1024,
        },
    }
    request = urllib.request.Request(
        args.llm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise RuntimeError("LLM 请求超时（%.1f 秒）。首次运行会加载模型，可重试或调大 --llm-timeout。%s"
                           % (time.time() - start, exc))
    except urllib.error.URLError as exc:
        raise RuntimeError("LLM 请求失败，请确认 Ollama 正在运行且已拉取 %s: %s" % (args.llm_model, exc))

    result = json.loads(raw)
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("LLM 返回空内容: %s" % raw)
    action = parse_action(content)
    return action, content, time.time() - start


def parse_action(content):
    """从 LLM 原始输出解析开关动作，容错各种写法。"""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return "unknown"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "unknown"

    if not isinstance(parsed, dict):
        return "unknown"
    # 优先看 action 字段；没有的话扫一遍所有值，避免模型换了字段名
    values = [parsed.get("action")]
    values.extend(str(v) for v in parsed.values() if not isinstance(v, (dict, list)))
    for value in values:
        action = ACTION_MAP.get(str(value).strip().lower())
        if action:
            return action
    return "unknown"


def keyword_fallback(transcript):
    """LLM 不可用时的兜底：按关键词判断开关意图。

    只在 LLM 调用失败时使用；包含否定词的句子标记为 unknown，
    避免把"别开灯"误判成开启。
    """
    text = transcript.strip()
    negation = re.search(r"(不要|不用|别|不想|无需|不需要|禁止|不能|别把|不用把)", text)
    if re.search(r"(关|断开|断电|停|拔掉|灭)", text) and not re.search(r"(开|接通|上电|亮)", text):
        return "off"
    if re.search(r"(开|接通|上电|启动|亮)", text) and not negation and not re.search(r"(关|断开|断电|灭)", text):
        return "on"
    return "unknown"


def classify(args, transcript):
    """判断开关意图；LLM 失败时退回关键词启发式。"""
    try:
        action, raw_content, elapsed = call_llm(args, transcript)
        print("LLM 输出: %s (LLM %.2f 秒)" % (raw_content, elapsed))
        return action, "llm"
    except Exception as exc:
        print("WARN: %s" % exc)
        action = keyword_fallback(transcript)
        print("LLM 不可用，使用关键词兜底判断。")
        return action, "keyword"


def warmup_llm(args):
    print("正在预热 LLM（%s），把模型加载进显存..." % args.llm_model)
    start = time.time()
    action, _ = classify(args, "这是预热测试，请忽略")
    print("LLM 预热完成，用时 %.2f 秒（返回: %s）" % (time.time() - start, action))


def transcribe(args, asr_model, round_index=1):
    wav_path = args.input_wav or args.record_wav
    if not args.input_wav and (args.loop or args.count != 1):
        base, ext = os.path.splitext(args.record_wav)
        wav_path = "%s_%03d%s" % (base, round_index, ext or ".wav")
    if not args.input_wav:
        print("正在从 ALSA 设备 '%s' 录音 %.1f 秒..." % (args.mic_device, args.seconds))
        start = time.time()
        rms = local_voice_test.record_wav(
            wav_path, args.seconds, args.sample_rate, args.mic_device, args.energy_threshold)
        print("录音文件: %s (RMS=%d, %.2f 秒)" % (wav_path, rms, time.time() - start))

    start = time.time()
    segments, _ = asr_model.transcribe(
        wav_path,
        language=args.language,
        vad_filter=not args.no_vad,
        beam_size=args.beam_size,
        condition_on_previous_text=False,
        no_speech_threshold=args.no_speech_threshold,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    print("识别文本: %s (ASR %.2f 秒)" % (text or "<空>", time.time() - start))
    if args.keep_wav and not args.input_wav:
        print("保留录音: %s" % wav_path)
    return text


def synthesize_piper(args, text, output_wav):
    """用 Piper 把中文文本合成为 wav，返回合成耗时。"""
    cmd = [
        args.piper,
        "--model", args.piper_model,
        "--output_file", output_wav,
        "--length_scale", str(args.length_scale),
        "--noise_scale", str(args.noise_scale),
        "--noise_w", str(args.noise_w),
    ]
    if args.piper_config and os.path.exists(args.piper_config):
        cmd.extend(["--config", args.piper_config])
    subprocess.run(cmd, input=text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def speak_result(args, action, tts_ctx):
    engine, melo_ctx = tts_ctx
    reply = {"on": REPLY_ON, "off": REPLY_OFF, "unknown": REPLY_UNKNOWN}.get(action, REPLY_UNKNOWN)
    start = time.time()
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    if engine == "piper":
        synthesize_piper(args, reply, args.output)
    else:
        melo_model, speaker_id = melo_ctx
        melo_model.tts_to_file(reply, speaker_id, args.output, speed=args.speed, quiet=True)
    print("播报文本: %s" % reply)
    print("TTS(%s) 合成用时 %.2f 秒，WAV: %s" % (engine, time.time() - start, args.output))
    if not args.no_play:
        local_melo_tts_zh.play_wav(args.output, args.player, args.speaker_device)


def iter_rounds(args):
    if args.input_wav or args.input_text:
        yield 1
        return
    if args.loop or args.count == 0:
        round_index = 1
        while True:
            yield round_index
            round_index += 1
    else:
        for round_index in range(1, max(1, args.count) + 1):
            yield round_index


def main():
    args = build_parser().parse_args()
    try:
        had_text = False
        if not args.input_text:
            asr_model = load_asr_model(args)
            tts_ctx = load_tts(args)
        else:
            asr_model = None
            tts_ctx = None
        if not args.no_llm_warmup:
            warmup_llm(args)
        print("模型已预加载。开始测试；连续模式下按 Ctrl+C 退出。")
        for round_index in iter_rounds(args):
            print("\n=== 第 %d 轮 ===" % round_index)
            if args.input_text:
                transcript = args.input_text.strip()
                print("输入文本: %s" % transcript)
            else:
                transcript = transcribe(args, asr_model, round_index)
            if not transcript:
                print("没有识别到文本，跳过本轮。")
                continue
            had_text = True
            action, source = classify(args, transcript)
            print("判断结果: %s (来源: %s)" % (action, source))
            if tts_ctx is not None:
                speak_result(args, action, tts_ctx)
        if not had_text:
            return 2
    except KeyboardInterrupt:
        print("\n已退出连续测试。")
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

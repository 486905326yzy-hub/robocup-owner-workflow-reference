#!/usr/bin/env python3
import argparse
import os
import sys

import local_voice_test


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ZH_MODEL = os.path.join(ROOT_DIR, "models", "piper", "zh_CN-huayan-medium.onnx")
DEFAULT_ZH_CONFIG = os.path.join(ROOT_DIR, "models", "piper", "zh_CN-huayan-medium.onnx.json")
DEFAULT_WHISPER_ZH_MODEL = os.path.join(ROOT_DIR, "models", "whisper", "faster-whisper-small")
DEFAULT_MIC_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_SPEAKER_DEVICE = "plughw:CARD=Generic_1,DEV=0"


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone Chinese offline ASR/TTS test.")
    parser.add_argument("--mode", choices=["tts", "asr"], default="asr")
    parser.add_argument("--text", default="你好，这是离线中文语音播报测试。")
    parser.add_argument("--echo", action="store_true", help="识别后用中文播报识别结果")
    parser.add_argument("--no-play", action="store_true", help="生成音频但不播放")
    parser.add_argument("--output", help="TTS 输出 wav 路径")
    parser.add_argument("--keep-wav", action="store_true", help="打印并保留录音 wav 路径")
    parser.add_argument("--input-wav", help="识别已有 wav，不录音")

    parser.add_argument("--mic-device", default=DEFAULT_MIC_DEVICE)
    parser.add_argument("--speaker-device", default=DEFAULT_SPEAKER_DEVICE)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--energy-threshold", type=int, default=300)

    parser.add_argument("--model-size", default=DEFAULT_WHISPER_ZH_MODEL)
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="zh")

    parser.add_argument("--engine", choices=["auto", "piper", "flite"], default="auto")
    parser.add_argument("--piper", default=local_voice_test.DEFAULT_PIPER)
    parser.add_argument("--piper-model", default=DEFAULT_ZH_MODEL)
    parser.add_argument("--piper-config", default=DEFAULT_ZH_CONFIG)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--flite", default="flite")
    parser.add_argument("--flite-voice", default="slt")
    parser.add_argument("--player", default="aplay")
    return parser


def test_tts(args):
    wav_path = args.output or "/dev/shm/local_voice_tts_zh.wav"
    engine = local_voice_test.synthesize(args.text, wav_path, args)
    print("TTS engine: %s" % engine)
    print("WAV: %s" % wav_path)
    if not args.no_play:
        local_voice_test.play_wav(wav_path, args.player, args.speaker_device)


def test_asr(args):
    wav_path = args.input_wav or "/dev/shm/local_voice_record_zh.wav"
    if not args.input_wav:
        print("正在从 ALSA 设备 '%s' 录音 %.1f 秒..." % (args.mic_device, args.seconds))
        rms = local_voice_test.record_wav(wav_path, args.seconds, args.sample_rate, args.mic_device, args.energy_threshold)
        print("录音文件: %s (RMS=%d)" % (wav_path, rms))
    text = local_voice_test.transcribe_wav(wav_path, args.model_size, args.device, args.compute_type, args.language)
    print("识别文本: %s" % (text or "<空>"))
    if args.echo and text:
        reply = "我听到你说：" + text
        tts_wav = "/dev/shm/local_voice_echo_zh.wav"
        engine = local_voice_test.synthesize(reply, tts_wav, args)
        print("回放 TTS engine: %s" % engine)
        if not args.no_play:
            local_voice_test.play_wav(tts_wav, args.player, args.speaker_device)
    if args.keep_wav and not args.input_wav:
        print("保留录音: %s" % wav_path)


def main():
    args = build_parser().parse_args()
    local_voice_test.CURRENT_ARGS = args
    try:
        if args.mode == "tts":
            test_tts(args)
        else:
            test_asr(args)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import os
import sys
import time

import local_melo_tts_zh
import local_voice_test


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WHISPER_ZH_MODEL = os.path.join(ROOT_DIR, "models", "whisper", "faster-whisper-small")
DEFAULT_MIC_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_SPEAKER_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_RECORD_WAV = "/dev/shm/local_melo_echo_record_zh.wav"
DEFAULT_REPLY_WAV = "/dev/shm/local_melo_echo_reply_zh.wav"


def build_parser():
    parser = argparse.ArgumentParser(description="Chinese ASR + MeloTTS natural echo test.")
    parser.add_argument("--input-wav", help="识别已有 wav，不从麦克风录音")
    parser.add_argument("--record-wav", default=DEFAULT_RECORD_WAV, help="麦克风录音保存路径")
    parser.add_argument("--output", default=DEFAULT_REPLY_WAV, help="MeloTTS 回放 wav 输出路径")
    parser.add_argument("--loop", action="store_true", help="连续测试；ASR 和 MeloTTS 模型只加载一次，Ctrl+C 退出")
    parser.add_argument("--count", type=int, default=1, help="测试轮数；--loop 时忽略，0 表示一直循环")
    parser.add_argument("--keep-wav", action="store_true", help="打印并保留录音 wav 路径")
    parser.add_argument("--no-play", action="store_true", help="只生成回放 wav，不播放")
    parser.add_argument("--prefix", default="我听到你说：", help="回放前缀；设为空字符串可只播报识别文本")

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
    parser.add_argument("--beam-size", type=int, default=1, help="Whisper beam size；1 最快，5 更稳但更慢")
    parser.add_argument("--no-vad", action="store_true", help="关闭 faster-whisper VAD，固定短录音时可更快")
    parser.add_argument("--vad-filter", dest="no_vad", action="store_false", help="开启 faster-whisper VAD；环境噪声大时可试")
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.set_defaults(no_vad=True)

    parser.add_argument("--melo-language", default="ZH")
    parser.add_argument("--voice", default="ZH")
    parser.add_argument("--speed", type=float, default=1.0, help="MeloTTS 语速；小于 1 更慢，大于 1 更快")
    parser.add_argument("--melo-device", default="cpu")
    parser.add_argument("--melo-config", default=local_melo_tts_zh.DEFAULT_CONFIG)
    parser.add_argument("--melo-ckpt", default=local_melo_tts_zh.DEFAULT_CKPT)
    parser.add_argument("--no-warmup", action="store_true", help="关闭 MeloTTS 预热；默认会在录音前预热一次以降低说完后的等待")
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
    model = TTS(language=args.melo_language, device=args.melo_device, config_path=args.melo_config, ckpt_path=args.melo_ckpt)
    speaker_map = dict(model.hps.data.spk2id)
    if args.voice not in speaker_map:
        raise RuntimeError("voice '%s' not found. Available: %s" % (args.voice, ", ".join(sorted(speaker_map))))
    speaker_id = speaker_map[args.voice]
    print("MeloTTS 模型加载完成，用时 %.2f 秒，speaker_id=%s" % (time.time() - start, speaker_id))
    if not args.no_warmup:
        warmup_wav = "/dev/shm/local_melo_echo_warmup_zh.wav"
        print("正在预热 MeloTTS，后续复述会更快...")
        start = time.time()
        model.tts_to_file("预热。", speaker_id, warmup_wav, speed=args.speed, quiet=True)
        print("MeloTTS 预热完成，用时 %.2f 秒" % (time.time() - start))
    return model, speaker_id


def transcribe(args, asr_model, round_index=1):
    wav_path = args.input_wav or args.record_wav
    if not args.input_wav and (args.loop or args.count != 1):
        base, ext = os.path.splitext(args.record_wav)
        wav_path = "%s_%03d%s" % (base, round_index, ext or ".wav")
    if not args.input_wav:
        print("正在从 ALSA 设备 '%s' 录音 %.1f 秒..." % (args.mic_device, args.seconds))
        start = time.time()
        rms = local_voice_test.record_wav(
            wav_path,
            args.seconds,
            args.sample_rate,
            args.mic_device,
            args.energy_threshold,
        )
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


def speak_back(args, text, melo_model, speaker_id):
    reply = (args.prefix or "") + text
    start = time.time()
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    melo_model.tts_to_file(reply, speaker_id, args.output, speed=args.speed, quiet=True)
    print("TTS engine: MeloTTS")
    print("Voice: %s (speaker_id=%s)" % (args.voice, speaker_id))
    print("WAV: %s" % args.output)
    print("MeloTTS 合成用时 %.2f 秒" % (time.time() - start))
    print("回放文本: %s" % reply)
    if not args.no_play:
        local_melo_tts_zh.play_wav(args.output, args.player, args.speaker_device)


def iter_rounds(args):
    if args.input_wav:
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
        asr_model = load_asr_model(args)
        melo_model, speaker_id = load_melo_model(args)
        print("模型已预加载。开始测试；连续模式下按 Ctrl+C 退出。")
        had_text = False
        for round_index in iter_rounds(args):
            print("\n=== 第 %d 轮 ===" % round_index)
            text = transcribe(args, asr_model, round_index)
            if not text:
                print("没有识别到文本，跳过播报。")
                continue
            had_text = True
            speak_back(args, text, melo_model, speaker_id)
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

#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MELO_ROOT = os.path.join(ROOT_DIR, "third_party", "MeloTTS")
DEFAULT_CONFIG = os.path.join(ROOT_DIR, "models", "melo", "zh", "config_no_bert.json")
DEFAULT_CKPT = os.path.join(ROOT_DIR, "models", "melo", "zh", "checkpoint.pth")
DEFAULT_OUTPUT = "/dev/shm/melo_tts_zh.wav"
DEFAULT_SPEAKER_DEVICE = "plughw:CARD=Generic_1,DEV=0"


VOICE_TYPES = [
    ("ZH", "中文普通话", "MeloTTS Chinese official voice; good for Chinese and light Chinese-English mixed text."),
]


def add_melo_to_path():
    if not os.path.isdir(MELO_ROOT):
        raise RuntimeError("MeloTTS source is missing: %s" % MELO_ROOT)
    if MELO_ROOT not in sys.path:
        sys.path.insert(0, MELO_ROOT)


def play_wav(wav_path, player, speaker_device):
    cmd = [player]
    if speaker_device:
        cmd.extend(["-D", speaker_device])
    cmd.append(wav_path)
    subprocess.run(cmd, check=True)


def list_voices():
    print("Available MeloTTS voice types in this local install:")
    for key, label, note in VOICE_TYPES:
        print("  %s\t%s\t%s" % (key, label, note))
    print("\nInstalled local model:")
    print("  config: %s" % DEFAULT_CONFIG)
    print("  ckpt:   %s" % DEFAULT_CKPT)


def synthesize(args):
    add_melo_to_path()
    from melo.api import TTS

    if not os.path.exists(args.config):
        raise RuntimeError("Melo config is missing: %s" % args.config)
    if not os.path.exists(args.ckpt):
        raise RuntimeError("Melo checkpoint is missing: %s" % args.ckpt)

    model = TTS(language=args.language, device=args.device, config_path=args.config, ckpt_path=args.ckpt)
    speaker_map = dict(model.hps.data.spk2id)
    if args.voice not in speaker_map:
        raise RuntimeError("voice '%s' not found. Available: %s" % (args.voice, ", ".join(sorted(speaker_map))))
    speaker_id = speaker_map[args.voice]

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    model.tts_to_file(args.text, speaker_id, args.output, speed=args.speed, quiet=True)
    print("TTS engine: MeloTTS")
    print("Voice: %s (speaker_id=%s)" % (args.voice, speaker_id))
    print("WAV: %s" % args.output)


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone MeloTTS Chinese natural voice test.")
    parser.add_argument("--mode", choices=["tts", "voices"], default="tts")
    parser.add_argument("--text", default="你好，这是更自然的中文离线语音播报测试。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--player", default="aplay")
    parser.add_argument("--speaker-device", default=DEFAULT_SPEAKER_DEVICE)

    parser.add_argument("--language", default="ZH")
    parser.add_argument("--voice", default="ZH")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed; lower is slower, higher is faster.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda if a compatible GPU environment is ready.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        if args.mode == "voices":
            list_voices()
            return 0
        synthesize(args)
        if not args.no_play:
            play_wav(args.output, args.player, args.speaker_device)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

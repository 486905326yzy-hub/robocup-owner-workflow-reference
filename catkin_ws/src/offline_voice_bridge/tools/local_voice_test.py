#!/usr/bin/env python3
import argparse
import audioop
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PIPER = os.path.join(ROOT_DIR, "tools", "piper", "piper")
DEFAULT_PIPER_MODEL = os.path.join(ROOT_DIR, "models", "piper", "en_US-lessac-medium.onnx")
DEFAULT_PIPER_CONFIG = os.path.join(ROOT_DIR, "models", "piper", "en_US-lessac-medium.onnx.json")


def command_exists(command):
    if os.path.sep in command:
        return os.path.exists(command) and os.access(command, os.X_OK)
    return shutil.which(command) is not None


def record_wav(path, seconds, sample_rate, device, energy_threshold):
    raw_bytes = int(seconds * sample_rate * 2)
    arecord_seconds = max(1, int(math.ceil(seconds)))
    cmd = [
        "arecord",
        "-q",
        "-D",
        device,
        "-d",
        str(arecord_seconds),
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-t",
        "raw",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=arecord_seconds + 5)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip() or "arecord failed")
    raw = proc.stdout[:raw_bytes]
    if not raw:
        raise RuntimeError("arecord returned no audio")
    rms = audioop.rms(raw, 2)
    if rms < energy_threshold:
        print("WARN: recorded audio RMS is low (%d < %d). Check microphone input." % (rms, energy_threshold))

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw)
    return rms


def transcribe_wav(path, model_size, device, compute_type, language):
    if CURRENT_ARGS.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", CURRENT_ARGS.hf_endpoint)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run tools/setup_offline_voice.sh first. %s" % exc)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        path,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize(text, output_wav, args):
    if args.engine in ("auto", "piper") and piper_ready(args):
        cmd = [
            args.piper,
            "--model",
            args.piper_model,
            "--output_file",
            output_wav,
            "--length_scale",
            str(args.length_scale),
            "--noise_scale",
            str(args.noise_scale),
            "--noise_w",
            str(args.noise_w),
        ]
        if args.piper_config and os.path.exists(args.piper_config):
            cmd.extend(["--config", args.piper_config])
        subprocess.run(cmd, input=text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "piper"

    if args.engine == "piper":
        raise RuntimeError("Piper was selected but binary/model/config is not ready")

    if command_exists(args.flite):
        cmd = [args.flite, "-voice", args.flite_voice, "-t", text, "-o", output_wav]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "flite"

    raise RuntimeError("No TTS engine available. Install Piper or keep flite installed.")


def piper_ready(args):
    return command_exists(args.piper) and os.path.exists(args.piper_model) and os.path.exists(args.piper_config)


def play_wav(path, player, speaker_device=None):
    if not command_exists(player):
        raise RuntimeError("Audio player not found: %s" % player)
    cmd = [player, "-q"]
    if speaker_device and os.path.basename(player) == "aplay":
        cmd.extend(["-D", speaker_device])
    cmd.append(path)
    subprocess.run(cmd, check=True)


def test_tts(args):
    wav_path = args.output or "/dev/shm/local_voice_tts.wav"
    engine = synthesize(args.text, wav_path, args)
    print("TTS engine: %s" % engine)
    print("WAV: %s" % wav_path)
    if not args.no_play:
        play_wav(wav_path, args.player, args.speaker_device)


def test_asr(args):
    wav_path = args.input_wav or "/dev/shm/local_voice_record.wav"
    if not args.input_wav:
        print("Recording %.1f seconds from ALSA device '%s'..." % (args.seconds, args.mic_device))
        rms = record_wav(wav_path, args.seconds, args.sample_rate, args.mic_device, args.energy_threshold)
        print("Recorded WAV: %s (RMS=%d)" % (wav_path, rms))
    text = transcribe_wav(wav_path, args.model_size, args.device, args.compute_type, args.language)
    print("ASR text: %s" % (text or "<empty>"))
    if args.echo and text:
        reply = "I heard: " + text
        tts_wav = "/dev/shm/local_voice_echo.wav"
        engine = synthesize(reply, tts_wav, args)
        print("Echo TTS engine: %s" % engine)
        if not args.no_play:
            play_wav(tts_wav, args.player, args.speaker_device)
    if args.keep_wav and not args.input_wav:
        print("Kept recording: %s" % wav_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone offline microphone/speaker test for offline_voice_bridge.")
    parser.add_argument("--mode", choices=["tts", "asr"], default="asr")
    parser.add_argument("--text", default="Hello. This is an offline text to speech test.")
    parser.add_argument("--echo", action="store_true", help="Speak back the recognized text")
    parser.add_argument("--no-play", action="store_true", help="Generate audio but do not play it")
    parser.add_argument("--output", help="Output wav path for --mode tts")
    parser.add_argument("--keep-wav", action="store_true", help="Print and keep the recorded wav path")
    parser.add_argument("--input-wav", help="Transcribe an existing wav instead of recording")

    parser.add_argument("--mic-device", default="default")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--energy-threshold", type=int, default=300)

    parser.add_argument("--model-size", default="small.en")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="en")

    parser.add_argument("--engine", choices=["auto", "piper", "flite"], default="auto")
    parser.add_argument("--piper", default=DEFAULT_PIPER)
    parser.add_argument("--piper-model", default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-config", default=DEFAULT_PIPER_CONFIG)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--flite", default="flite")
    parser.add_argument("--flite-voice", default="slt")
    parser.add_argument("--player", default="aplay")
    parser.add_argument("--speaker-device", help="ALSA playback device for aplay, for example plughw:CARD=Generic_1,DEV=0")
    return parser


def main():
    global CURRENT_ARGS
    args = build_parser().parse_args()
    CURRENT_ARGS = args
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
    CURRENT_ARGS = None
    sys.exit(main())

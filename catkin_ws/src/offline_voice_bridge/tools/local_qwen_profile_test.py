#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

import local_voice_test


DEFAULT_MIC_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_SPEAKER_DEVICE = "plughw:CARD=Generic_1,DEV=0"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_QWEN_MODEL = "qwen3.5:9b"


def record_and_transcribe(args):
    if args.input_text:
        return args.input_text.strip(), None

    wav_path = args.input_wav or "/dev/shm/local_qwen_profile_record.wav"
    if not args.input_wav:
        print("Recording %.1f seconds from ALSA device '%s'..." % (args.seconds, args.mic_device))
        rms = local_voice_test.record_wav(
            wav_path,
            args.seconds,
            args.sample_rate,
            args.mic_device,
            args.energy_threshold,
        )
        print("Recorded WAV: %s (RMS=%d)" % (wav_path, rms))

    text = local_voice_test.transcribe_wav(
        wav_path,
        args.model_size,
        args.device,
        args.compute_type,
        args.language,
    )
    return text.strip(), wav_path


def call_qwen(args, transcript):
    user_prompt = (
        "/no_think\n"
        "Fast extraction. Return JSON only. No explanation.\n"
        'Schema: {"name": string|null, "drink": string|null}\n'
        "Extract the speaker's stated name and favorite drink from this English transcript.\n"
        "Transcript: %s" % transcript
    )
    payload = {
        "model": args.llm_model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": args.llm_keep_alive,
        "messages": [{"role": "user", "content": user_prompt}],
        "options": {
            "temperature": 0,
            "top_p": 0.7,
            "num_predict": args.llm_max_tokens,
            "num_ctx": 1024,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.llm_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise RuntimeError("Qwen request timed out. First run may load the model; retry or increase --llm-timeout. %s" % exc)
    except urllib.error.URLError as exc:
        raise RuntimeError("Qwen request failed: %s" % exc)

    result = json.loads(raw)
    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Qwen returned empty content: %s" % raw)
    return parse_qwen_json(content)


def parse_qwen_json(content):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise RuntimeError("Qwen did not return JSON: %s" % content)
        parsed = json.loads(match.group(0))

    name = normalize_field(parsed.get("name"))
    drink = normalize_field(parsed.get("drink"))
    if name and drink:
        greeting = "Hello %s, nice to meet you. Your favorite drink is %s." % (name, drink)
    elif name:
        greeting = "Hello %s, nice to meet you. I could not catch your favorite drink." % name
    elif drink:
        greeting = "Hello friend, nice to meet you. Your favorite drink is %s." % drink
    else:
        greeting = "Hello friend, nice to meet you. I could not catch your name or favorite drink."
    return {"name": name, "drink": drink, "greeting": greeting}


def normalize_field(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "unknown", "not found"):
        return None
    return text


def speak_result(args, result):
    wav_path = args.output_wav or "/dev/shm/local_qwen_profile_reply.wav"
    engine = local_voice_test.synthesize(result["greeting"], wav_path, args)
    print("TTS engine: %s" % engine)
    print("Reply WAV: %s" % wav_path)
    if not args.no_play:
        local_voice_test.play_wav(wav_path, args.player, args.speaker_device)


def build_parser():
    parser = argparse.ArgumentParser(description="Local ASR + Qwen extraction + TTS profile test.")
    parser.add_argument("--input-text", help="Skip microphone/ASR and analyze this text directly")
    parser.add_argument("--input-wav", help="Skip microphone and transcribe an existing wav")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--mic-device", default=DEFAULT_MIC_DEVICE)
    parser.add_argument("--speaker-device", default=DEFAULT_SPEAKER_DEVICE)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--energy-threshold", type=int, default=300)
    parser.add_argument("--keep-wav", action="store_true")

    parser.add_argument("--model-size", default="small.en")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="en")

    parser.add_argument("--llm-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llm-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=60)
    parser.add_argument("--llm-keep-alive", default="10m")

    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--output-wav")
    parser.add_argument("--engine", choices=["auto", "piper", "flite"], default="auto")
    parser.add_argument("--piper", default=local_voice_test.DEFAULT_PIPER)
    parser.add_argument("--piper-model", default=local_voice_test.DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-config", default=local_voice_test.DEFAULT_PIPER_CONFIG)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--flite", default="flite")
    parser.add_argument("--flite-voice", default="slt")
    parser.add_argument("--player", default="aplay")
    return parser


def main():
    args = build_parser().parse_args()
    local_voice_test.CURRENT_ARGS = args
    try:
        transcript, wav_path = record_and_transcribe(args)
        print("ASR text: %s" % (transcript or "<empty>"))
        if not transcript:
            raise RuntimeError("No transcript to analyze")

        result = call_qwen(args, transcript)
        print("Name: %s" % (result["name"] or "<missing>"))
        print("Drink: %s" % (result["drink"] or "<missing>"))
        print("Greeting: %s" % result["greeting"])
        speak_result(args, result)
        if args.keep_wav and wav_path and not args.input_wav:
            print("Kept recording: %s" % wav_path)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

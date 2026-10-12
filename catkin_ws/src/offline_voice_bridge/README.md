# offline_voice_bridge

Independent ROS1 package for replacing the xfyun speech stack with offline ASR and TTS. The original English tests stay unchanged; Chinese tests are added as separate `_zh` files.

It does not modify `xfyun_waterplus` or any Waterplus examples. It keeps compatible topics so existing demos can be migrated gradually:

- ASR publishes to `/voice/asr_text` and legacy `/xfyun/iat`.
- TTS subscribes to `/voice/say` and legacy `/xfyun/tts`.
- Audio playback still uses `sound_play` on `/robotsound`.

## Setup

```bash
cd ~/catkin_ws/src/offline_voice_bridge
bash tools/setup_offline_voice.sh
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

The setup script downloads Piper voices from `hf-mirror.com` by default because direct `huggingface.co` access can be slow. Override it with `PIPER_VOICE_BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium` if needed.

Chinese setup is separate and does not overwrite the English files:

```bash
cd ~/catkin_ws/src/offline_voice_bridge
bash tools/setup_chinese_voice.sh
cd ~/catkin_ws
catkin_make --pkg offline_voice_bridge
source devel/setup.bash
```

## TTS smoke test

```bash
roslaunch offline_voice_bridge offline_tts_demo.launch
```

## Local microphone/speaker test without ROS

Use this when the robot is not available:

```bash
cd ~/catkin_ws/src/offline_voice_bridge
python3 tools/local_voice_test.py --mode tts --text "Hello, this is a speaker test."
python3 tools/local_voice_test.py \
  --mode asr \
  --mic-device plughw:CARD=Generic_1,DEV=0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0 \
  --seconds 4 \
  --echo \
  --model-size small.en \
  --keep-wav
```

The local test uses `arecord` for the microphone and `aplay` for speakers. It uses Piper when available and falls back to `flite` for basic offline playback.
Whisper model downloads use `https://hf-mirror.com` by default; override with `--hf-endpoint` if your network prefers another endpoint.

## Local Chinese microphone/speaker test without ROS

Use the separate Chinese script. It defaults to the validated local ALSA devices and the local multilingual Whisper model under `models/whisper/faster-whisper-small`.

```bash
cd ~/catkin_ws/src/offline_voice_bridge
python3 tools/local_voice_test_zh.py \
  --mode tts \
  --text "你好，这是中文离线语音播报测试。"
```

Chinese ASR with echo playback:

```bash
python3 tools/local_voice_test_zh.py \
  --mode asr \
  --seconds 4 \
  --echo \
  --keep-wav
```

Equivalent explicit device command:

```bash
python3 tools/local_voice_test_zh.py \
  --mode asr \
  --mic-device plughw:CARD=Generic_1,DEV=0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0 \
  --seconds 4 \
  --echo \
  --keep-wav
```

Quick offline ASR check from an existing wav:

```bash
python3 tools/local_voice_test_zh.py \
  --mode asr \
  --input-wav /tmp/chinese_tts_test.wav \
  --no-play
```

## More natural Chinese TTS with MeloTTS

Piper stays as the lightweight default. For a more human-like Chinese voice, use the separate MeloTTS path:

```bash
cd ~/catkin_ws/src/offline_voice_bridge
bash tools/setup_melo_tts.sh
python3 tools/local_melo_tts_zh.py --mode voices
python3 tools/local_melo_tts_zh.py \
  --mode tts \
  --text "你好，这是更自然的中文离线语音播报测试。"
```

Generate only, without playback:

```bash
python3 tools/local_melo_tts_zh.py \
  --mode tts \
  --text "你好，这是更自然的中文离线语音播报测试。" \
  --no-play \
  --output /tmp/melo_zh_test.wav
```

ROS MeloTTS demo and full bridge:

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch offline_voice_bridge offline_tts_demo_zh_melo.launch
roslaunch offline_voice_bridge offline_voice_zh_melo.launch
```

Installed voice type in this workspace:

```text
ZH  Chinese Mandarin / Chinese-English mixed text  / MeloTTS Chinese official voice
```

Upstream MeloTTS also provides `EN-Default`, `EN-US`, `EN-BR`, `EN_INDIA`, `EN-AU`, `ES`, `FR`, `JP`, and `KR`. Those are not installed here yet; this workspace currently installs only `ZH` to keep the robot runtime smaller.

## Chinese ASR + natural echo test

Use this to speak Chinese into the computer microphone, transcribe it with faster-whisper, then speak the recognized text back with MeloTTS:

```bash
cd ~/catkin_ws/src/offline_voice_bridge
python3 tools/local_melo_echo_zh.py \
  --seconds 4 \
  --loop \
  --keep-wav
```

Explicit validated devices:

```bash
python3 tools/local_melo_echo_zh.py \
  --mic-device plughw:CARD=Generic_1,DEV=0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0 \
  --seconds 4 \
  --loop \
  --keep-wav
```

Generate the echo wav without playback:

```bash
python3 tools/local_melo_echo_zh.py \
  --seconds 4 \
  --no-play \
  --output /tmp/melo_echo_zh_test.wav
```

For lower latency this script preloads ASR/TTS before recording, warms up MeloTTS once, uses `beam-size=1`, and disables VAD by default. Add `--vad-filter` if the room is noisy.

## Local Qwen profile voice test without ROS

Use this to say your name and favorite drink in English, then let local Qwen extract them and speak a greeting back.

```bash
cd ~/catkin_ws/src/offline_voice_bridge
python3 tools/local_qwen_profile_test.py \
  --mic-device plughw:CARD=Generic_1,DEV=0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0 \
  --seconds 6 \
  --model-size small.en \
  --keep-wav
```

Qwen defaults to Ollama at `http://127.0.0.1:11434/api/chat` with model `qwen3.5:9b`. The request uses `think:false`, `/no_think`, JSON-only extraction, temperature `0`, short `num_predict`, and `keep_alive:10m` to reduce response latency. The script builds the spoken greeting itself after Qwen extracts only `name` and `drink`.

Quick text-only check without microphone:

```bash
python3 tools/local_qwen_profile_test.py \
  --input-text "My name is Alice, and honestly I drink iced coffee more than anything else." \
  --no-play
```

## Local Qwen vision face description test without ROS

Open the computer camera, automatically capture a frame after a short warmup, let local Qwen describe visible non-sensitive facial appearance features, and speak the result with Piper.

```bash
cd ~/catkin_ws/src/offline_voice_bridge
python3 tools/local_qwen_vision_face_test.py \
  --camera-index 0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0
```

Use manual capture if needed:

```bash
python3 tools/local_qwen_vision_face_test.py \
  --camera-index 0 \
  --speaker-device plughw:CARD=Generic_1,DEV=0 \
  --manual
```

Use an existing image instead of the camera:

```bash
python3 tools/local_qwen_vision_face_test.py \
  --image /path/to/person.jpg \
  --no-play
```

Or publish text manually:

```bash
roslaunch offline_voice_bridge offline_voice.launch start_asr:=false
rostopic pub /voice/say std_msgs/String "data: 'Hello. This is offline Piper text to speech.'" -1
```

## Full offline voice bridge

```bash
roslaunch offline_voice_bridge offline_voice.launch
```

Speak English into the microphone. Recognized text is published to `/voice/asr_text` and `/xfyun/iat`.

Chinese ROS launch files are separate from the English launch files:

```bash
roslaunch offline_voice_bridge offline_tts_demo_zh.launch
roslaunch offline_voice_bridge offline_voice_zh.launch
roslaunch offline_voice_bridge offline_tts_demo_zh_melo.launch
roslaunch offline_voice_bridge offline_voice_zh_melo.launch
```

The Chinese config uses `language: zh`, Piper `zh_CN-huayan-medium`, and the local multilingual faster-whisper model path.
The Melo config uses `natural_tts_node.py`, `models/melo/zh/checkpoint.pth`, and `/dev/shm/offline_tts_melo_zh.wav`.

The ASR node records from ALSA with `arecord`, so it does not need PyAudio. If your microphone is not the default ALSA device, set `/asr/capture_device` in `config/offline_voice.yaml`.

## Notes

- Default ASR model is `small.en` on CPU with int8 compute. Change `/asr/model_size` to `medium.en` if accuracy is more important than latency.
- Default TTS voice is Piper `en_US-lessac-medium`, stored under `models/piper/`.
- Chinese ASR/TTS is kept in `tools/local_voice_test_zh.py`, `config/offline_voice_zh.yaml`, `launch/offline_voice_zh.launch`, and `launch/offline_tts_demo_zh.launch`.
- Natural Chinese TTS is kept separately in `tools/local_melo_tts_zh.py`, `scripts/natural_tts_node.py`, `config/offline_voice_zh_melo.yaml`, `launch/offline_voice_zh_melo.launch`, and `launch/offline_tts_demo_zh_melo.launch`.
- If Piper is not downloaded yet, the TTS node can fall back to the system `flite` command for a basic offline smoke test. Set `/tts/engine: piper` to require Piper only.
- The old xfyun package can remain installed for rollback. Do not launch xfyun ASR/TTS at the same time if both publish/subscribe legacy topics.

#!/usr/bin/env python3
import os
import subprocess
import wave
import audioop

import rospy
from std_msgs.msg import String


class OfflineAsrNode:
    def __init__(self):
        self.publish_topic = rospy.get_param("~publish_topic", rospy.get_param("/asr/publish_topic", "/voice/asr_text"))
        self.legacy_publish_topic = rospy.get_param(
            "~legacy_publish_topic", rospy.get_param("/asr/legacy_publish_topic", "/xfyun/iat")
        )
        self.model_size = rospy.get_param("~model_size", rospy.get_param("/asr/model_size", "small.en"))
        self.hf_endpoint = rospy.get_param("~hf_endpoint", rospy.get_param("/asr/hf_endpoint", "https://hf-mirror.com"))
        self.device = rospy.get_param("~device", rospy.get_param("/asr/device", "cpu"))
        self.compute_type = rospy.get_param("~compute_type", rospy.get_param("/asr/compute_type", "int8"))
        self.language = rospy.get_param("~language", rospy.get_param("/asr/language", "en"))
        self.sample_rate = int(rospy.get_param("~sample_rate", rospy.get_param("/asr/sample_rate", 16000)))
        self.chunk_seconds = float(rospy.get_param("~chunk_seconds", rospy.get_param("/asr/chunk_seconds", 4.0)))
        self.overlap_seconds = float(rospy.get_param("~overlap_seconds", rospy.get_param("/asr/overlap_seconds", 0.4)))
        self.min_voice_seconds = float(rospy.get_param("~min_voice_seconds", rospy.get_param("/asr/min_voice_seconds", 0.6)))
        self.energy_threshold = int(rospy.get_param("~energy_threshold", rospy.get_param("/asr/energy_threshold", 300)))
        self.capture_device = rospy.get_param("~capture_device", rospy.get_param("/asr/capture_device", "default"))

        self.pub = rospy.Publisher(self.publish_topic, String, queue_size=10)
        self.legacy_pub = None
        if self.legacy_publish_topic and self.legacy_publish_topic != self.publish_topic:
            self.legacy_pub = rospy.Publisher(self.legacy_publish_topic, String, queue_size=10)

        self._load_runtime()

    def _load_runtime(self):
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing offline ASR dependencies. Run tools/setup_offline_voice.sh first. Detail: %s" % exc
            )

        rospy.loginfo("Loading faster-whisper model %s (%s/%s)", self.model_size, self.device, self.compute_type)
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def run(self):
        bytes_per_sample = 2
        chunk_bytes = int(self.sample_rate * self.chunk_seconds * bytes_per_sample)
        overlap_bytes = int(self.sample_rate * self.overlap_seconds * bytes_per_sample)
        min_voice_bytes = int(self.sample_rate * self.min_voice_seconds * bytes_per_sample)
        overlap = b""

        cmd = [
            "arecord",
            "-q",
            "-D",
            self.capture_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            "1",
            "-t",
            "raw",
        ]
        rospy.loginfo("offline_asr_node ready: %s -> %s", " ".join(cmd), self.publish_topic)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while not rospy.is_shutdown():
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                    raise RuntimeError("arecord stopped: %s" % stderr.strip())
                phrase = overlap + raw
                overlap = phrase[-overlap_bytes:] if overlap_bytes > 0 else b""
                if len(phrase) < min_voice_bytes or audioop.rms(phrase, bytes_per_sample) < self.energy_threshold:
                    continue
                text = self._transcribe_raw(phrase)
                if text:
                    self._publish_text(text)
        finally:
            proc.terminate()

    def _publish_text(self, text):
        msg = String(data=text)
        self.pub.publish(msg)
        if self.legacy_pub:
            self.legacy_pub.publish(msg)
        rospy.loginfo("Recognized: %s", text)

    def _transcribe_raw(self, raw_audio):
        wav_path = "/dev/shm/offline_asr_phrase.wav"
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_audio)

        segments, info = self.model.transcribe(
            wav_path,
            language=self.language,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if len(text) < 2:
            return ""
        return text


if __name__ == "__main__":
    rospy.init_node("offline_asr_node")
    node = OfflineAsrNode()
    node.run()

#!/usr/bin/env python3
import os
import sys
import tempfile
import threading
import time
import wave

import actionlib
import rospy
from sound_play.msg import SoundRequest, SoundRequestAction, SoundRequestGoal
from std_msgs.msg import String


class NaturalTtsNode:
    def __init__(self):
        self.subscribe_topic = rospy.get_param("~subscribe_topic", rospy.get_param("/tts/subscribe_topic", "/voice/say"))
        self.legacy_subscribe_topic = rospy.get_param(
            "~legacy_subscribe_topic", rospy.get_param("/tts/legacy_subscribe_topic", "/xfyun/tts")
        )
        self.sound_topic = rospy.get_param("~sound_topic", rospy.get_param("/tts/sound_topic", "/robotsound"))
        self.wav_path = rospy.get_param("~wav_path", rospy.get_param("/tts/wav_path", "/dev/shm/offline_tts_melo_zh.wav"))
        self.volume = float(rospy.get_param("~volume", rospy.get_param("/tts/volume", 1.0)))
        self.sound_wait_timeout = float(rospy.get_param("~sound_wait_timeout", rospy.get_param("/tts/sound_wait_timeout", 10.0)))
        self.sound_action = rospy.get_param("~sound_action", rospy.get_param("/tts/sound_action", "sound_play"))
        self.blocking_playback = bool(rospy.get_param("~blocking_playback", rospy.get_param("/tts/blocking_playback", True)))
        self.playback_margin = float(rospy.get_param("~playback_margin", rospy.get_param("/tts/playback_margin", 0.6)))
        self.wav_keep_count = max(1, int(rospy.get_param("~wav_keep_count", rospy.get_param("/tts/wav_keep_count", 8))))

        self.language = rospy.get_param("~language", rospy.get_param("/tts/language", "ZH"))
        self.voice = rospy.get_param("~voice", rospy.get_param("/tts/voice", "ZH"))
        self.device = rospy.get_param("~device", rospy.get_param("/tts/device", "cpu"))
        self.speed = float(rospy.get_param("~speed", rospy.get_param("/tts/speed", 1.0)))
        self.config_path = rospy.get_param("~config_path", rospy.get_param("/tts/config_path", ""))
        self.ckpt_path = rospy.get_param("~ckpt_path", rospy.get_param("/tts/ckpt_path", ""))
        self.third_party_path = rospy.get_param("~third_party_path", rospy.get_param("/tts/third_party_path", ""))

        self._lock = threading.Lock()
        self._tts = None
        self._speaker_id = None
        self._sequence = 0
        self._generated_wavs = []

        self.play_pub = rospy.Publisher(self.sound_topic, SoundRequest, queue_size=10)
        self.action_client = actionlib.SimpleActionClient(self.sound_action, SoundRequestAction)
        self._load_model()

        rospy.Subscriber(self.subscribe_topic, String, self._speak_callback, queue_size=10)
        if self.legacy_subscribe_topic and self.legacy_subscribe_topic != self.subscribe_topic:
            rospy.Subscriber(self.legacy_subscribe_topic, String, self._speak_callback, queue_size=10)

        rospy.loginfo(
            "natural_tts_node ready: %s and %s -> %s, voice=%s",
            self.subscribe_topic,
            self.legacy_subscribe_topic,
            self.sound_topic,
            self.voice,
        )

    def _load_model(self):
        if self.third_party_path:
            if not os.path.isdir(self.third_party_path):
                raise RuntimeError("MeloTTS third_party_path is missing: %s" % self.third_party_path)
            if self.third_party_path not in sys.path:
                sys.path.insert(0, self.third_party_path)
        if not self.config_path or not os.path.exists(self.config_path):
            raise RuntimeError("MeloTTS config_path is missing: %s" % self.config_path)
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise RuntimeError("MeloTTS ckpt_path is missing: %s" % self.ckpt_path)

        from melo.api import TTS

        self._tts = TTS(language=self.language, device=self.device, config_path=self.config_path, ckpt_path=self.ckpt_path)
        speaker_map = dict(self._tts.hps.data.spk2id)
        if self.voice not in speaker_map:
            raise RuntimeError("voice '%s' not found. Available: %s" % (self.voice, ", ".join(sorted(speaker_map))))
        self._speaker_id = speaker_map[self.voice]
        rospy.loginfo("Using MeloTTS natural TTS engine, speaker_id=%s", self._speaker_id)

    def _speak_callback(self, msg):
        text = msg.data.strip()
        if not text:
            return
        with self._lock:
            try:
                wav_path = self._next_wav_path()
                self._synthesize(text, wav_path)
                self._play_wav(wav_path)
                self._remember_wav(wav_path)
            except Exception as exc:
                rospy.logerr("Natural TTS failed: %s", exc)

    def _next_wav_path(self):
        wav_dir = os.path.dirname(self.wav_path) or "."
        os.makedirs(wav_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.wav_path))[0] or "natural_tts"
        self._sequence += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(wav_dir, "%s_%s_%03d.wav" % (base, stamp, self._sequence))

    def _remember_wav(self, wav_path):
        self._generated_wavs.append(wav_path)
        while len(self._generated_wavs) > self.wav_keep_count:
            old_path = self._generated_wavs.pop(0)
            if old_path != wav_path and os.path.exists(old_path):
                try:
                    os.unlink(old_path)
                except OSError:
                    pass

    @staticmethod
    def _wav_duration(wav_path):
        try:
            with wave.open(wav_path, "rb") as wav_file:
                rate = float(wav_file.getframerate())
                if rate <= 0:
                    return 0.0
                return wav_file.getnframes() / rate
        except Exception:
            return 0.0

    def _synthesize(self, text, wav_path):
        wav_dir = os.path.dirname(wav_path) or "."
        os.makedirs(wav_dir, exist_ok=True)
        fd, tmp_wav = tempfile.mkstemp(prefix="natural_tts_", suffix=".wav", dir=wav_dir)
        os.close(fd)
        try:
            self._tts.tts_to_file(text, self._speaker_id, tmp_wav, speed=self.speed, quiet=True)
            os.replace(tmp_wav, wav_path)
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def _make_sound_request(self, wav_path):
        req = SoundRequest()
        req.sound = SoundRequest.PLAY_FILE
        req.command = SoundRequest.PLAY_ONCE
        req.volume = self.volume
        req.arg = wav_path
        return req

    def _play_wav(self, wav_path):
        req = self._make_sound_request(wav_path)
        duration = self._wav_duration(wav_path)
        wait_duration = max(0.6, duration + self.playback_margin)

        if self.blocking_playback:
            if self.action_client.wait_for_server(rospy.Duration(self.sound_wait_timeout)):
                goal = SoundRequestGoal()
                goal.sound_request = req
                self.action_client.send_goal(goal)
                if self.action_client.wait_for_result(rospy.Duration(wait_duration + self.sound_wait_timeout)):
                    rospy.loginfo("Spoken with MeloTTS: %s (%.2fs, blocking)", wav_path, duration)
                    return
                self.action_client.cancel_goal()
                rospy.logwarn("Timed out waiting for sound_play action; falling back to topic playback")
            else:
                rospy.logwarn("sound_play action [%s] is unavailable; falling back to topic playback", self.sound_action)

        deadline = time.time() + self.sound_wait_timeout
        while not rospy.is_shutdown() and self.play_pub.get_num_connections() == 0 and time.time() < deadline:
            time.sleep(0.05)
        if self.play_pub.get_num_connections() == 0:
            rospy.logwarn("No subscribers connected to %s; synthesized speech may not play", self.sound_topic)
        self.play_pub.publish(req)
        time.sleep(wait_duration)
        rospy.loginfo("Spoken with MeloTTS: %s (%.2fs, topic fallback)", wav_path, duration)


if __name__ == "__main__":
    rospy.init_node("natural_tts_node")
    NaturalTtsNode()
    rospy.spin()
